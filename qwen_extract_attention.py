import argparse
import torch
import os
import json
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from pathlib import Path

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        qid = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        return qid, qs, image_file

    def __len__(self):
        return len(self.questions)


# def collate_fn(batch):
#     input_ids, image_tensors, image_sizes, prompts = zip(*batch)
#     input_ids = torch.stack(input_ids, dim=0)
#     image_tensors = torch.stack(image_tensors, dim=0)
#     return input_ids, image_tensors, image_sizes, prompts


# DataLoader
def create_data_loader(questions, image_folder,batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    processor = AutoProcessor.from_pretrained(os.path.join(args.model_path, "Qwen2.5-VL-7B-Instruct"))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(args.model_path, "Qwen2.5-VL-7B-Instruct"), torch_dtype="auto", device_map="auto"
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    data_loader = create_data_loader(questions, args.image_folder)
    output_dir = args.output_dir
    output_tensor_dir = os.path.join(output_dir, "attention_tensor")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_tensor_dir).mkdir(parents=True, exist_ok=True)
    
    result = []

        
    for (question_id, question, image_file) in tqdm(data_loader, total=len(questions)):
        idx = int(question_id)
        image_file = image_file[0]
        question = question[0]
        image = Image.open(os.path.join(args.image_folder, image_file))
        resized_image = image.resize((336, 336), Image.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": resized_image
                    },
                    {
                        "type": "text", 
                        "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        
        with torch.inference_mode():
            
            output = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)

            img_token_start_index = int(torch.where(output.sequences[0]==torch.tensor(151652))[0])+1
            img_token_end_index = int(torch.where(output.sequences[0]==torch.tensor(151653))[0])
        
            output_ids = output.sequences
            output_attentions = output.attentions
            
            output_attentions_image_index0 = torch.cat(output_attentions[0],dim=0)[:,:,-1,img_token_start_index:img_token_end_index].cpu()
            output_attentions_image_index1 = [torch.cat(output_attentions[i],dim=0).squeeze(2)[:,:,img_token_start_index:img_token_end_index].cpu() for i in range(1, len(output_attentions))]
            
            output_attentions = torch.cat([output_attentions_image_index0.unsqueeze(0), torch.stack(output_attentions_image_index1, dim=0)], dim=0) # torch.Size([22, 32, 32, 576])
           
            output_attentions = torch.mean(output_attentions, dim=0)
            
            output_text = processor.batch_decode(
                output_ids[:, len(inputs.input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            
            # for pope
            if output_text.lower()=="yes": # answer yes
                if idx%2==0: # gt no
                    y_label = 3
                else: # gt yes
                    y_label = 0
            elif output_text.lower()=="no": # answer no
                if idx%2==0: # gt no
                    y_label = 1
                else: # gt yes
                    y_label = 2
            else:
                print(idx, output_text)
                y_label = -1
            
            result.append({
                "attention_file": f"x_{idx}.tensor",
                "label": y_label
            })

            torch.save(output_attentions, os.path.join(output_tensor_dir, f"x_{idx}.tensor"))
    json.dump(result, open(os.path.join(output_dir, f"rank_{args.chunk_idx}.json"), "w"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="/root/nfs/dataset/val2014")
    parser.add_argument("--question-file", type=str, default="POPE-COCO.jsonl")
    parser.add_argument("--output-dir", type=str, default="attention_file/Qwen2.5-VL-7b_pope_coco")
    parser.add_argument("--model-path", type=str, default="/root/huggingface_model/")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
