import json
import os

# 存储合并后的数据
combined_data = []

res_train = []
res_test = []

for i in range(48):
    file_name = f"attention_file/Qwen2.5-VL-7b_pope_coco/rank_{i}.json"
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            for item in data:
                # if int(item["image_id"])>20000000:
                # if int(item["attention_file"][2:4])>20:
                if int(item["attention_file"].split("_")[1].split(".")[0])>200000000:
                    res_test.append(item)
                else:
                    res_train.append(item)

    except FileNotFoundError:
        print(f"File {file_name} not found.")

os.makedirs("dhcp_label_file", exist_ok=True)
with open('dhcp_label_file/Qwen2.5-VL-7b_pope_coco_test.jsonl', 'w') as jsonl_file:
    for item in res_test:
        jsonl_file.write(json.dumps(item) + "\n")

with open('dhcp_label_file/Qwen2.5-VL-7b_pope_coco_train.jsonl', 'w') as jsonl_file:
    for item in res_train:
        jsonl_file.write(json.dumps(item) + "\n")
        