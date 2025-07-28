import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import multiprocessing.shared_memory as shm
import numpy as np
import json
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import classification_report
import random
import argparse

# 定义 MLP 模型
class MLP_l2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_l2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)
        # print(x.type())
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class JSONLDataset(Dataset):
    def __init__(self, file_path, tensor_path):
        self.data = []
        self.tensor_path = tensor_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 未找到。")
        except json.JSONDecodeError:
            print(f"错误: 在解析 {file_path} 时发生 JSON 解码错误。")
            
        print("loading")
         # 预加载数据
        self.tensors = []
        self.labels = []
        for item in tqdm(self.data):
            tensor = torch.load(os.path.join(self.tensor_path, item["attention_file"]))
            self.tensors.append(tensor)
            self.labels.append(item["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--dhcp_path', type=str, required=True)
    args = parser.parse_args()
    
    test_dataset = JSONLDataset("dhcp_label_file/Qwen2.5-VL-7b_pope_coco_test.jsonl", "attention_file/Qwen2.5-VL-7b_pope/attention_tensor")
    train_loader = DataLoader(test_dataset, batch_size=1024, num_workers=4, shuffle=False)

    dhcp_labels = [int(i/2) for i in test_dataset.labels]

    model_weight = torch.load(args.dhcp_path)

    device = torch.device("cuda:{}".format(args.gpu))
    model = MLP_l2(28 * 28 * 144, 128, 2).to(device)
    model.load_state_dict(model_weight)

    res = []

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images.to(torch.float32))
        predict = torch.argmax(outputs, dim=1)
        res.append(predict)

    predict = torch.cat(res, dim=0).cpu()
        
    print(classification_report(dhcp_labels, predict, target_names=["no hallu", "hallu"], digits=4))