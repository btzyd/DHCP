import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import multiprocessing.shared_memory as shm
import numpy as np
import json
from tqdm import tqdm
import argparse
from torch.utils.data.sampler import WeightedRandomSampler
from pathlib import Path
from collections import Counter

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


def train(args):
    # 创建数据加载器
    
    device = torch.device(f"cuda:{args.gpu}")
    
    train_dataset = JSONLDataset("dhcp_label_file/Qwen2.5-VL-7b_pope_coco_train.jsonl", "attention_file/Qwen2.5-VL-7b_pope_coco/attention_tensor")


    train_dataset.labels = [int(i/2) for i in train_dataset.labels]
        
    if args.sampler:
        counter = Counter(train_dataset.labels)
        sorted_items = sorted(counter.items())
        sorted_items = [i[1] for i in sorted_items]
        class_sample_count = torch.tensor(sorted_items)
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in train_dataset.labels])
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=4, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    
    # 初始化模型
    print("init model")
    input_size = 28 * 28 * 144
    
    num_classes = 2
    
    model = MLP_l2(input_size, 128, num_classes)

    model = model.to(device)
    
    print("optimizer")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练模型
    log_file = os.path.join(args.output_dir, "train.log")
    
    with open(log_file, 'w') as f:
        for epoch in range(args.epoch):
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(images.to(torch.float32))
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            log_line = f'Epoch {epoch + 1}/{args.epoch}, Loss: {running_loss / len(train_loader)}\n'
            print(log_line)
            f.write(log_line)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_ep_{epoch + 1}.pth'))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--sampler', action="store_true")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bs', type=int, default=1024)
    args = parser.parse_args()
    args.output_dir = f"dhcp_checkpoint/Qwen2.5-VL-7b_pope-dhcp-checkpoint/dhcp_epoch{args.epoch}_lr{args.lr}_bs{args.bs}"
    if args.sampler:
        args.output_dir = args.output_dir + "_sampler"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)
