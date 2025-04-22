import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import logging

# 设置日志
logger = logging.getLogger('baijiu_app')

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

class ClassifierModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = MinMaxScaler()
        
    def train(self, train_data_path):
        try:
            # 加载训练数据
            data = pd.read_excel(train_data_path)
            
            # 数据预处理
            label = data.iloc[:, 0].tolist()
            features = data.iloc[:, 1:]
            features = self.scaler.fit_transform(features)
            
            # 转换为torch tensors
            data = torch.tensor(features, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            
            # 创建和训练模型
            self.model = self._create_model(data.shape[1])
            self._train_model(data, label)
            
            return True
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise e
    
    def predict(self, sample_data_path):
        if self.model is None:
            raise ValueError("模型未训练")
            
        try:
            # 加载单个样品数据
            sample = pd.read_excel(sample_data_path)
            features = sample.iloc[:, 1:]  # 假设第一列是标签或ID
            
            # 预处理
            features = self.scaler.transform(features)
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                
            # 返回三种酒的概率分布
            return probabilities[0].cpu().numpy()
        except Exception as e:
            logger.error(f"样品预测失败: {str(e)}")
            raise e
    
    def _create_model(self, input_size):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # 3个输出对应三种酒
        ).to(self.device)
        return model
    
    def _train_model(self, data, labels, epochs=1500):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        dataset = MyDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step() 