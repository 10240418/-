import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 设置日志
logger = logging.getLogger('baijiu_app')

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_classification_model(data_path=None, save_path=None):
    """
    训练白酒分类模型并保存
    """
    # 如果未指定路径，使用默认路径
    if data_path is None:
        # 尝试获取工作目录下的训练数据
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'classification_data.xlsx')
        
        # 如果找不到文件，则使用classify_tool.py中的默认路径
        if not os.path.exists(data_path):
            data_path = "C:\\Users\\19176\\Desktop\\MSC-PCA(三维).xlsx"
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'baijiu_classifier.pth')

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # 加载数据
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_excel(data_path)
        
        # 检查数据格式
        logger.info(f"Data shape: {df.shape}")
        
        # 假设最后一列是标签，前面的列是特征
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # 数据处理
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 创建数据集和加载器
        train_dataset = MyDataset(X_train, y_train)
        test_dataset = MyDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 定义模型
        input_size = X.shape[1]  # 特征数量
        hidden_size = 128
        num_classes = len(np.unique(y))  # 类别数量
        
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        ).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练参数
        num_epochs = 100  # 减少迭代次数以便快速训练，实际应用可以增加
        
        # 训练模型
        logger.info("Starting model training...")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # 每10个epoch打印一次损失
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        
        # 评估模型
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-score: {f1:.4f}")
        
        # 保存模型
        logger.info(f"Saving model to: {save_path}")
        torch.save(model.state_dict(), save_path)
        logger.info("Model training and saving complete!")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 设置基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 训练并保存模型
    train_classification_model() 