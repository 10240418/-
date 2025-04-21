import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import seaborn as sns
import psutil
import os
import time

# Load and preprocess the data
file_path = 'C:\\Users\\19176\\Desktop\\MSC-PCA(三维).xlsx'
data = pd.read_excel(file_path)

# Shuffle and split the data
data = data.sample(frac=1).reset_index(drop=True)
t_data = data.iloc[360:450, :]
data = data.iloc[:360, :]

# Extract labels and features
t_label = t_data.iloc[:, 0].tolist()
t_data = t_data.iloc[:, 1:]
t_data = np.array(t_data)

label = data.iloc[:, 0].tolist()
data = data.iloc[:, 1:]
data = np.array(data)

# Scale the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
t_data = scaler.transform(t_data)

# Convert to torch tensors
data = torch.tensor(data, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.long)
t_data = torch.tensor(t_data, dtype=torch.float32)
t_label = torch.tensor(t_label, dtype=torch.long)

# Define custom dataset
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

# Set device and create dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MyDataset(data, label)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

t_data = t_data.to(device)
t_label = t_label.to(device)

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

# Objective function for hyperparameter optimization
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 1.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    train_loader = dataloader

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(data.shape[1], hidden_size),  # Adjust input size dynamically
        nn.ReLU(),
        nn.Dropout(dropout),
        *(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)) for _ in range(num_layers - 1)),
        nn.Linear(hidden_size, 3),
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(1500):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            yp = model(t_data)
            _, pred = torch.max(yp, 1)
            pred = pred.cpu().numpy()
            t_label_np = t_label.cpu().numpy()
            acc = accuracy_score(t_label_np, pred)
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return acc

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_model_params = study.best_params
    best_lr = best_model_params['lr']
    best_momentum = best_model_params['momentum']
    best_weight_decay = best_model_params['weight_decay']
    best_num_layers = best_model_params['num_layers']
    best_hidden_size = best_model_params['hidden_size']
    best_dropout = best_model_params['dropout']

    # 使用最佳参数重新训练模型
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(data.shape[1], best_hidden_size),
        nn.ReLU(),
        nn.Dropout(best_dropout),
        *(nn.Sequential(nn.Linear(best_hidden_size, best_hidden_size), nn.ReLU(), nn.Dropout(best_dropout)) for _ in range(best_num_layers - 1)),
        nn.Linear(best_hidden_size, 3),
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=best_momentum, weight_decay=best_weight_decay)

    train_start_time = time.time()
    train_memory_before = get_memory_usage()

    for epoch in range(1500):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_end_time = time.time()
    train_memory_after = get_memory_usage()
    train_time = train_end_time - train_start_time
    train_memory_used = train_memory_after - train_memory_before

    # 使用最佳模型进行预测并计算指标
    predict_start_time = time.time()
    predict_memory_before = get_memory_usage()

    with torch.no_grad():
        yp = model(t_data)
        _, pred = torch.max(yp, 1)
        pred = pred.cpu().numpy()
        t_label_np = t_label.cpu().numpy()

        acc = accuracy_score(t_label_np, pred)
        precision = precision_score(t_label_np, pred, average='weighted')
        recall = recall_score(t_label_np, pred, average='weighted')
        f1 = f1_score(t_label_np, pred, average='weighted')
        conf_matrix = confusion_matrix(t_label_np, pred)

    predict_end_time = time.time()
    predict_memory_after = get_memory_usage()
    predict_time = predict_end_time - predict_start_time
    predict_memory_used = predict_memory_after - predict_memory_before

    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.show()

    print('Best Accuracy:', acc)
    print('Best Precision:', precision)
    print('Best Recall:', recall)
    print('Best F1-Score:', f1)
    print('Best Confusion Matrix:')
    print(conf_matrix)
    print('Training time:', train_time, 'seconds')
    print('Prediction time:', predict_time, 'seconds')
    print('Memory used during training:', train_memory_used, 'MB')
    print('Memory used during prediction:', predict_memory_used, 'MB')
