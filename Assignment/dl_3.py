import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
import time
import wandb

# 데이터 불러오기
data = pd.read_csv('WeTIE_dl\log.csv')

num_epoch = 50
batch_size = 32

# X, y 분리
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Tensor로 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 회귀용 MLP 모델 정의
class RegressionMLP(nn.Module):
    def __init__(self, input_dim):
        super(RegressionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 125)
        self.fc2 = nn.Linear(125, 25)
        self.fc3 = nn.Linear(25, 1)

        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(125)
        self.batch_norm2 = nn.BatchNorm1d(25)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        return x

# 가중치 초기화 함수
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)

input_dim = X.shape[1]
model = RegressionMLP(input_dim)
model.apply(weight_init)

# optimizer, loss 정의
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 모델 학습 함수 정의
def train(model, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

# 모델 평가 함수 정의
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    return test_loss

# 학습 및 평가
train_losses, test_losses = [], []

for epoch in range(1, num_epoch + 1):
    train(model, train_loader, optimizer)
    train_loss = evaluate(model, train_loader)
    test_loss = evaluate(model, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# 결과 시각화
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epoch + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epoch + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss per Epoch')
plt.legend()
plt.grid()
plt.show()
