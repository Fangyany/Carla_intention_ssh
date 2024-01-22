import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.cuda.set_device(0)

device = 'cuda'
batch_size = 32
num_epochs = 1500

class CarlaDataset(Dataset):
    def __init__(self):
        self.data = pickle.load(open('data_and_label_new.pkl', "rb"))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        traj, label = self.data[index]
        traj = torch.tensor(np.array(traj))   # shape=(20,3)
        if label == 0:
            label = [1, 0, 0]
        elif label == 1:
            label = [0, 1, 0]
        else:
            label = [0, 0, 1]
        label = torch.tensor(label)
        return traj, label

dataset = CarlaDataset()

# 划分数据集
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=5) 

# 保存训练集
with open("train_data.pkl", "wb") as f:
    pickle.dump(train_data, f)

# 保存测试集
with open("test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.fc4 = nn.Linear(3, 64)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        ctr = x[:, -1]
        ctr_feature = self.fc4(ctr)
        
        x = x.view(-1, 60)
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x = x.view(batch_size, sequence_length, 1)
        
        # Apply LSTM
        h0 = torch.zeros(2, batch_size, 64).to(x.device)  # Initial hidden state
        c0 = torch.zeros(2, batch_size, 64).to(x.device)  # Initial cell state
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Reshape and pass through linear layers
        lstm_out = lstm_out[:, -1, :]  # Take the last time step output
        x = self.relu(lstm_out)
        x = self.fc1(x)
        result_tensor = torch.cat((x, ctr_feature), dim=1)

        x = self.relu(result_tensor)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        x = F.softmax(x, dim=1)
        
        return x

model = Net().to(device)


# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

log_file = open("log.txt", "w")

# 训练模型
losses = []
for epoch in range(num_epochs):
    for trajs, labels in train_dataloader:
        trajs = trajs.to(device).float()
        labels = labels.to(device).float()
        model.train()
        outputs = model(trajs).squeeze()
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_file.write(f"Epoch: {epoch+1}, Loss: {loss.item()}\n")



# 关闭 log.txt 文件
log_file.close()

# 保存模型权重
torch.save(model.state_dict(), "model_weights.pth")

plt.plot(losses)
plt.savefig("loss.png")



size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
model.eval().to(device)
test_loss, correct = 0, 0
with torch.no_grad():
    for trajs, labels in test_dataloader:
        trajs, labels = trajs.to(device).float(), labels.to(device).float()
        pred = model(trajs)

        test_loss += loss_fn(pred, labels)
        _, predicted_labels = torch.max(pred, dim=1)
        _, labels = torch.max(labels, dim=1)

        correct += (predicted_labels == labels).sum().item()

test_loss /= num_batches
accuracy = correct / size
print(size, correct)
print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")





