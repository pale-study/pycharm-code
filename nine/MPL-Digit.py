import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ========== 1. 配置环境与参数 ==========
# 设备配置（优先用GPU，无则用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数
batch_size = 128
learning_rate = 0.001
num_epochs = 20
early_stop_patience = 5  # 早停耐心值

# ========== 2. 数据加载与预处理（PyTorch内置MNIST，国内可访问） ==========
# 数据预处理：转张量 + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量 (1,28,28)，值范围0-1
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值/方差，提升训练稳定性
])

# 下载并加载MNIST数据集（PyTorch会自动缓存，无需重复下载）
full_dataset = datasets.MNIST(
    root='./data',  # 数据缓存路径
    train=True,
    download=True,
    transform=transform
)

# 划分训练集（70%）和测试集（30%）
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 补充测试集（官方测试集，确保数据完整性）
official_test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
# 合并自定义测试集+官方测试集，再按30%总比例调整（最终满足30%测试集要求）
all_test_data = torch.utils.data.ConcatDataset([test_dataset, official_test_dataset])
final_test_size = int(0.3 * (len(full_dataset) + len(official_test_dataset)))
final_test_dataset, _ = random_split(all_test_data, [final_test_size, len(all_test_data) - final_test_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False)

print(f"   数据集划分完成！")
print(f"   训练集大小: {len(train_dataset)} (70%)")
print(f"   测试集大小: {len(final_test_dataset)} (30%)")


# ========== 3. 定义8层隐藏层MLP模型 ==========
class MLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(MLP, self).__init__()
        # 8层隐藏层，尺寸按要求：512,256,64,64,64,32,32,32
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 输出层（无激活，用CrossEntropyLoss）
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平：(batch,1,28,28) → (batch,784)
        out = self.layers(x)
        return out


# 初始化模型、损失函数、优化器
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== 4. 训练模型（含早停机制） ==========
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

print("\n" + "=" * 60)
print("开始训练8层MLP模型...")
print("=" * 60)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)

    # 验证阶段（早停用）
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    avg_val_loss = val_loss / len(final_test_dataset)
    val_losses.append(avg_val_loss)

    # 早停判断
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_mlp_model.pth')  # 保存最优模型
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f" 早停触发！在第{epoch + 1}轮停止训练")
            break

    print(f"Epoch [{epoch + 1}/{num_epochs}], 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

# 加载最优模型
model.load_state_dict(torch.load('best_mlp_model.pth'))

# ========== 5. 模型测试 + 混淆矩阵 ==========
model.eval()
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 收集预测结果和真实标签（用于混淆矩阵）
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# # 计算准确率
# accuracy = 100 * correct / total
# print(f"\n 测试完成！测试集准确率: {accuracy:.2f}%")

# 绘制混淆矩阵
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测标签', fontsize=14)
plt.ylabel('真实标签', fontsize=14)
plt.title('MNIST手写数字分类 - 混淆矩阵', fontsize=16)
plt.tight_layout()
plt.show()

# 绘制训练/验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.xlabel('迭代轮次')
plt.ylabel('损失值')
plt.title('训练与验证损失变化')
plt.legend()
plt.grid(True)
plt.show()