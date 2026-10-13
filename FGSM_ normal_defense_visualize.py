import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==================== 1. 超参数配置 ====================
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 2  # 为了快速看到效果，这里只训2轮，只要能找出被攻击的图即可
EPSILON = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 2. 数据准备 ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), 
                         batch_size=1000, shuffle=False)

# ==================== 3. 模型定义 ====================
class VictimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ==================== 4. FGSM 攻击函数 ====================
def fgsm_attack(model, criterion, images, labels, epsilon):
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images_adv + epsilon * images_adv.grad.sign()
    adv_images = torch.clamp(adv_images, -1, 1).detach()
    return adv_images

# ==================== 5. 快速训练一个普通模型 ====================
print("正在训练普通模型，用于可视化...")
model = VictimNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"第 {epoch+1} 轮训练完成")

# ==================== 5.5 训练对抗模型（复制自 train_adv.py） ====================
print("正在训练对抗模型，用于对比可视化...")
model_adv = VictimNet().to(device)
optimizer_adv = optim.Adam(model_adv.parameters(), lr=LR)

def train_model_adversarial(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # 先用当前模型生成对抗样本
        images_adv = fgsm_attack(model, criterion, images, labels, EPSILON)
        # 用对抗样本训练
        optimizer.zero_grad()
        outputs = model(images_adv)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train_model_adversarial(model_adv, train_loader, optimizer_adv, criterion, device)
    print(f"对抗模型第 {epoch+1} 轮训练完成")

model.eval()

# ==================== 6. 挑选被攻击成功的图片 ====================
# 1. 先拿一批测试数据
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 2. 预测干净图片（定义 orig_preds）
with torch.no_grad():
    orig_outputs = model(images)
    orig_preds = orig_outputs.argmax(dim=1)

# 3. 生成对抗样本
adv_images = fgsm_attack(model, criterion, images, labels, EPSILON)

# 4. 同时预测两个模型（普通模型 vs 对抗模型）
with torch.no_grad():
    preds_normal_adv = model(adv_images).argmax(dim=1)
    preds_adv_adv = model_adv(adv_images).argmax(dim=1)

# 5. 挑选“普通模型被骗，但对抗模型扛住”的图片
# 条件：原图预测对 + 普通模型在对抗图上预测错 + 对抗模型在对抗图上预测对
success_mask = (orig_preds == labels) & (preds_normal_adv != labels) & (preds_adv_adv == labels)

# 6. 取出符合条件的索引
success_indices = success_mask.nonzero(as_tuple=True)[0]
if len(success_indices) >= 3:
    idx_list = success_indices[:3].tolist()
else:
    print("符合条件的图片不足3张，直接取前3张。")
    idx_list = [0, 1, 2]

# ==================== 7. 画 4x3 对比图 ====================
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
fig.suptitle(f'FGSM Attack: Normal vs Adversarial Model (epsilon={EPSILON})', fontsize=16)

for i, idx in enumerate(idx_list):
    orig_np = images[idx].cpu().squeeze().numpy() * 0.5 + 0.5
    adv_np = adv_images[idx].cpu().squeeze().numpy() * 0.5 + 0.5
    pert_np = (adv_np - orig_np) * 10

    # 第 1 行：原图
    axes[0, i].imshow(orig_np, cmap='gray')
    axes[0, i].set_title(f'Original\nTrue: {labels[idx].item()}, Pred: {orig_preds[idx].item()}')
    axes[0, i].axis('off')

    # 第 2 行：扰动图
    axes[1, i].imshow(pert_np, cmap='gray')
    axes[1, i].set_title('Perturbation (x10)')
    axes[1, i].axis('off')

    # 第 3 行：普通模型预测对抗图
    axes[2, i].imshow(adv_np, cmap='gray')
    axes[2, i].set_title(f'Normal Model (Fooled)\nPred: {preds_normal_adv[idx].item()}')
    axes[2, i].axis('off')

    # 第 4 行：对抗模型预测对抗图
    axes[3, i].imshow(adv_np, cmap='gray')
    axes[3, i].set_title(f'Adversarial Model (Robust)\nPred: {preds_adv_adv[idx].item()}')
    axes[3, i].axis('off')

plt.tight_layout()
plt.savefig('attack_visualization.png', dpi=300, bbox_inches='tight')
print("图片已保存为 attack_visualization.png")
plt.show()

