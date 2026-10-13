import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==================== 1. 超参数配置 ====================
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 3
EPSILON = 0.2  # 对抗训练和测试时使用的扰动强度

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
    """
    输入干净图片，返回对抗图片。
    注意：内部会计算梯度，但不会更新模型参数。
    """
    # 克隆并打开梯度追踪
    images_adv = images.clone().detach().requires_grad_(True)
    
    # 前向传播 + 反向传播算梯度
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    
    # 生成对抗样本，并裁剪回合法范围
    adv_images = images_adv + epsilon * images_adv.grad.sign()
    adv_images = torch.clamp(adv_images, -1, 1).detach()
    return adv_images

# ==================== 5. 训练函数 ====================
def train_model(model, train_loader, optimizer, criterion, device, use_adversarial):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 如果开启对抗训练，先用当前模型生成对抗样本
        if use_adversarial:
            images = fgsm_attack(model, criterion, images, labels, EPSILON)
        
        # 标准训练步骤
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ==================== 6. 评估函数 ====================
def evaluate_model(model, test_loader, criterion, device, epsilon=EPSILON):
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
  
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            # 1. 干净数据准确率
            outputs_clean = model(images)
            preds_clean = outputs_clean.argmax(dim=1)
            correct_clean += (preds_clean == labels).sum().item()
        
        # 2. FGSM 攻击下的准确率
        adv_images = fgsm_attack(model, criterion, images, labels, epsilon)

        with torch.no_grad():
            outputs_adv = model(adv_images)
            preds_adv = outputs_adv.argmax(dim=1)
            correct_adv += (preds_adv == labels).sum().item()
        
        total += labels.size(0)
    
    #print(f"[DEBUG] total={total}, correct_clean={correct_clean}, correct_adv={correct_adv}")
    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    return clean_acc, adv_acc

# ==================== 7. 主流程：对比实验 ====================
if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()

    # ---- 阶段一：普通训练 ----
    print("\n" + "="*50)
    print("开始训练【普通模型】（只用干净数据）")
    print("="*50)
    model_normal = VictimNet().to(device)
    optimizer_normal = optim.Adam(model_normal.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        train_model(model_normal, train_loader, optimizer_normal, criterion, device, use_adversarial=False)
        print(f"第 {epoch+1} 轮训练完成")

    # ---- 阶段二：对抗训练 ----
    print("\n" + "="*50)
    print("开始训练【对抗模型】（每个 batch 混入 FGSM 样本）")
    print("="*50)
    model_adv = VictimNet().to(device)
    optimizer_adv = optim.Adam(model_adv.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        train_model(model_adv, train_loader, optimizer_adv, criterion, device, use_adversarial=True)
        print(f"第 {epoch+1} 轮训练完成")

    # ---- 阶段三：对比评估 ----
    print("\n" + "="*50)
    print("开始评估对比（测试 FGSM 攻击，epsilon = {:.2f}）".format(EPSILON))
    print("="*50)
    
    clean_norm, adv_norm = evaluate_model(model_normal, test_loader, criterion, device)
    clean_adv, adv_adv = evaluate_model(model_adv, test_loader, criterion, device)

    print(f"\n{'模型类型':<12} | {'干净准确率':<10} | {'FGSM攻击下准确率':<15}")
    print("-" * 50)
    print(f"{'普通模型':<12} | {clean_norm:>8.2f}% | {adv_norm:>13.2f}%")
    print(f"{'对抗模型':<12} | {clean_adv:>8.2f}% | {adv_adv:>13.2f}%")
    print("="*50)

    # ==================== 8. 可视化：攻击强度 vs 准确率曲线 ====================
    print("\n开始绘制鲁棒性曲线（这步会跑几轮评估，稍等片刻）...")

    # 定义扰动强度列表（0.0 代表干净数据，所以取干净准确率，其他值取攻击下的准确率）
    epsilon_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    # 用来存放准确率
    acc_normal_list = []
    acc_adv_list = []

    # 遍历每个 epsilon，分别评估两个模型
    for eps in epsilon_list:
        # evaluate_model 返回 (clean_acc, adv_acc)
        clean_n, adv_n = evaluate_model(model_normal, test_loader, criterion, device, epsilon=eps)
        clean_a, adv_a = evaluate_model(model_adv, test_loader, criterion, device, epsilon=eps)
        
        # 如果是 0.0，就用干净准确率，代表没攻击时的表现；否则用攻击下的准确率
        acc_normal_list.append(clean_n if eps == 0.0 else adv_n)
        acc_adv_list.append(clean_a if eps == 0.0 else adv_a)
        print(f"epsilon = {eps:.2f} 评估完成")

    # 开始画图
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_list, acc_normal_list, marker='o', linewidth=2, label='Normal Model')
    plt.plot(epsilon_list, acc_adv_list, marker='s', linewidth=2, label='Adversarial Model')

    plt.title('Model Robustness against FGSM Attack', fontsize=14)
    plt.xlabel('Perturbation Strength (Epsilon)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(epsilon_list)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # 保存图片到本地
    plt.savefig('robustness_curve.png', dpi=300, bbox_inches='tight')
    print("图片已保存为 robustness_curve.png")

    # 弹出窗口显示图片
    plt.show()