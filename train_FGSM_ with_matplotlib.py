import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==================== 随机种子设置 ====================
# 固定 Python、NumPy 和 PyTorch 随机源，便于在相同环境下复现实验。
# 不保证跨设备、跨 PyTorch/CUDA 版本得到完全相同的结果。
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 对使用 cuDNN 的算子优先采用确定性实现；关闭自动测速选择算法。
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"随机种子已设置为: {SEED}")

# ==================== 1. 超参数配置 ====================
# 路径以本脚本所在目录为基准，不受终端当前目录影响。
PROJECT_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 3
EPSILON = 0.2  # 在归一化后的 [-1, 1] 空间施加扰动；对应原始 [0, 1] 空间的 0.1。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 2. 数据准备 ====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 将像素转为 [0, 1] 的浮点张量。
    transforms.Normalize((0.5,), (0.5,))  # (x - 0.5) / 0.5，将像素映射到 [-1, 1]。
])

train_loader = DataLoader(datasets.MNIST(str(PROJECT_DIR / 'data'), train=True, download=True, transform=transform),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST(str(PROJECT_DIR / 'data'), train=False, transform=transform),
                         batch_size=1000, shuffle=False)

# ==================== 3. 模型定义 ====================
# 全连接分类器：28×28 像素 → 784 → 128 → 10；输出 logits，交叉熵内部处理归一化。
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
    使用真实标签执行非定向 FGSM：沿增大分类损失的方向扰动输入。
    epsilon 是归一化输入空间中的 L∞ 扰动上限；返回脱离计算图的对抗图片。
    backward() 会产生输入和模型参数的梯度，但这里不调用 optimizer.step()，
    因而不更新参数；训练时会在正式反向传播前清空这些参数梯度。
    """
    # 保留原图不变，将输入副本作为需要梯度的叶子张量。
    images_adv = images.clone().detach().requires_grad_(True)

    # 计算损失对输入的梯度；使用真实标签，而非模型预测作为攻击目标。
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    # 沿梯度符号方向走一步，再裁剪到归一化像素合法范围 [-1, 1]。
    # detach() 使后续训练不通过攻击生成过程继续求导。
    adv_images = images_adv + epsilon * images_adv.grad.sign()
    adv_images = torch.clamp(adv_images, -1, 1).detach()
    return adv_images

# ==================== 5. 训练函数 ====================
def train_model(model, train_loader, optimizer, criterion, device, use_adversarial):
    # use_adversarial=False：干净样本训练；True：仅对抗样本训练。
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 用当前模型为整批输入生成 FGSM 样本并替换原图；没有混合干净样本损失。
        if use_adversarial:
            images = fgsm_attack(model, criterion, images, labels, EPSILON)

        # 清空攻击生成时留下的参数梯度，再基于当前这批样本计算损失并更新参数。
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ==================== 6. 评估函数 ====================
def evaluate_model(model, test_loader, criterion, device, epsilon=EPSILON):
    # eval() 设置评估模式，不会禁用梯度；FGSM 仍需要输入梯度。
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

        # 2. 针对当前受评模型生成白盒 FGSM 样本，因此两模型的攻击图片一般不同。
        # 此步骤不能放进 torch.no_grad()；只有预测和计数无需梯度。
        adv_images = fgsm_attack(model, criterion, images, labels, epsilon)

        with torch.no_grad():
            outputs_adv = model(adv_images)
            preds_adv = outputs_adv.argmax(dim=1)
            correct_adv += (preds_adv == labels).sum().item()

        total += labels.size(0)

    # 分母为全部测试样本数，不仅统计原先分类正确的样本。
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

    # ---- 阶段二：重新初始化一个独立模型，只用对抗样本训练（非普通模型微调） ----
    print("\n" + "="*50)
    print("开始训练【对抗模型】（每个 batch 仅用生成的 FGSM 对抗样本更新参数）")
    print("="*50)
    model_adv = VictimNet().to(device)
    optimizer_adv = optim.Adam(model_adv.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_model(model_adv, train_loader, optimizer_adv, criterion, device, use_adversarial=True)
        print(f"第 {epoch+1} 轮训练完成")

    # state_dict() 提取参数字典，torch.save() 写入文件；同名文件会被覆盖。
    # 不保存优化器状态，供案例脚本加载推理，而非完整恢复训练进度。
    torch.save(model_normal.state_dict(), PROJECT_DIR / "normal_model.pth")
    torch.save(model_adv.state_dict(), PROJECT_DIR / "adv_model.pth")
    print(f"模型权重已保存到: {PROJECT_DIR}")

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

    # 遍历每个 epsilon，分别针对两个模型生成攻击并评估；这里只评估，不重新训练。
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
    plt.savefig(PROJECT_DIR / 'robustness_curve.png', dpi=300, bbox_inches='tight')
    print("图片已保存为 robustness_curve.png")

    # 弹出窗口显示图片
    plt.show()
