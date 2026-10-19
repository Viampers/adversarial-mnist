import random
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
EPSILON = 0.2  # 归一化空间中的扰动上限；应结合训练脚本的攻击设置解读。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 2. 数据准备 ====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 将像素转为 [0, 1] 的浮点张量。
    transforms.Normalize((0.5,), (0.5,))  # (x - 0.5) / 0.5，将像素映射到 [-1, 1]。
])

# 缺少权重时提示先运行训练脚本，不自动重新训练。
for filename in ("normal_model.pth", "adv_model.pth"):
    if not (PROJECT_DIR / filename).is_file():
        raise FileNotFoundError(
            f"未找到权重: {PROJECT_DIR / filename}。"
            "请先运行 train_FGSM_ with_matplotlib.py。"
        )

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
    """用真实标签生成非定向 FGSM 样本；求梯度但不更新模型参数。"""
    # 保留原图，仅让输入副本追踪梯度。
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    # 清空参数梯度，backward() 同时计算输入梯度；此处没有优化器更新。
    model.zero_grad()
    loss.backward()
    # epsilon 定义在 [-1, 1] 的归一化空间；裁剪后断开攻击生成的计算图。
    adv_images = images_adv + epsilon * images_adv.grad.sign()
    adv_images = torch.clamp(adv_images, -1, 1).detach()
    return adv_images

# ==================== 5. 加载保存的模型权重 ====================
criterion = nn.CrossEntropyLoss()
model = VictimNet().to(device)
model_adv = VictimNet().to(device)

# 先创建与训练时相同的网络结构，再用 load_state_dict() 装入参数。
# map_location 将张量映射到当前设备；weights_only=True 按权重方式读取。
model.load_state_dict(torch.load(
    PROJECT_DIR / "normal_model.pth", map_location=device, weights_only=True
))
model_adv.load_state_dict(torch.load(
    PROJECT_DIR / "adv_model.pth", map_location=device, weights_only=True
))
# 两个模型均进入评估模式；这不会关闭 FGSM 所需的梯度计算。
model.eval()
model_adv.eval()
print("两组模型权重已加载，无需重新训练。")

# ==================== 6. 挑选被攻击成功的图片 ====================
# 1. 仅在第一批 1000 张测试图片中挑选案例，不代表全测试集统计。
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 2. 普通模型对干净图片的预测；这里没有计算对抗模型的干净预测。
with torch.no_grad():
    orig_outputs = model(images)
    orig_preds = orig_outputs.argmax(dim=1)

# 3. 仅针对普通模型生成攻击；攻击生成不能放在 torch.no_grad() 中。
adv_images = fgsm_attack(model, criterion, images, labels, EPSILON)

# 4. 两个模型预测同一批对抗图片：对抗模型这里接受的是迁移攻击，
# 并非针对它自身生成的白盒攻击；与曲线中各模型单独受攻击的设置不同。
with torch.no_grad():
    preds_normal_adv = model(adv_images).argmax(dim=1)
    preds_adv_adv = model_adv(adv_images).argmax(dim=1)

# 5. 挑选“普通模型被骗，但对抗模型扛住”的图片
# 条件：普通模型原图预测对 + 普通模型对抗图预测错 + 对抗模型对抗图预测对。
# 这是按条件挑选的展示案例，不能由此推断整体防御成功率。
success_mask = (orig_preds == labels) & (preds_normal_adv != labels) & (preds_adv_adv == labels)

# 6. 只展示符合条件的案例，不用无关图片补齐。
success_indices = success_mask.nonzero(as_tuple=True)[0]
idx_list = success_indices[:3].tolist()
if not idx_list:
    raise SystemExit("第一批测试图片中没有符合条件的案例，未生成图片；已有图片未更新。")
if len(idx_list) < 3:
    print(f"仅找到 {len(idx_list)} 张符合条件的图片，其余列留空。")

# ==================== 7. 画 4x3 对比图 ====================
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
fig.suptitle(f'FGSM Attack: Normal vs Adversarial Model (epsilon={EPSILON})', fontsize=16)

# 案例不足 3 张时隐藏空列，避免显示空坐标轴。
for column in range(len(idx_list), 3):
    for row in range(4):
        axes[row, column].axis('off')

for i, idx in enumerate(idx_list):
    # 反归一化到 [0, 1] 便于显示；扰动为显示空间中的像素差，再放大 10 倍。
    orig_np = images[idx].cpu().squeeze().numpy() * 0.5 + 0.5
    adv_np = adv_images[idx].cpu().squeeze().numpy() * 0.5 + 0.5
    pert_np = (adv_np - orig_np) * 10

    # 第 1 行：原图
    axes[0, i].imshow(orig_np, cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'Original\nTrue: {labels[idx].item()}, Pred: {orig_preds[idx].item()}')
    axes[0, i].axis('off')

    # 第 2 行：使用固定、对称的灰度范围；0 为中灰，便于跨列比较。
    # 归一化空间扰动除以 2 后对应显示空间，乘 10 后上限为 EPSILON * 5。
    axes[1, i].imshow(pert_np, cmap='gray', vmin=-max(EPSILON * 5, 1e-8),
                      vmax=max(EPSILON * 5, 1e-8))
    axes[1, i].set_title('Perturbation (x10)')
    axes[1, i].axis('off')

    # 第 3 行：普通模型预测对抗图
    axes[2, i].imshow(adv_np, cmap='gray', vmin=0, vmax=1)
    axes[2, i].set_title(f'Normal Model\nPred: {preds_normal_adv[idx].item()}')
    axes[2, i].axis('off')

    # 第 4 行：同一张对抗图在对抗模型上的预测（没有再次生成攻击）。
    axes[3, i].imshow(adv_np, cmap='gray', vmin=0, vmax=1)
    axes[3, i].set_title(f'Adversarial Model\nPred: {preds_adv_adv[idx].item()}')
    axes[3, i].axis('off')

plt.tight_layout()
plt.savefig(PROJECT_DIR / 'attack_visualization.png', dpi=300, bbox_inches='tight')
print("图片已保存为 attack_visualization.png")
plt.show()
