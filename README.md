# FGSM 攻击与对抗训练防御 (MNIST)

本项目实现了基于 MNIST 手写数字数据集的 FGSM 对抗攻击，并通过对抗训练提升了模型的鲁棒性。

## 📊 实验结果

### 1. 攻击强度 vs 模型准确率 (宏观对比)
![鲁棒性曲线](robustness_curve.png)
*随着扰动强度 (epsilon) 的增加，普通模型 (Normal Model) 准确率断崖式下跌，而对抗训练模型 (Adversarial Model) 依然保持较高的准确率。*

### 2. 案例可视化对比 (微观对比)
![案例对比](attack_visualization.png)
*同一张图片，普通模型被轻微扰动欺骗（预测错误），而对抗训练模型成功防御（预测正确）。*

## 🛠️ 代码结构
* `train_FGSM_with_matplotlib.py`：训练普通模型和对抗模型，并绘制攻击强度与准确率曲线图。
* `FGSM_normal_defense_visualize.py`：加载/训练模型，生成对抗样本，并可视化正常模型与对抗模型的预测对比图。
* `data/`：存放 MNIST 数据集的文件夹。

## 🚀 运行环境
* PyTorch
* torchvision
* matplotlib

## 📖 运行方式
1. 先运行 `train_FGSM_with_matplotlib.py`，生成曲线图。
2. 再运行 `FGSM_normal_defense_visualize.py`，生成案例对比图。

