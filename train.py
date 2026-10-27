# -*- coding: utf-8 -*-
"""
深度学习课程设计 —— 基于卷积神经网络的图像分类研究（CIFAR-10）
完整实验脚本：对比实验 + 消融实验 + 可视化，全程自动保存结果。

运行方式：python train.py
输出目录：./output/
  - results_summary.txt      所有实验指标汇总
  - training_curves.png      三个模型的训练 loss/acc 曲线
  - comparison_bar.png       模型准确率对比柱状图
  - confusion_matrix.png     最优模型的混淆矩阵
  - ablation_bar.png         消融实验柱状图
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")  # 无界面保存图片
import matplotlib.pyplot as plt

# 中文字体配置（避免图中中文显示为方框/乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------- 全局配置 -------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS_MLP = 10        # MLP 训练轮数
EPOCHS_CNN = 20        # 两个 CNN 训练轮数
LR = 1e-3
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------- 数据 -------------------------
def get_dataloaders(augment=False):
    """加载 CIFAR-10。augment=True 时使用数据增强（训练集）。"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])

    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ------------------------- 模型 -------------------------
class MLP(nn.Module):
    """基线：全连接多层感知机（把图像展平成向量）。"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.fc(x)


class SimpleCNN(nn.Module):
    """经典 CNN：两个卷积块 + 全连接。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """改进 CNN：在 SimpleCNN 基础上加入 BatchNorm 与 Dropout。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------- 训练 / 评估 -------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    acc = correct / total
    return acc, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def run_experiment(name, model, train_loader, test_loader, epochs):
    """训练一个模型，返回训练曲线与测试集预测结果。"""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "test_acc": []}
    print(f"\n=== 训练 {name} ({epochs} epochs, device={DEVICE}) ===")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        te_acc, _, _ = evaluate(model, test_loader)
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        print(f"  [{name}] epoch {ep:>2}/{epochs}  "
              f"loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")
    print(f"  [{name}] 用时 {time.time() - t0:.1f}s")
    te_acc, preds, labels = evaluate(model, test_loader)
    return hist, te_acc, preds, labels, model


# ------------------------- 绘图 -------------------------
def plot_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, h) in zip(axes, histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="train loss", marker="o", markersize=3)
        ax.plot(epochs, h["train_acc"], label="train acc", marker="o", markersize=3)
        ax.plot(epochs, h["test_acc"], label="test acc", marker="o", markersize=3)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("训练过程曲线（loss / 准确率）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_comparison(results):
    names = list(results.keys())
    accs = [results[n]["test_acc"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=["#9aa5b1", "#4a90d9", "#2e7d32"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("三种模型在 CIFAR-10 上的准确率对比")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparison_bar.png"), dpi=150)
    plt.close(fig)


def plot_confusion(preds, labels, title="ImprovedCNN 混淆矩阵"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CIFAR10_CLASSES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_ablation(ablation):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(ablation.keys())
    accs = [ablation[k] for k in labels]
    bars = ax.bar(labels, accs, color="#4a90d9")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("消融实验：去掉单个组件对准确率的影响")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ablation_bar.png"), dpi=150)
    plt.close(fig)


# ------------------------- 主流程 -------------------------
def main():
    set_seed(SEED)
    print(f"设备：{DEVICE}")

    # 1. 数据（增强版给 ImprovedCNN，其余用基础版）
    train_loader, test_loader = get_dataloaders(augment=False)
    train_aug_loader, _ = get_dataloaders(augment=True)

    results, histories = {}, {}

    # 2. 对比实验：MLP / SimpleCNN / ImprovedCNN
    mlp_hist, mlp_acc, mlp_preds, mlp_labels, _ = run_experiment(
        "MLP", MLP(), train_loader, test_loader, EPOCHS_MLP)
    results["MLP"] = {"test_acc": mlp_acc}
    histories["MLP"] = mlp_hist

    cnn_hist, cnn_acc, cnn_preds, cnn_labels, _ = run_experiment(
        "SimpleCNN", SimpleCNN(), train_loader, test_loader, EPOCHS_CNN)
    results["SimpleCNN"] = {"test_acc": cnn_acc}
    histories["SimpleCNN"] = cnn_hist

    imp_hist, imp_acc, imp_preds, imp_labels, imp_model = run_experiment(
        "ImprovedCNN", ImprovedCNN(), train_aug_loader, test_loader, EPOCHS_CNN)
    results["ImprovedCNN"] = {"test_acc": imp_acc}
    histories["ImprovedCNN"] = imp_hist

    # 3. 消融实验（基于 ImprovedCNN，逐个去掉组件）
    print("\n=== 消融实验（ImprovedCNN 变体）===")
    ablation = {"完整 ImprovedCNN": imp_acc}

    # 去掉数据增强
    ab_hist, ab_acc, _, _, _ = run_experiment(
        "ImprovedCNN(无数据增强)", ImprovedCNN(), train_loader, test_loader, EPOCHS_CNN)
    ablation["无数据增强"] = ab_acc

    # 去掉 BatchNorm
    class NoBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.5),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, NUM_CLASSES),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    _, ab_acc, _, _, _ = run_experiment("ImprovedCNN(无BN)", NoBN(), train_aug_loader, test_loader, EPOCHS_CNN)
    ablation["无 BatchNorm"] = ab_acc

    # 去掉 Dropout
    class NoDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    _, ab_acc, _, _, _ = run_experiment("ImprovedCNN(无Dropout)", NoDropout(), train_aug_loader, test_loader, EPOCHS_CNN)
    ablation["无 Dropout"] = ab_acc

    # 4. 保存结果
    print("\n================ 结果汇总 ================")
    for k, v in results.items():
        print(f"  {k:<12} test_acc = {v['test_acc']:.4f} ({v['test_acc']:.2%})")
    print("\n消融实验：")
    for k, v in ablation.items():
        print(f"  {k:<18} test_acc = {v:.4f} ({v:.2%})")

    with open(os.path.join(OUT_DIR, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write("对比实验（CIFAR-10 测试集准确率）：\n")
        for k, v in results.items():
            f.write(f"  {k:<12} {v['test_acc']:.4f} ({v['test_acc']:.2%})\n")
        f.write("\n消融实验（ImprovedCNN 变体）：\n")
        for k, v in ablation.items():
            f.write(f"  {k:<18} {v:.4f} ({v:.2%})\n")
        f.write("\nImprovedCNN 分类报告：\n")
        f.write(classification_report(imp_labels, imp_preds, target_names=CIFAR10_CLASSES, digits=4))

    # 4.5 保存原始数据（便于日后单独重绘图表，无需重新训练）
    with open(os.path.join(OUT_DIR, "histories.json"), "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "ablation": ablation}, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(OUT_DIR, "confusion_data.npz"),
             preds=imp_preds, labels=imp_labels)

    # 5. 绘图
    plot_curves(histories)
    plot_comparison(results)
    plot_confusion(imp_preds, imp_labels)
    plot_ablation(ablation)

    print(f"\n全部结果已保存到：{OUT_DIR}")
    print("生成文件：results_summary.txt / training_curves.png / comparison_bar.png / "
          "confusion_matrix.png / ablation_bar.png")


if __name__ == "__main__":
    main()
