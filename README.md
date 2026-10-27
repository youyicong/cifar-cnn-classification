# 基于 CNN 的图像分类（CIFAR-10）课程设计

## 运行环境
- Python 3.12
- 依赖：`pip install torch torchvision matplotlib scikit-learn`

## 运行
```bash
python train.py
```
首次运行会自动下载 CIFAR-10 数据集到 `./data/`，结果输出到 `./output/`。

## 输出文件
| 文件 | 内容 |
|---|---|
| `results_summary.txt` | 对比实验 + 消融实验指标 + 分类报告 |
| `training_curves.png` | 三个模型的训练 loss/acc 曲线 |
| `comparison_bar.png` | 模型准确率对比柱状图 |
| `confusion_matrix.png` | 最优模型混淆矩阵 |
| `ablation_bar.png` | 消融实验柱状图 |

## 实验设计
1. 对比实验：MLP（基线） vs SimpleCNN vs ImprovedCNN（BN+Dropout+数据增强）
2. 消融实验：逐个去掉数据增强 / BatchNorm / Dropout，观察准确率变化
3. 可视化：训练曲线、混淆矩阵、柱状图
