# -*- coding: utf-8 -*-
"""从已保存的实验数据直接重绘图（无需重新训练）。
前提：train.py 已运行过，output/ 下存在 histories.json / metrics.json / confusion_data.npz。
运行：python replot.py
"""
import os
import json
import numpy as np
import train  # 复用其中的字体配置与绘图函数

OUT = train.OUT_DIR


def main():
    with open(os.path.join(OUT, "histories.json"), encoding="utf-8") as f:
        histories = json.load(f)
    with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    results = m["results"]
    ablation = m["ablation"]

    d = np.load(os.path.join(OUT, "confusion_data.npz"))
    preds, labels = d["preds"], d["labels"]

    train.plot_curves(histories)
    train.plot_comparison(results)
    train.plot_confusion(preds, labels)
    train.plot_ablation(ablation)
    print(f"图表已重新生成到：{OUT}")


if __name__ == "__main__":
    main()
