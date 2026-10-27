# -*- coding: utf-8 -*-
"""生成课程设计报告的 Word 文档（规范排版：三线表、黑体标题、宋体正文、页码）。
运行：python build_report.py
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
DOC_PATH = os.path.join(BASE, "52612048014-游宜聪.docx")

# 字体（中文 Word 标准名）
HEI = "黑体"      # 标题/图表题
SONG = "宋体"     # 正文/表内
EN = "Times New Roman"  # 西文/数字

doc = Document()

for s in doc.sections:
    s.left_margin = Inches(1.1)
    s.right_margin = Inches(1.1)
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)


def set_run(run, size=12, bold=False, cn=SONG, en=EN):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)


def H(text, level=1):
    """标题：一级黑体三号(16pt)，二级黑体四号(14pt)。"""
    p = doc.add_paragraph()
    r = p.add_run(text)
    size = {1: 16, 2: 14}.get(level, 14)
    set_run(r, size=size, bold=True, cn=HEI)
    p.paragraph_format.space_before = Pt(13 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(13 if level == 1 else 6)
    p.paragraph_format.line_spacing = 1.5
    return p


def P(text, size=12, bold=False, align=None, indent=True):
    """正文：宋体小四(12pt)，1.5 倍行距，首行缩进 2 字符。"""
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    set_run(r, size=size, bold=bold)
    p.paragraph_format.line_spacing = 1.5
    return p


def CAPTION(text, size=10.5):
    """图表题：黑体五号(10.5pt)，居中。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run(r, size=size, bold=True, cn=HEI)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    return p


def IMG(filename, caption, width_in):
    """插图：图片居中，图题在下方。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(OUT, filename), width=Inches(width_in))
    CAPTION(caption)


def _set_border(tcBorders, edge, sz, val="single"):
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), val)
    el.set(qn("w:sz"), str(sz))   # 单位 1/8 pt：12=1.5pt，6=0.75pt
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    tcBorders.append(el)


def TABLE(headers, rows):
    """三线表：顶线(1.5pt)、栏目线(0.75pt)、底线(1.5pt)，无竖线。"""
    t = doc.add_table(rows=1, cols=len(headers))  # 默认无边框
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True

    # 表头
    for i, htext in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cell.paragraphs[0].add_run(htext)
        set_run(r, size=10.5, bold=True, cn=SONG)
    # 数据行
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cells[i].paragraphs[0].add_run(str(val))
            set_run(r, size=10.5, cn=SONG)

    # 三线边框
    nrows = len(t.rows)
    for ri, row in enumerate(t.rows):
        is_header = (ri == 0)
        is_last = (ri == nrows - 1)
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            tcPr.append(tcBorders)
            if is_header:
                _set_border(tcBorders, "top", 12)     # 顶线 1.5pt
                _set_border(tcBorders, "bottom", 6)   # 栏目线 0.75pt
            if is_last:
                _set_border(tcBorders, "bottom", 12)  # 底线 1.5pt
    doc.add_paragraph()
    return t


def add_page_number(footer_paragraph):
    """在页脚居中插入页码域。"""
    run = footer_paragraph.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)
    set_run(run, size=9, cn=SONG)


# ================= 页脚页码 =================
footer_p = doc.sections[0].footer.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_page_number(footer_p)

# ================= 题目与作者 =================
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("基于卷积神经网络的图像分类研究"); set_run(r, size=22, bold=True, cn=HEI)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("——以 CIFAR-10 数据集为例"); set_run(r, size=15, bold=True, cn=HEI)
doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("姓名：游宜聪　　学号：52612048014"); set_run(r, size=12)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("代码仓库：https://github.com/youyicong/cifar-cnn-classification"); set_run(r, size=10.5)
doc.add_paragraph()

# ================= 摘要 =================
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("摘　要"); set_run(r, size=14, bold=True, cn=HEI)
P("图像分类是计算机视觉最基础也是最核心的任务之一。本文以公开数据集 CIFAR-10 为对象，围绕"
  "“卷积结构、BatchNorm、Dropout 与数据增强分别对分类性能有多大贡献”这一问题，设计并实现了从全连接网络"
  "（MLP）到卷积神经网络（CNN）的递进式对比实验，并进一步开展消融实验。实验结果表明：多层感知机仅取得 53.24% 的测试准确率，"
  "引入卷积结构后提升至 70.33%，加深网络并加入正则化技巧后进一步提升至 78.71%，充分验证了卷积结构对图像分类的关键作用。"
  "消融实验还揭示出在较短训练轮数下模型处于欠拟合状态、过强正则化反而抑制性能的现象，为后续优化指明了方向。")
p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(24)
r = p.add_run("关键词："); set_run(r, size=12, bold=True, cn=HEI)
r = p.add_run("图像分类；卷积神经网络；CIFAR-10；对比实验；消融实验"); set_run(r, size=12)

# ================= 1 研究问题 =================
H("1　研究问题", 1)
H("1.1　要解决什么问题", 2)
P("图像分类的目标是将输入图像自动归入预定义类别。本文要解决的具体问题是：在 CIFAR-10 数据集上，不同的网络结构与"
  "训练技巧能在多大程度上影响图像分类的准确率？具体分解为三个子问题：")
P("（1）全连接网络与卷积网络在图像分类任务上存在多大的性能差距？")
P("（2）加深网络、引入 BatchNorm、Dropout 与数据增强能否进一步提升性能？")
P("（3）这些技巧各自的贡献有多大？")
H("1.2　为什么值得研究", 2)
P("图像分类是目标检测、语义分割、人脸识别、医学影像分析等更复杂视觉任务的基石，其准确率直接关系到众多落地场景的可用性。"
  "理解“为什么卷积网络比全连接网络更适合图像”“正则化技巧在什么条件下起作用”，是深入掌握深度学习的重要途径。"
  "CIFAR-10 作为经典的公开基准数据集，规模适中、类别清晰，非常适合开展一次可复现、可量化的对比研究。")

# ================= 2 数据与实验环境 =================
H("2　数据与实验环境", 1)
H("2.1　数据集", 2)
P("本文使用公开数据集 CIFAR-10（来源：https://www.cs.toronto.edu/~kriz/cifar.html，由 Krizhevsky 收集发布）。"
  "该数据集共 60,000 张 32×32 的彩色图像，分为 10 个类别（airplane、automobile、bird、cat、deer、dog、frog、horse、"
  "ship、truck），每类 6,000 张，其中训练集 50,000 张、测试集 10,000 张，如表 1 所示。")
CAPTION("表 1　CIFAR-10 数据集概况")
TABLE(["属性", "说明"], [
    ["图像尺寸", "32×32×3（彩色）"],
    ["类别数", "10"],
    ["训练集", "50,000 张"],
    ["测试集", "10,000 张"],
    ["预处理", "归一化（均值 0.4914/0.4822/0.4465，标准差 0.2023/0.1994/0.2010）"],
])
H("2.2　实验环境", 2)
P("操作系统：Windows 11；编程语言：Python 3.12；深度学习框架：PyTorch 2.14.1（CPU 版）+ torchvision 0.29.1；"
  "其他依赖：NumPy、scikit-learn、Matplotlib；训练设备：CPU（无 GPU）。")
P("数据集与代码均基于公开资料，模型结构参考经典的 LeNet/VGG 思想，由本人自行实现（代码见 GitHub 仓库）。")

# ================= 3 方法与技术 =================
H("3　方法与技术", 1)
H("3.1　方法总览", 2)
P("本文采用“递进式对比”的研究思路，构建三个由浅入深的模型，并在此基础上开展消融实验，如表 2 所示。")
CAPTION("表 2　本文采用的模型与方法")
TABLE(["模型", "结构要点", "作用"], [
    ["MLP（基线）", "3 层全连接（3072→512→256→10）", "最朴素的方法，作为性能下界"],
    ["SimpleCNN", "2 个卷积块 + 全连接", "引入卷积结构"],
    ["ImprovedCNN", "3 个卷积块 + BatchNorm + Dropout + 数据增强", "在卷积基础上加入正则化技巧"],
])
H("3.2　技术选型理由", 2)
P("（1）卷积神经网络（CNN）：卷积通过“局部感受野 + 参数共享”能够高效提取图像的局部空间特征（边缘、纹理等），"
  "天然适合图像任务，是图像分类的主流结构。")
P("（2）BatchNorm：对每层输出做归一化，稳定训练、加速收敛，并带有轻微的正则化作用。")
P("（3）Dropout：训练时随机丢弃部分神经元，缓解过拟合、提升泛化能力。")
P("（4）数据增强：通过对训练图像随机裁剪、水平翻转，扩充有效数据量，进一步防止过拟合。")
P("选择这一组方法的理由是：它们构成了一条清晰、可解释的递进链条，且全部可复现、可量化对比，能够系统地回答第 1 节提出的研究问题。")

# ================= 4 设计与实现 =================
H("4　设计与实现", 1)
H("4.1　整体流程", 2)
P("实验整体流程为：数据加载与预处理 → 模型定义 → 训练 → 测试 → 结果可视化。数据加载与预处理使用 torchvision 完成，"
  "训练与评估逻辑统一封装。")
H("4.2　模型结构", 2)
P("MLP：将 32×32×3 图像展平为 3072 维向量，经三层全连接（512→256→10），使用 ReLU 激活。")
P("SimpleCNN：两个卷积块（conv3×3→ReLU→MaxPool2×2），通道数由 3→32→64，最后经全连接（128→10）输出。")
P("ImprovedCNN：三个卷积块（conv3×3→BatchNorm→ReLU→MaxPool2×2），通道数 3→32→64→128，分类头加入 Dropout（0.5）"
  "与全连接（256→10），并对训练集施加数据增强。")
H("4.3　训练配置", 2)
P("统一使用 Adam 优化器（学习率 1×10⁻³）、交叉熵损失函数（CrossEntropyLoss），batch size 为 128。MLP 训练 10 个 epoch，"
  "两个 CNN 训练 20 个 epoch。为便于公平对比，所有实验固定随机种子（seed=42）。")

# ================= 5 实验与结果 =================
H("5　实验与结果", 1)
H("5.1　对比实验", 2)
P("三个模型在 CIFAR-10 测试集上的准确率如表 3 与图 1 所示。")
CAPTION("表 3　对比实验结果")
TABLE(["模型", "测试集准确率"], [
    ["MLP", "53.24%"],
    ["SimpleCNN", "70.33%"],
    ["ImprovedCNN", "78.71%"],
])
IMG("comparison_bar.png", "图 1　三种模型在 CIFAR-10 上的测试准确率对比", 4.2)
P("结果显示：全连接网络只能达到 53.24% 的准确率，因为它将图像展平为向量，完全丢失了二维空间结构；引入卷积结构后"
  "准确率跃升至 70.33%（提升 17.09 个百分点），充分说明卷积对图像局部特征的提取能力是分类性能的关键；进一步加深网络并加入"
  "BatchNorm、Dropout、数据增强后，准确率再提升至 78.71%。")
H("5.2　训练过程分析", 2)
IMG("training_curves.png", "图 2　三个模型的训练 loss 与准确率曲线", 6.0)
P("由图 2 可见，三个模型的训练 loss 均随 epoch 稳步下降，训练准确率与测试准确率逐渐收敛且未出现明显分叉，说明模型"
  "正常收敛、未发生严重过拟合。")
H("5.3　混淆矩阵分析", 2)
IMG("confusion_matrix.png", "图 3　ImprovedCNN 在测试集上的混淆矩阵", 4.0)
P("表 4 给出 ImprovedCNN 各类别的 precision、recall 与 F1 值。可以看到 automobile（0.91）、ship（0.88）、truck（0.88）"
  "等类别识别效果最好，而 cat（precision 0.59）、bird（recall 0.61）、dog（precision 0.66）表现最差，猫、狗、鸟三类之间混淆明显"
  "——这与直觉相符：这三类动物外观相似度高，且 CIFAR-10 图像分辨率低，细节特征难以分辨。")
CAPTION("表 4　ImprovedCNN 主要类别分类指标（节选）")
TABLE(["类别", "precision", "recall", "F1"], [
    ["automobile", "0.9145", "0.9090", "0.9117"],
    ["ship", "0.8440", "0.9140", "0.8776"],
    ["truck", "0.8988", "0.8700", "0.8841"],
    ["dog", "0.6559", "0.7720", "0.7092"],
    ["cat", "0.5883", "0.6260", "0.6066"],
    ["bird", "0.7454", "0.6090", "0.6703"],
])
H("5.4　消融实验", 2)
P("为量化每个技巧的贡献，本文以 ImprovedCNN 为基础，分别去掉数据增强、BatchNorm、Dropout，结果如表 5 与图 4 所示。")
CAPTION("表 5　消融实验结果")
TABLE(["变体", "测试集准确率", "相对完整版变化"], [
    ["完整 ImprovedCNN", "78.71%", "—"],
    ["无 BatchNorm", "78.29%", "−0.42%"],
    ["无数据增强", "80.27%", "+1.56%"],
    ["无 Dropout", "82.34%", "+3.63%"],
])
IMG("ablation_bar.png", "图 4　消融实验结果对比", 4.2)
P("一个有意思的现象是：去掉数据增强或 Dropout 后准确率不降反升。这说明在 20 个 epoch 的较短训练下，模型尚未过拟合，"
  "过强的正则化反而压制了模型的拟合能力；只有 BatchNorm 带来了稳定的轻微正收益（+0.42%）。该现象的详细讨论见第 6 节。")

# ================= 6 分析与思考 =================
H("6　分析与思考", 1)
H("6.1　结论", 2)
P("本文通过递进式对比实验与消融实验，系统回答了研究问题：卷积结构是图像分类性能的关键（较 MLP 提升 17 个百分点），"
  "加深网络与加入正则化技巧可进一步提升性能（较 SimpleCNN 提升 8.38 个百分点）。")
H("6.2　意外发现与思考", 2)
P("消融实验揭示了一个有价值的现象：在 20 个 epoch 的较短训练下，模型处于欠拟合而非过拟合状态，因此 Dropout（0.5）与"
  "数据增强这类“防止过拟合”的正则化手段反而拖慢了收敛、降低了最终准确率。这一结果提醒我们：正则化并非越多越好，"
  "其强度应与训练时长、模型容量相匹配——只有当模型真正开始过拟合时，正则化的泛化收益才会显现。")
H("6.3　不足", 2)
P("（1）CIFAR-10 图像分辨率低、类别简单，与真实场景差距较大；（2）未使用预训练模型或更深的骨干网络（如 ResNet），"
  "精度存在上限；（3）受 CPU 训练条件限制，训练轮数偏少，未能让数据增强的泛化优势充分体现。")
H("6.4　改进方向", 2)
P("（1）降低 Dropout 强度（0.5→0.2），并延长训练轮数（20→50+）配合学习率衰减；（2）采用 ResNet/EfficientNet 等更强骨干，"
  "或使用 ImageNet 预训练权重进行迁移学习；（3）引入 CutMix/MixUp 等更强的数据增强策略。")

# ================= 参考文献 =================
H("参考文献", 1)
refs = [
    "[1] Krizhevsky A. Learning multiple layers of features from tiny images[R]. Technical Report, 2009.",
    "[2] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.",
    "[3] Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[C]. NeurIPS, 2012.",
    "[4] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]. ICML, 2015.",
    "[5] Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: A simple way to prevent neural networks from overfitting[J]. JMLR, 2014, 15(1): 1929-1958.",
    "[6] Paszke A, Gross S, Massa F, et al. PyTorch: An imperative style, high-performance deep learning library[C]. NeurIPS, 2019.",
]
for ref in refs:
    p = doc.add_paragraph()
    r = p.add_run(ref)
    set_run(r, size=10.5, cn=SONG)
    p.paragraph_format.line_spacing = 1.3

doc.save(DOC_PATH)
print("已生成：", DOC_PATH)
