"""生成《项目交付》文档用的系统架构图。

产物：docs/architecture.png
布局：左侧「可观测层」竖带（横切）→ 层次分组标签 → 五层主体（单向数据流）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "architecture.png"

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 画布 ----
FIG_W = 12.1

OBS_X, OBS_W = 0.30, 0.92        # 可观测层竖带
GRP_X = 4.98                     # 分层标签列（右对齐锚点）
X0, X1 = 5.35, 11.70             # 五层主体
W = X1 - X0

BOX_H = 1.16                     # 模块小框高度
LAYER_H = 1.82                   # 单层高度
GAP = 0.58                       # 层间距
TITLE_Y = 14.98                  # 标题基线
SUB_Y = 14.70                    # 副标题基线
TOP = 14.42                      # 首层上沿


def layer_bottom(i: int) -> float:
    """第 i 层（0 起）的下沿。层与层之间恒为 GAP。"""
    return TOP - (i + 1) * LAYER_H - i * GAP

# (纵向下沿, 层名, 副标题, 底色, 分层标签, [模块小框…])
LAYERS = [
    (layer_bottom(0), "展示层", "Streamlit 单页三栏", "#DCE9F7", "UI 层", [
        "左栏 文档管理\n上传 / 列表 / 删除",
        "中栏 对话\n流式输出 / 引用标记",
        "右栏 可观测\n轨迹树 / 指标 / 健康检查"]),
    (layer_bottom(1), "调度层", "手写 ReAct Agent", "#FBE3D0", "Agent 决策层", [
        "react\nReAct 循环\nThought→Action\n→Observation",
        "router\n工具路由\ndirect_tool 短路",
        "executor\n并行执行\nThreadPool(4)",
        "memory\n会话隔离\n滑动窗口 / 摘要"]),
    (layer_bottom(2), "生成层", "RAG 流水线", "#DFF0DC", "RAG 生成层", [
        "prompts\nPrompt 模板",
        "rag\n流式生成 / 引用溯源",
        "cache\n语义缓存",
        "fallback\n四级降级"]),
    (layer_bottom(3), "检索层", "三档可切换", "#F6E2E7", "检索层", [
        "vectorstore\n向量 Top-K",
        "bm25_index\n关键词检索",
        "fusion\n手写 RRF 融合",
        "reranker\n交叉编码器精排"]),
    (layer_bottom(4), "数据层", "文档与索引", "#E6E0F0", "数据层", [
        "loaders\nPyMuPDF / docx",
        "chunkers\nfixed / recursive\n/ semantic",
        "pipeline\n批量入库",
        "storage\nChroma / BM25"]),
]

# ---- 画布（纵向范围由层几何推导，避免内容被裁掉）----
BOTTOM = LAYERS[-1][0]
FOOT_Y = BOTTOM - 0.40
FIG_H = TITLE_Y + 0.62
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(FOOT_Y - 0.30, FIG_H)
ax.axis("off")

for bottom, name, sub, color, side, items in LAYERS:
    ax.add_patch(FancyBboxPatch(
        (X0, bottom), W, LAYER_H,
        boxstyle="round,pad=0.045,rounding_size=0.09",
        linewidth=1.5, edgecolor="#5A6B7B", facecolor=color, zorder=2))

    # 层名与副标题同处一行，为模块小框让出整行空间
    ax.text(X0 + 0.20, bottom + LAYER_H - 0.42, name,
            fontsize=15, fontweight="bold", color="#1F3B57",
            ha="left", va="center", zorder=4)
    ax.text(X0 + 0.20 + 0.235 * len(name) + 0.18, bottom + LAYER_H - 0.455, sub,
            fontsize=9.6, color="#5A6B7B", ha="left", va="center", zorder=4)

    # 左侧分层标签
    ax.text(GRP_X, bottom + LAYER_H / 2, side,
            fontsize=11.0, color="#33475B", ha="right", va="center",
            fontweight="bold", zorder=4)

    n = len(items)
    gap = 0.13
    bw = (W - 0.26 - gap * (n - 1)) / n
    by = bottom + 0.15
    for i, label in enumerate(items):
        bx = X0 + 0.13 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0, edgecolor="#8A9AAB", facecolor="white", zorder=3))
        ax.text(bx + bw / 2, by + BOX_H / 2, label,
                fontsize=8.8, color="#22313F", ha="center", va="center",
                linespacing=1.42, zorder=4)

# ---- 层间单向箭头与数据契约标注 ----
CONTRACTS = ["Message / AgentStep", "Tool 调用",
             "top-k ScoredChunk", "Chunk"]
for i in range(len(LAYERS) - 1):
    y_from = LAYERS[i][0]
    y_to = LAYERS[i + 1][0] + LAYER_H
    ax.add_patch(FancyArrowPatch(
        (X0 + W / 2, y_from), (X0 + W / 2, y_to),
        arrowstyle="-|>", mutation_scale=17,
        linewidth=1.7, color="#3B556F", zorder=5))
    ax.text(X0 + W / 2 + 0.16, (y_from + y_to) / 2, CONTRACTS[i],
            fontsize=9.0, color="#3B556F", ha="left", va="center", zorder=6)

# ---- 横切：可观测层 ----
obs_bottom = LAYERS[-1][0]
obs_top = LAYERS[0][0] + LAYER_H
ax.add_patch(FancyBboxPatch(
    (OBS_X, obs_bottom), OBS_W, obs_top - obs_bottom,
    boxstyle="round,pad=0.045,rounding_size=0.09",
    linewidth=1.5, edgecolor="#B08A3E", facecolor="#FBF0D2", zorder=2))
ax.text(OBS_X + OBS_W / 2, (obs_bottom + obs_top) / 2,
        "可 观 测 层\ntrace · metrics",
        fontsize=11.2, fontweight="bold", color="#7A5A18",
        ha="center", va="center", rotation=90, linespacing=1.6, zorder=4)

# ---- 标题与图注 ----
ax.text((X0 + X1) / 2, TITLE_Y, "智能科研助理 —— 系统架构",
        fontsize=18.5, fontweight="bold", color="#12283D",
        ha="center", va="center")
ax.text((X0 + X1) / 2, SUB_Y,
        "基于 RAG + Agent 的论文知识库问答系统",
        fontsize=10.6, color="#5A6B7B", ha="center", va="center")

ax.text(X0 + W / 2, FOOT_Y,
        "层间依赖单向（自上而下），仅通过 core/schemas.py 的数据结构通信；"
        "可观测层横切所有层，不参与数据流。",
        fontsize=9.7, color="#5A6B7B", ha="center", va="center")

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white",
            pad_inches=0.14)
print(f"saved: {OUT}")
