import numpy as np
import matplotlib.pyplot as plt


def add_value_labels(ax, bars, dy=0.012):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main():
    # 数据（来自图中数值）
    metrics = [r"Weight R$^2$", r"Trait R$^2$", "Level Acc"]  # Shape -> Trait

    cucumber_single = [0.964, 0.915, 0.760]
    cucumber_ours = [0.969, 0.897, 0.920]

    banana_single = [0.281, 0.713, 0.833]
    banana_ours = [0.831, 0.906, 0.933]

    x = np.arange(len(metrics))
    width = 0.35

    # 加高画布，避免图例/数值与顶部元素重叠
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)

    # 保持显示上限为 1.0（刻度到 1.0），但绘图区留一点上边距
    y_max_tick = 1.0
    y_max_plot = 1.12
    y_ticks = np.linspace(0, y_max_tick, 6)

    # 左图：Cucumber
    ax = axes[0]
    b1 = ax.bar(x - width / 2, cucumber_single, width, label="Single Task", color="#1f77b4")
    b2 = ax.bar(x + width / 2, cucumber_ours, width, label="Ours", color="#ff7f0e")
    ax.set_title("Cucumber: Detailed Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, y_max_plot)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    add_value_labels(ax, b1)
    add_value_labels(ax, b2)

    # 右图：Banana
    ax = axes[1]
    b1 = ax.bar(x - width / 2, banana_single, width, label="Single Task", color="#1f77b4")
    b2 = ax.bar(x + width / 2, banana_ours, width, label="Ours", color="#ff7f0e")
    ax.set_title("Banana: Detailed Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, y_max_plot)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    add_value_labels(ax, b1)
    add_value_labels(ax, b2)

    out = "detailed_comparison_trait.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

