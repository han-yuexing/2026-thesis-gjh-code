import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")


def load_data(csv_path: Path) -> pd.DataFrame:
    """加载CSV数据"""
    df = pd.read_csv(csv_path)
    return df


def print_statistics(df: pd.DataFrame, dataset_name: str):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} 统计信息")
    print(f"{'='*60}")
    
    print(f"\n样本数量: {len(df)}")
    
    # Weight 统计
    print(f"\nWeight (重量) 统计:")
    print(f"  均值: {df['weight'].mean():.2f} g")
    print(f"  标准差: {df['weight'].std():.2f} g")
    print(f"  最小值: {df['weight'].min():.2f} g")
    print(f"  最大值: {df['weight'].max():.2f} g")
    print(f"  中位数: {df['weight'].median():.2f} g")
    
    # Ripeness 统计
    print(f"\nRipeness (成熟度) 统计:")
    print(f"  均值: {df['ripeness'].mean():.4f}")
    print(f"  标准差: {df['ripeness'].std():.4f}")
    print(f"  最小值: {df['ripeness'].min():.4f}")
    print(f"  最大值: {df['ripeness'].max():.4f}")
    print(f"  中位数: {df['ripeness'].median():.4f}")
    
    # Level 分布
    print(f"\nLevel (等级) 分布:")
    level_counts = df['level'].value_counts().sort_index()
    level_props = df['level'].value_counts(normalize=True).sort_index() * 100
    for level in level_counts.index:
        count = level_counts[level]
        prop = level_props[level]
        print(f"  {level}: {count} ({prop:.1f}%)")


def compare_distributions(df30: pd.DataFrame, df100: pd.DataFrame):
    """比较两个数据集的分布"""
    print(f"\n{'='*60}")
    print("分布比较")
    print(f"{'='*60}")
    
    # Weight 比较
    print(f"\nWeight (重量) 比较:")
    print(f"  banana_30:  均值={df30['weight'].mean():.2f}, 标准差={df30['weight'].std():.2f}, 范围=[{df30['weight'].min():.2f}, {df30['weight'].max():.2f}]")
    print(f"  banana_100: 均值={df100['weight'].mean():.2f}, 标准差={df100['weight'].std():.2f}, 范围=[{df100['weight'].min():.2f}, {df100['weight'].max():.2f}]")
    print(f"  差异: 均值差={df100['weight'].mean() - df30['weight'].mean():.2f}, 标准差差={df100['weight'].std() - df30['weight'].std():.2f}")
    
    # Ripeness 比较
    print(f"\nRipeness (成熟度) 比较:")
    print(f"  banana_30:  均值={df30['ripeness'].mean():.4f}, 标准差={df30['ripeness'].std():.4f}, 范围=[{df30['ripeness'].min():.4f}, {df30['ripeness'].max():.4f}]")
    print(f"  banana_100: 均值={df100['ripeness'].mean():.4f}, 标准差={df100['ripeness'].std():.4f}, 范围=[{df100['ripeness'].min():.4f}, {df100['ripeness'].max():.4f}]")
    print(f"  差异: 均值差={df100['ripeness'].mean() - df30['ripeness'].mean():.4f}, 标准差差={df100['ripeness'].std() - df30['ripeness'].std():.4f}")
    
    # Level 分布比较
    print(f"\nLevel (等级) 分布比较:")
    levels = ['super', 'first', 'second', 'third']
    for level in levels:
        count30 = (df30['level'] == level).sum()
        count100 = (df100['level'] == level).sum()
        prop30 = count30 / len(df30) * 100
        prop100 = count100 / len(df100) * 100
        print(f"  {level}:")
        print(f"    banana_30:  {count30:2d} ({prop30:5.1f}%)")
        print(f"    banana_100: {count100:2d} ({prop100:5.1f}%)")
        print(f"    差异: {count100 - count30:2d} ({prop100 - prop30:5.1f}%)")


def plot_distributions(df30: pd.DataFrame, df100: pd.DataFrame, output_dir: Path):
    """绘制分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Banana Dataset Distribution Comparison', fontsize=16, fontweight='bold')
    
    # Weight 分布
    ax1 = axes[0, 0]
    ax1.hist(df30['weight'], bins=20, alpha=0.6, label='banana_30', color='blue', edgecolor='black')
    ax1.hist(df100['weight'], bins=20, alpha=0.6, label='banana_100', color='red', edgecolor='black')
    ax1.set_xlabel('Weight (g)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Weight Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Weight 箱线图
    ax2 = axes[0, 1]
    data_weight = [df30['weight'], df100['weight']]
    bp1 = ax2.boxplot(data_weight, labels=['banana_30', 'banana_100'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Weight (g)')
    ax2.set_title('Weight Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Ripeness 分布
    ax3 = axes[0, 2]
    ax3.hist(df30['ripeness'], bins=20, alpha=0.6, label='banana_30', color='blue', edgecolor='black')
    ax3.hist(df100['ripeness'], bins=20, alpha=0.6, label='banana_100', color='red', edgecolor='black')
    ax3.set_xlabel('Ripeness')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Ripeness Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ripeness 箱线图
    ax4 = axes[1, 0]
    data_ripeness = [df30['ripeness'], df100['ripeness']]
    bp2 = ax4.boxplot(data_ripeness, labels=['banana_30', 'banana_100'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Ripeness')
    ax4.set_title('Ripeness Box Plot')
    ax4.grid(True, alpha=0.3)
    
    # Level 分布 - 柱状图
    ax5 = axes[1, 1]
    levels = ['super', 'first', 'second', 'third']
    counts30 = [df30['level'].value_counts().get(level, 0) for level in levels]
    counts100 = [df100['level'].value_counts().get(level, 0) for level in levels]
    x = np.arange(len(levels))
    width = 0.35
    ax5.bar(x - width/2, counts30, width, label='banana_30', color='blue', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, counts100, width, label='banana_100', color='red', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Level')
    ax5.set_ylabel('Count')
    ax5.set_title('Level Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(levels)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Level 分布 - 百分比
    ax6 = axes[1, 2]
    props30 = [counts30[i] / len(df30) * 100 for i in range(len(levels))]
    props100 = [counts100[i] / len(df100) * 100 for i in range(len(levels))]
    ax6.bar(x - width/2, props30, width, label='banana_30', color='blue', alpha=0.7, edgecolor='black')
    ax6.bar(x + width/2, props100, width, label='banana_100', color='red', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Level')
    ax6.set_ylabel('Percentage (%)')
    ax6.set_title('Level Distribution (Percentage)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(levels)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n分布图已保存到: {output_path}")
    plt.close()


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 文件路径
    csv30_path = script_dir / 'banana_30.csv'
    csv100_path = script_dir / 'banana_100.csv'
    
    # 加载数据
    print("正在加载数据...")
    df30 = load_data(csv30_path)
    df100 = load_data(csv100_path)
    
    # 打印统计信息
    print_statistics(df30, "banana_30")
    print_statistics(df100, "banana_100")
    
    # 比较分布
    compare_distributions(df30, df100)
    
    # 绘制分布图
    print("\n正在生成分布图...")
    plot_distributions(df30, df100, script_dir)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()

