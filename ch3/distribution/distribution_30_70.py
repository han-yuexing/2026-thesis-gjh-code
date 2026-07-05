import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, chi2_contingency

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


def test_distribution_shift(df30: pd.DataFrame, df70: pd.DataFrame):
    """测试分布偏移 - 统计检验"""
    print(f"\n{'='*60}")
    print("分布偏移检验 (Distribution Shift Test)")
    print(f"{'='*60}")
    
    results = {}
    
    # Weight 分布检验
    print(f"\n1. Weight (重量) 分布检验:")
    # Kolmogorov-Smirnov 检验
    ks_stat_weight, ks_p_weight = ks_2samp(df30['weight'], df70['weight'])
    print(f"   Kolmogorov-Smirnov 检验:")
    print(f"     统计量: {ks_stat_weight:.4f}, p值: {ks_p_weight:.4f}")
    if ks_p_weight < 0.05:
        print(f"     ⚠️  分布显著不同 (p < 0.05) - 存在分布偏移")
    else:
        print(f"     ✓ 分布无显著差异 (p >= 0.05) - 无分布偏移")
    
    # Mann-Whitney U 检验 (非参数检验)
    u_stat_weight, u_p_weight = mannwhitneyu(df30['weight'], df70['weight'], alternative='two-sided')
    print(f"   Mann-Whitney U 检验:")
    print(f"     统计量: {u_stat_weight:.4f}, p值: {u_p_weight:.4f}")
    if u_p_weight < 0.05:
        print(f"     ⚠️  中位数显著不同 (p < 0.05)")
    else:
        print(f"     ✓ 中位数无显著差异 (p >= 0.05)")
    
    results['weight'] = {
        'ks_stat': ks_stat_weight,
        'ks_p': ks_p_weight,
        'u_stat': u_stat_weight,
        'u_p': u_p_weight
    }
    
    # Ripeness 分布检验
    print(f"\n2. Ripeness (成熟度) 分布检验:")
    ks_stat_ripe, ks_p_ripe = ks_2samp(df30['ripeness'], df70['ripeness'])
    print(f"   Kolmogorov-Smirnov 检验:")
    print(f"     统计量: {ks_stat_ripe:.4f}, p值: {ks_p_ripe:.4f}")
    if ks_p_ripe < 0.05:
        print(f"     ⚠️  分布显著不同 (p < 0.05) - 存在分布偏移")
    else:
        print(f"     ✓ 分布无显著差异 (p >= 0.05) - 无分布偏移")
    
    u_stat_ripe, u_p_ripe = mannwhitneyu(df30['ripeness'], df70['ripeness'], alternative='two-sided')
    print(f"   Mann-Whitney U 检验:")
    print(f"     统计量: {u_stat_ripe:.4f}, p值: {u_p_ripe:.4f}")
    if u_p_ripe < 0.05:
        print(f"     ⚠️  中位数显著不同 (p < 0.05)")
    else:
        print(f"     ✓ 中位数无显著差异 (p >= 0.05)")
    
    results['ripeness'] = {
        'ks_stat': ks_stat_ripe,
        'ks_p': ks_p_ripe,
        'u_stat': u_stat_ripe,
        'u_p': u_p_ripe
    }
    
    # Level 分布检验 (卡方检验)
    print(f"\n3. Level (等级) 分布检验:")
    levels = ['super', 'first', 'second', 'third']
    # 创建交叉表
    dataset_labels = ['banana_30'] * len(df30) + ['banana_70'] * len(df70)
    level_values = list(df30['level']) + list(df70['level'])
    contingency = pd.crosstab(
        pd.Series(dataset_labels, name='dataset'),
        pd.Series(level_values, name='level')
    )
    chi2, chi2_p, dof, expected = chi2_contingency(contingency)
    print(f"   卡方检验:")
    print(f"     卡方统计量: {chi2:.4f}, p值: {chi2_p:.4f}, 自由度: {dof}")
    if chi2_p < 0.05:
        print(f"     ⚠️  等级分布显著不同 (p < 0.05) - 存在分布偏移")
    else:
        print(f"     ✓ 等级分布无显著差异 (p >= 0.05) - 无分布偏移")
    
    results['level'] = {
        'chi2': chi2,
        'chi2_p': chi2_p,
        'dof': dof
    }
    
    return results


def compare_distributions(df30: pd.DataFrame, df70: pd.DataFrame):
    """比较两个数据集的分布"""
    print(f"\n{'='*60}")
    print("分布比较")
    print(f"{'='*60}")
    
    # Weight 比较
    print(f"\nWeight (重量) 比较:")
    print(f"  banana_30:  均值={df30['weight'].mean():.2f}, 标准差={df30['weight'].std():.2f}, 范围=[{df30['weight'].min():.2f}, {df30['weight'].max():.2f}]")
    print(f"  banana_70:  均值={df70['weight'].mean():.2f}, 标准差={df70['weight'].std():.2f}, 范围=[{df70['weight'].min():.2f}, {df70['weight'].max():.2f}]")
    mean_diff = df70['weight'].mean() - df30['weight'].mean()
    std_diff = df70['weight'].std() - df30['weight'].std()
    print(f"  差异: 均值差={mean_diff:.2f}, 标准差差={std_diff:.2f}")
    if abs(mean_diff) > 5 or abs(std_diff) > 5:
        print(f"     ⚠️  差异较大，可能存在分布偏移")
    
    # Ripeness 比较
    print(f"\nRipeness (成熟度) 比较:")
    print(f"  banana_30:  均值={df30['ripeness'].mean():.4f}, 标准差={df30['ripeness'].std():.4f}, 范围=[{df30['ripeness'].min():.4f}, {df30['ripeness'].max():.4f}]")
    print(f"  banana_70:  均值={df70['ripeness'].mean():.4f}, 标准差={df70['ripeness'].std():.4f}, 范围=[{df70['ripeness'].min():.4f}, {df70['ripeness'].max():.4f}]")
    mean_diff_ripe = df70['ripeness'].mean() - df30['ripeness'].mean()
    std_diff_ripe = df70['ripeness'].std() - df30['ripeness'].std()
    print(f"  差异: 均值差={mean_diff_ripe:.4f}, 标准差差={std_diff_ripe:.4f}")
    if abs(mean_diff_ripe) > 0.02 or abs(std_diff_ripe) > 0.02:
        print(f"     ⚠️  差异较大，可能存在分布偏移")
    
    # Level 分布比较
    print(f"\nLevel (等级) 分布比较:")
    levels = ['super', 'first', 'second', 'third']
    for level in levels:
        count30 = (df30['level'] == level).sum()
        count70 = (df70['level'] == level).sum()
        prop30 = count30 / len(df30) * 100
        prop70 = count70 / len(df70) * 100
        print(f"  {level}:")
        print(f"    banana_30:  {count30:2d} ({prop30:5.1f}%)")
        print(f"    banana_70:  {count70:2d} ({prop70:5.1f}%)")
        prop_diff = prop70 - prop30
        print(f"    差异: {count70 - count30:2d} ({prop_diff:5.1f}%)")
        if abs(prop_diff) > 10:
            print(f"     ⚠️  比例差异较大，可能存在分布偏移")


def analyze_combined_distribution(df30: pd.DataFrame, df70: pd.DataFrame):
    """分析合并后的分布"""
    print(f"\n{'='*60}")
    print("合并数据集分析 (30 + 70 = 100)")
    print(f"{'='*60}")
    
    # 合并数据集
    df_combined = pd.concat([df30, df70], ignore_index=True)
    
    print(f"\n合并后样本数量: {len(df_combined)}")
    
    # 比较合并后的分布与原始30的分布
    print(f"\n合并后 vs banana_30 的分布变化:")
    
    # Weight
    mean_combined = df_combined['weight'].mean()
    mean_30 = df30['weight'].mean()
    std_combined = df_combined['weight'].std()
    std_30 = df30['weight'].std()
    print(f"  Weight: 均值 {mean_30:.2f} → {mean_combined:.2f} (变化 {mean_combined - mean_30:.2f}), "
          f"标准差 {std_30:.2f} → {std_combined:.2f} (变化 {std_combined - std_30:.2f})")
    
    # Ripeness
    mean_ripe_combined = df_combined['ripeness'].mean()
    mean_ripe_30 = df30['ripeness'].mean()
    std_ripe_combined = df_combined['ripeness'].std()
    std_ripe_30 = df30['ripeness'].std()
    print(f"  Ripeness: 均值 {mean_ripe_30:.4f} → {mean_ripe_combined:.4f} (变化 {mean_ripe_combined - mean_ripe_30:.4f}), "
          f"标准差 {std_ripe_30:.4f} → {std_ripe_combined:.4f} (变化 {std_ripe_combined - std_ripe_30:.4f})")
    
    # Level
    levels = ['super', 'first', 'second', 'third']
    print(f"  Level 分布变化:")
    for level in levels:
        prop_30 = (df30['level'] == level).sum() / len(df30) * 100
        prop_combined = (df_combined['level'] == level).sum() / len(df_combined) * 100
        print(f"    {level}: {prop_30:.1f}% → {prop_combined:.1f}% (变化 {prop_combined - prop_30:.1f}%)")
    
    return df_combined


def plot_distributions(df30: pd.DataFrame, df70: pd.DataFrame, df_combined: pd.DataFrame, output_dir: Path):
    """绘制分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Banana Dataset Distribution Comparison: 30 vs 70 (Distribution Shift Analysis)', 
                 fontsize=16, fontweight='bold')
    
    # Weight 分布
    ax1 = axes[0, 0]
    ax1.hist(df30['weight'], bins=20, alpha=0.6, label='banana_30', color='blue', edgecolor='black')
    ax1.hist(df70['weight'], bins=20, alpha=0.6, label='banana_70', color='red', edgecolor='black')
    ax1.set_xlabel('Weight (g)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Weight Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Weight 箱线图
    ax2 = axes[0, 1]
    data_weight = [df30['weight'], df70['weight']]
    bp1 = ax2.boxplot(data_weight, labels=['banana_30', 'banana_70'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Weight (g)')
    ax2.set_title('Weight Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Ripeness 分布
    ax3 = axes[0, 2]
    ax3.hist(df30['ripeness'], bins=20, alpha=0.6, label='banana_30', color='blue', edgecolor='black')
    ax3.hist(df70['ripeness'], bins=20, alpha=0.6, label='banana_70', color='red', edgecolor='black')
    ax3.set_xlabel('Ripeness')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Ripeness Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Ripeness 箱线图
    ax4 = axes[1, 0]
    data_ripeness = [df30['ripeness'], df70['ripeness']]
    bp2 = ax4.boxplot(data_ripeness, labels=['banana_30', 'banana_70'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Ripeness')
    ax4.set_title('Ripeness Box Plot')
    ax4.grid(True, alpha=0.3)
    
    # Level 分布 - 柱状图
    ax5 = axes[1, 1]
    levels = ['super', 'first', 'second', 'third']
    counts30 = [df30['level'].value_counts().get(level, 0) for level in levels]
    counts70 = [df70['level'].value_counts().get(level, 0) for level in levels]
    x = np.arange(len(levels))
    width = 0.35
    ax5.bar(x - width/2, counts30, width, label='banana_30', color='blue', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, counts70, width, label='banana_70', color='red', alpha=0.7, edgecolor='black')
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
    props70 = [counts70[i] / len(df70) * 100 for i in range(len(levels))]
    ax6.bar(x - width/2, props30, width, label='banana_30', color='blue', alpha=0.7, edgecolor='black')
    ax6.bar(x + width/2, props70, width, label='banana_70', color='red', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Level')
    ax6.set_ylabel('Percentage (%)')
    ax6.set_title('Level Distribution (Percentage)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(levels)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'distribution_30_70_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n分布图已保存到: {output_path}")
    plt.close()
    
    # 绘制合并前后的对比图
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Test Data: Before vs After', fontsize=22, fontweight='bold')
    
    # 重新配色
    color_30 = '#689F38'  # 深草绿色
    color_100 = '#1565C0'  # 深蓝色
    
    # Weight 对比 - 先绘制100（在后面），再绘制30（在前面）
    ax1 = axes2[0]
    n2, bins2, patches2 = ax1.hist(df_combined['weight'], bins=15, alpha=0.7, label='100 bananas', color=color_100, edgecolor='black')
    n1, bins1, patches1 = ax1.hist(df30['weight'], bins=15, alpha=0.7, label='30 bananas', color=color_30, edgecolor='black')
    ax1.set_xlabel('Weight (g)', fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=16)
    ax1.set_title('Weight', fontsize=18, fontweight='bold')
    # 手动创建图例，确保颜色完全一致
    legend_elements1 = [
        Rectangle((0, 0), 1, 1, facecolor=color_30, alpha=0.7, edgecolor='black', label='30 bananas'),
        Rectangle((0, 0), 1, 1, facecolor=color_100, alpha=0.7, edgecolor='black', label='100 bananas')
    ]
    ax1.legend(handles=legend_elements1, fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)
    
    # Trait (Ripeness) 对比 - 先绘制100（在后面），再绘制30（在前面）
    ax2 = axes2[1]
    n4, bins4, patches4 = ax2.hist(df_combined['ripeness'], bins=15, alpha=0.7, label='100 bananas', color=color_100, edgecolor='black')
    n3, bins3, patches3 = ax2.hist(df30['ripeness'], bins=15, alpha=0.7, label='30 bananas', color=color_30, edgecolor='black')
    ax2.set_xlabel('Trait', fontsize=16)
    ax2.set_ylabel('Frequency', fontsize=16)
    ax2.set_title('Trait', fontsize=18, fontweight='bold')
    # 手动创建图例，确保颜色完全一致
    legend_elements2 = [
        Rectangle((0, 0), 1, 1, facecolor=color_30, alpha=0.7, edgecolor='black', label='30 bananas'),
        Rectangle((0, 0), 1, 1, facecolor=color_100, alpha=0.7, edgecolor='black', label='100 bananas')
    ]
    ax2.legend(handles=legend_elements2, fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    
    # Grade (Level) 对比
    ax3 = axes2[2]
    props30_plot = [counts30[i] / len(df30) * 100 for i in range(len(levels))]
    counts_combined = [df_combined['level'].value_counts().get(level, 0) for level in levels]
    props_combined = [counts_combined[i] / len(df_combined) * 100 for i in range(len(levels))]
    ax3.bar(x - width/2, props30_plot, width, label='30 bananas', color=color_30, alpha=0.7, edgecolor='black')
    ax3.bar(x + width/2, props_combined, width, label='100 bananas', color=color_100, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Grade', fontsize=16)
    ax3.set_ylabel('Percentage (%)', fontsize=16)
    ax3.set_title('Grade', fontsize=18, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(levels, fontsize=14)
    ax3.legend(fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=14)
    
    plt.tight_layout()
    output_path2 = output_dir / 'distribution_shift_analysis.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"分布偏移分析图已保存到: {output_path2}")
    plt.close()


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 文件路径
    csv30_path = script_dir / 'banana_30.csv'
    csv70_path = script_dir / 'banana_70.csv'
    
    # 加载数据
    print("正在加载数据...")
    df30 = load_data(csv30_path)
    df70 = load_data(csv70_path)
    
    # 打印统计信息
    print_statistics(df30, "banana_30")
    print_statistics(df70, "banana_70")
    
    # 比较分布
    compare_distributions(df30, df70)
    
    # 分布偏移检验
    shift_results = test_distribution_shift(df30, df70)
    
    # 分析合并后的分布
    df_combined = analyze_combined_distribution(df30, df70)
    
    # 绘制分布图
    print("\n正在生成分布图...")
    plot_distributions(df30, df70, df_combined, script_dir)
    
    # 总结
    print(f"\n{'='*60}")
    print("分布偏移总结")
    print(f"{'='*60}")
    print("\n是否引起分布偏移？")
    
    has_shift = False
    if shift_results['weight']['ks_p'] < 0.05:
        print("  ⚠️  Weight: 存在分布偏移 (p < 0.05)")
        has_shift = True
    else:
        print("  ✓ Weight: 无分布偏移")
    
    if shift_results['ripeness']['ks_p'] < 0.05:
        print("  ⚠️  Ripeness: 存在分布偏移 (p < 0.05)")
        has_shift = True
    else:
        print("  ✓ Ripeness: 无分布偏移")
    
    if shift_results['level']['chi2_p'] < 0.05:
        print("  ⚠️  Level: 存在分布偏移 (p < 0.05)")
        has_shift = True
    else:
        print("  ✓ Level: 无分布偏移")
    
    if has_shift:
        print("\n结论: ⚠️  合并 banana_70 到 banana_30 引起了分布偏移")
        print("建议: 需要关注模型在合并数据集上的性能变化")
    else:
        print("\n结论: ✓ 合并 banana_70 到 banana_30 未引起显著分布偏移")
        print("建议: 合并后的数据集分布相对稳定")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()

