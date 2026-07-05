#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cucumber模型Wilcoxon signed-rank显著性分析

对以下模型进行两两比较：
1. CucumberModel (验证集准确率)
2. CIDIS-Multi (测试集准确率)
3. FruitVision-Multi-Simple-c-1 (测试集准确率，冻结骨干)
4. FruitVision-Multi-Simple-c-2 (测试集准确率，非冻结骨干)
"""

import numpy as np
from scipy import stats
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义各模型的准确率数据（10次运行）
model_data = {
    'CucumberModel': np.array([0.9231, 0.8974, 1.0000, 0.9744, 0.9487, 0.9231, 0.8718, 0.9744, 0.9744, 0.9487]),
    'CIDIS-Multi': np.array([0.8400, 0.8400, 0.8400, 0.8400, 0.8400, 0.8400, 0.8800, 0.8800, 0.8000, 0.9600]),
    'FruitVision-Multi-冻结': np.array([0.6000, 0.6800, 0.5600, 0.7600, 0.7600, 0.7200, 0.7600, 0.6400, 0.6400, 0.8000]),
    'FruitVision-Multi-非冻结': np.array([0.8000, 0.6000, 0.9200, 0.6800, 0.4000, 0.3200, 0.5200, 0.4800, 0.4800, 0.2000]),
}

def perform_wilcoxon_test(values1, values2, model1_name, model2_name, alpha=0.05):
    """
    执行Wilcoxon signed-rank检验
    
    参数:
        values1: 第一组数据
        values2: 第二组数据
        model1_name: 第一个模型名称
        model2_name: 第二个模型名称
        alpha: 显著性水平
    
    返回:
        包含检验结果的字典
    """
    # 基本统计量
    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
    median1, median2 = np.median(values1), np.median(values2)
    
    # 计算差值
    differences = values1 - values2
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    
    # Wilcoxon signed-rank检验
    try:
        statistic, pvalue = stats.wilcoxon(values1, values2, alternative='two-sided')
    except ValueError as e:
        # 如果所有差值都相同，无法进行检验
        print(f"警告: {model1_name} vs {model2_name} 无法进行Wilcoxon检验: {e}")
        statistic, pvalue = None, None
    
    # 判断显著性
    significant = pvalue is not None and pvalue < alpha
    
    # 计算效应量 (r = Z / sqrt(N))
    if statistic is not None and pvalue is not None:
        # 使用近似Z统计量
        n = len(values1)
        z_stat = stats.norm.ppf(pvalue / 2) if pvalue < 1.0 else 0
        effect_size = abs(z_stat) / np.sqrt(n)
    else:
        effect_size = None
    
    # 判断效应量大小
    if effect_size is not None:
        if effect_size < 0.1:
            effect_size_interpretation = "可忽略"
        elif effect_size < 0.3:
            effect_size_interpretation = "小"
        elif effect_size < 0.5:
            effect_size_interpretation = "中"
        else:
            effect_size_interpretation = "大"
    else:
        effect_size_interpretation = None
    
    return {
        'model1': model1_name,
        'model2': model2_name,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'median1': median1,
        'median2': median2,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': significant,
        'effect_size': effect_size,
        'effect_size_interpretation': effect_size_interpretation
    }

def main():
    """主函数"""
    alpha = 0.05
    output_dir = 'wilcoxon_cucumber_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Cucumber模型Wilcoxon Signed-Rank显著性分析")
    print("="*80)
    print(f"\n显著性水平: α = {alpha}")
    print(f"\n各模型基本统计:")
    print("-"*80)
    
    # 打印各模型基本统计
    for model_name, values in model_data.items():
        print(f"\n{model_name}:")
        print(f"  均值: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
        print(f"  中位数: {np.median(values):.4f}")
        print(f"  范围: [{np.min(values):.4f}, {np.max(values):.4f}]")
        print(f"  数据: {values}")
    
    # 进行两两比较
    print("\n" + "="*80)
    print("Wilcoxon Signed-Rank检验结果 (两两比较)")
    print("="*80)
    
    model_names = list(model_data.keys())
    all_results = []
    
    for model1, model2 in combinations(model_names, 2):
        values1 = model_data[model1]
        values2 = model_data[model2]
        
        result = perform_wilcoxon_test(values1, values2, model1, model2, alpha)
        all_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"{model1} vs {model2}")
        print(f"{'='*80}")
        print(f"均值: {result['mean1']:.4f} vs {result['mean2']:.4f} (差值: {result['mean_diff']:.4f})")
        print(f"中位数: {result['median1']:.4f} vs {result['median2']:.4f} (差值: {result['median_diff']:.4f})")
        print(f"标准差: {result['std1']:.4f} vs {result['std2']:.4f}")
        
        if result['statistic'] is not None:
            print(f"\nWilcoxon Signed-Rank检验:")
            print(f"  统计量 (W): {result['statistic']:.4f}")
            print(f"  p值: {result['pvalue']:.6f}")
            
            if result['significant']:
                print(f"  结果: 差异显著 (p < {alpha})")
                if result['mean1'] > result['mean2']:
                    print(f"  → {model1} 显著优于 {model2}")
                else:
                    print(f"  → {model2} 显著优于 {model1}")
            else:
                print(f"  结果: 差异不显著 (p >= {alpha})")
            
            if result['effect_size'] is not None:
                print(f"  效应量 (r): {result['effect_size']:.4f} ({result['effect_size_interpretation']})")
        else:
            print(f"\nWilcoxon检验: 无法执行（可能所有差值都相同）")
    
    # 保存结果到CSV
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'wilcoxon_test_results.csv')
    df_results.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n\n结果已保存到: {csv_path}")
    
    # 生成文本报告
    report_path = os.path.join(output_dir, 'wilcoxon_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Cucumber模型Wilcoxon Signed-Rank显著性分析报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"显著性水平: α = {alpha}\n\n")
        
        f.write("各模型基本统计\n")
        f.write("-"*80 + "\n")
        for model_name, values in model_data.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  均值: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}\n")
            f.write(f"  中位数: {np.median(values):.4f}\n")
            f.write(f"  范围: [{np.min(values):.4f}, {np.max(values):.4f}]\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Wilcoxon Signed-Rank检验结果\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            f.write(f"{result['model1']} vs {result['model2']}\n")
            f.write("-"*80 + "\n")
            f.write(f"均值: {result['mean1']:.4f} vs {result['mean2']:.4f} (差值: {result['mean_diff']:.4f})\n")
            f.write(f"中位数: {result['median1']:.4f} vs {result['median2']:.4f} (差值: {result['median_diff']:.4f})\n")
            f.write(f"标准差: {result['std1']:.4f} vs {result['std2']:.4f}\n")
            
            if result['statistic'] is not None:
                f.write(f"\nWilcoxon检验:\n")
                f.write(f"  统计量 (W): {result['statistic']:.4f}\n")
                f.write(f"  p值: {result['pvalue']:.6f}\n")
                
                if result['significant']:
                    f.write(f"  结果: 差异显著 (p < {alpha})\n")
                    if result['mean1'] > result['mean2']:
                        f.write(f"  → {result['model1']} 显著优于 {result['model2']}\n")
                    else:
                        f.write(f"  → {result['model2']} 显著优于 {result['model1']}\n")
                else:
                    f.write(f"  结果: 差异不显著 (p >= {alpha})\n")
                
                if result['effect_size'] is not None:
                    f.write(f"  效应量 (r): {result['effect_size']:.4f} ({result['effect_size_interpretation']})\n")
            else:
                f.write(f"\nWilcoxon检验: 无法执行\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"分析报告已保存到: {report_path}")
    
    # 生成可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 箱线图
    ax = axes[0]
    data_to_plot = [model_data[name] for name in model_names]
    bp = ax.boxplot(data_to_plot, labels=model_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_ylabel('准确率')
    ax.set_title('各模型准确率分布 (箱线图)')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 误差条图
    ax = axes[1]
    means = [np.mean(model_data[name]) for name in model_names]
    stds = [np.std(model_data[name], ddof=1) for name in model_names]
    x_pos = np.arange(len(model_names))
    ax.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, markersize=10, linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_ylabel('准确率')
    ax.set_title('各模型准确率均值 ± 标准差')
    ax.grid(True, alpha=0.3)
    
    # 小提琴图（显示分布）
    ax = axes[2]
    parts = ax.violinplot(data_to_plot, positions=range(len(model_names)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45)
    ax.set_ylabel('准确率')
    ax.set_title('各模型准确率分布 (小提琴图)')
    ax.grid(True, alpha=0.3)
    
    # 条形图（均值）
    ax = axes[3]
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_ylabel('准确率')
    ax.set_title('各模型准确率均值 ± 标准差 (条形图)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {plot_path}")
    
    # 生成显著性矩阵图
    create_significance_matrix(all_results, model_names, output_dir)
    
    print(f"\n所有结果已保存到: {output_dir}/")
    print("\n分析完成！")

def create_significance_matrix(all_results, model_names, output_dir):
    """创建显著性矩阵图"""
    n = len(model_names)
    significance_matrix = np.full((n, n), np.nan)
    pvalue_matrix = np.full((n, n), np.nan)
    
    # 创建模型名称到索引的映射
    name_to_idx = {name: idx for idx, name in enumerate(model_names)}
    
    # 填充矩阵
    for result in all_results:
        idx1 = name_to_idx[result['model1']]
        idx2 = name_to_idx[result['model2']]
        
        if result['pvalue'] is not None:
            pvalue_matrix[idx1, idx2] = result['pvalue']
            pvalue_matrix[idx2, idx1] = result['pvalue']
            
            if result['significant']:
                # 如果model1优于model2，标记为1，否则为-1
                if result['mean1'] > result['mean2']:
                    significance_matrix[idx1, idx2] = 1
                    significance_matrix[idx2, idx1] = -1
                else:
                    significance_matrix[idx1, idx2] = -1
                    significance_matrix[idx2, idx1] = 1
            else:
                significance_matrix[idx1, idx2] = 0
                significance_matrix[idx2, idx1] = 0
    
    # 绘制显著性矩阵
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # p值矩阵
    im1 = axes[0].imshow(pvalue_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.05)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_yticklabels(model_names)
    axes[0].set_title('Wilcoxon检验 p值矩阵')
    
    # 添加数值标注
    for i in range(n):
        for j in range(n):
            if not np.isnan(pvalue_matrix[i, j]):
                text = axes[0].text(j, i, f'{pvalue_matrix[i, j]:.4f}',
                                   ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=axes[0], label='p值')
    
    # 显著性矩阵
    im2 = axes[1].imshow(significance_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_yticklabels(model_names)
    axes[1].set_title('显著性矩阵 (1=行优于列, -1=列优于行, 0=无显著差异)')
    
    # 添加数值标注
    for i in range(n):
        for j in range(n):
            if not np.isnan(significance_matrix[i, j]):
                val = significance_matrix[i, j]
                if val == 1:
                    text = '✓'
                elif val == -1:
                    text = '✗'
                else:
                    text = '—'
                axes[1].text(j, i, text, ha="center", va="center", 
                           color="white" if abs(val) > 0 else "black", 
                           fontsize=14, fontweight='bold')
    
    plt.colorbar(im2, ax=axes[1], label='显著性')
    
    plt.tight_layout()
    matrix_path = os.path.join(output_dir, 'significance_matrix.png')
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"显著性矩阵已保存到: {matrix_path}")

if __name__ == "__main__":
    main()

