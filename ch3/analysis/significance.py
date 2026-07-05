import numpy as np
import re
import os
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations


def parse_results_file(file_path):
    """
    解析结果文件，提取grade相关的指标
    
    参数:
        file_path: 结果文件路径
    
    返回:
        包含各次运行指标的字典列表
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取各次运行的准确率
        accuracy_pattern = r'运行 \d+ \(种子: \d+\):\s*\n\s*测试准确率: ([\d.]+)'
        accuracy_matches = re.findall(accuracy_pattern, content)
        
        # 提取宏平均指标
        macro_precision_pattern = r'运行 \d+ \(种子: \d+\):\s*\n.*?宏平均Precision: ([\d.]+)'
        macro_precision_matches = re.findall(macro_precision_pattern, content, re.DOTALL)
        
        macro_recall_pattern = r'运行 \d+ \(种子: \d+\):\s*\n.*?宏平均Recall:\s+([\d.]+)'
        macro_recall_matches = re.findall(macro_recall_pattern, content, re.DOTALL)
        
        macro_f1_pattern = r'运行 \d+ \(种子: \d+\):\s*\n.*?宏平均F1-Score:\s+([\d.]+)'
        macro_f1_matches = re.findall(macro_f1_pattern, content, re.DOTALL)
        
        # 提取加权平均指标
        weighted_precision_pattern = r'运行 \d+ \(种子: \d+\):\s*\n.*?加权平均Precision: ([\d.]+)'
        weighted_precision_matches = re.findall(weighted_precision_pattern, content, re.DOTALL)
        
        weighted_recall_pattern = r'运行 \d+ \(种子: \d+\):\s*\n.*?加权平均Recall:\s+([\d.]+)'
        weighted_recall_matches = re.findall(weighted_recall_pattern, content, re.DOTALL)
        
        weighted_f1_pattern = r'运行 \d+ \(种子: \d+\):\s*\n.*?加权平均F1-Score:\s+([\d.]+)'
        weighted_f1_matches = re.findall(weighted_f1_pattern, content, re.DOTALL)
        
        # 如果找不到详细运行结果，尝试从汇总行提取
        if not accuracy_matches:
            # 尝试从"各次运行"行提取
            runs_line_match = re.search(r'各次运行: ([\d.,\s]+)', content)
            if runs_line_match:
                runs_str = runs_line_match.group(1)
                accuracy_matches = [x.strip() for x in runs_str.split(',')]
        
        # 确保所有指标数量一致
        num_runs = len(accuracy_matches)
        if num_runs == 0:
            print(f"警告: 无法从 {file_path} 提取数据")
            return results
        
        # 填充缺失的指标（如果某些运行没有记录）
        def pad_list(lst, target_len, default_val=0.0):
            while len(lst) < target_len:
                lst.append(default_val)
            return lst[:target_len]
        
        macro_precision_matches = pad_list(macro_precision_matches, num_runs)
        macro_recall_matches = pad_list(macro_recall_matches, num_runs)
        macro_f1_matches = pad_list(macro_f1_matches, num_runs)
        weighted_precision_matches = pad_list(weighted_precision_matches, num_runs)
        weighted_recall_matches = pad_list(weighted_recall_matches, num_runs)
        weighted_f1_matches = pad_list(weighted_f1_matches, num_runs)
        
        # 构建结果列表
        for i in range(num_runs):
            result = {
                'accuracy': float(accuracy_matches[i]),
                'macro_precision': float(macro_precision_matches[i]) if i < len(macro_precision_matches) else 0.0,
                'macro_recall': float(macro_recall_matches[i]) if i < len(macro_recall_matches) else 0.0,
                'macro_f1': float(macro_f1_matches[i]) if i < len(macro_f1_matches) else 0.0,
                'weighted_precision': float(weighted_precision_matches[i]) if i < len(weighted_precision_matches) else 0.0,
                'weighted_recall': float(weighted_recall_matches[i]) if i < len(weighted_recall_matches) else 0.0,
                'weighted_f1': float(weighted_f1_matches[i]) if i < len(weighted_f1_matches) else 0.0,
            }
            results.append(result)
        
        print(f"从 {file_path} 提取了 {len(results)} 次运行的结果")
        
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
    
    return results


def extract_metric_values(results, metric_name='accuracy'):
    """
    从结果列表中提取指定指标的值
    
    参数:
        results: 结果字典列表
        metric_name: 指标名称 ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 
                              'weighted_precision', 'weighted_recall', 'weighted_f1')
    
    返回:
        指标值数组
    """
    return np.array([r[metric_name] for r in results])


def perform_significance_test(values1, values2, metric_name="Metric", alpha=0.05):
    """
    对两组结果进行显著性检验
    
    参数:
        values1: 第一组结果（数组）
        values2: 第二组结果（数组）
        metric_name: 指标名称
        alpha: 显著性水平（默认0.05）
    
    返回:
        包含检验统计量和p值的字典
    """
    values1 = np.array(values1)
    values2 = np.array(values2)
    
    # 基本统计量
    mean1, mean2 = np.mean(values1), np.mean(values2)
    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
    
    # 判断是否配对（长度相同且可能配对）
    is_paired = len(values1) == len(values2)
    
    results = {
        'metric_name': metric_name,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'is_paired': is_paired,
        'n1': len(values1),
        'n2': len(values2)
    }
    
    if is_paired:
        # 配对t检验
        t_stat, t_pvalue = stats.ttest_rel(values1, values2)
        results['t_statistic'] = t_stat
        results['t_pvalue'] = t_pvalue
        
        # Wilcoxon符号秩检验（非参数）
        try:
            wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(values1, values2)
            results['wilcoxon_statistic'] = wilcoxon_stat
            results['wilcoxon_pvalue'] = wilcoxon_pvalue
        except ValueError:
            results['wilcoxon_statistic'] = None
            results['wilcoxon_pvalue'] = None
        
        # 配对效应量（Cohen's d）
        differences = values1 - values2
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = 0.0
        results['cohens_d'] = cohens_d
        
        # 判断显著性
        results['significant_t'] = t_pvalue < alpha
        results['significant_wilcoxon'] = results['wilcoxon_pvalue'] is not None and results['wilcoxon_pvalue'] < alpha
        
    else:
        # 独立样本t检验
        t_stat, t_pvalue = stats.ttest_ind(values1, values2)
        results['t_statistic'] = t_stat
        results['t_pvalue'] = t_pvalue
        
        # Mann-Whitney U检验（非参数）
        try:
            u_stat, u_pvalue = stats.mannwhitneyu(values1, values2, alternative='two-sided')
            results['mannwhitney_u'] = u_stat
            results['mannwhitney_pvalue'] = u_pvalue
        except ValueError:
            results['mannwhitney_u'] = None
            results['mannwhitney_pvalue'] = None
        
        # 独立样本效应量（Cohen's d）
        pooled_std = np.sqrt(((len(values1)-1)*std1**2 + (len(values2)-1)*std2**2) / 
                            (len(values1)+len(values2)-2))
        if pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        else:
            cohens_d = 0.0
        results['cohens_d'] = cohens_d
        
        # 判断显著性
        results['significant_t'] = t_pvalue < alpha
        results['significant_mannwhitney'] = results['mannwhitney_pvalue'] is not None and results['mannwhitney_pvalue'] < alpha
    
    return results


def compare_models(model_results_dict, output_dir="significance_analysis", alpha=0.05):
    """
    比较多个模型在grade分类上的性能
    
    参数:
        model_results_dict: 字典，格式为 {模型名称: 结果文件路径 或 结果列表}
        output_dir: 输出目录
        alpha: 显著性水平
    
    返回:
        比较结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析所有模型的结果
    parsed_results = {}
    for model_name, result_source in model_results_dict.items():
        if isinstance(result_source, str):
            # 如果是文件路径，解析文件
            parsed_results[model_name] = parse_results_file(result_source)
        elif isinstance(result_source, list):
            # 如果已经是结果列表，直接使用
            parsed_results[model_name] = result_source
        else:
            print(f"警告: {model_name} 的结果格式不支持")
            continue
    
    # 定义要比较的指标
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 
               'weighted_precision', 'weighted_recall', 'weighted_f1']
    
    # 存储所有比较结果
    all_comparisons = {}
    
    # 对每个指标进行两两比较
    model_names = list(parsed_results.keys())
    
    print("\n" + "="*80)
    print("模型间显著性检验结果")
    print("="*80)
    
    for metric in metrics:
        print(f"\n{'='*80}")
        print(f"指标: {metric.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        metric_comparisons = {}
        
        # 提取各模型的该指标值
        model_values = {}
        for model_name in model_names:
            if parsed_results[model_name]:
                values = extract_metric_values(parsed_results[model_name], metric)
                model_values[model_name] = values
                print(f"\n{model_name}:")
                print(f"  均值: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")
                print(f"  值: {values}")
        
        # 两两比较
        for model1, model2 in combinations(model_names, 2):
            if model1 not in model_values or model2 not in model_values:
                continue
            
            values1 = model_values[model1]
            values2 = model_values[model2]
            
            comparison_key = f"{model1}_vs_{model2}"
            test_results = perform_significance_test(values1, values2, metric, alpha)
            metric_comparisons[comparison_key] = test_results
            
            # 打印结果
            print(f"\n--- {model1} vs {model2} ---")
            print(f"均值: {test_results['mean1']:.4f} vs {test_results['mean2']:.4f}")
            print(f"标准差: {test_results['std1']:.4f} vs {test_results['std2']:.4f}")
            
            if test_results['is_paired']:
                print(f"配对t检验: t={test_results['t_statistic']:.4f}, p={test_results['t_pvalue']:.4f}")
                if test_results['significant_t']:
                    print(f"  → 差异显著 (p < {alpha})")
                else:
                    print(f"  → 差异不显著 (p >= {alpha})")
                
                if test_results['wilcoxon_pvalue'] is not None:
                    print(f"Wilcoxon检验: statistic={test_results['wilcoxon_statistic']:.4f}, p={test_results['wilcoxon_pvalue']:.4f}")
                    if test_results['significant_wilcoxon']:
                        print(f"  → 差异显著 (p < {alpha})")
                    else:
                        print(f"  → 差异不显著 (p >= {alpha})")
            else:
                print(f"独立样本t检验: t={test_results['t_statistic']:.4f}, p={test_results['t_pvalue']:.4f}")
                if test_results['significant_t']:
                    print(f"  → 差异显著 (p < {alpha})")
                else:
                    print(f"  → 差异不显著 (p >= {alpha})")
                
                if test_results.get('mannwhitney_pvalue') is not None:
                    print(f"Mann-Whitney U检验: U={test_results['mannwhitney_u']:.4f}, p={test_results['mannwhitney_pvalue']:.4f}")
                    if test_results['significant_mannwhitney']:
                        print(f"  → 差异显著 (p < {alpha})")
                    else:
                        print(f"  → 差异不显著 (p >= {alpha})")
            
            print(f"Cohen's d (效应量): {test_results['cohens_d']:.4f}")
            if abs(test_results['cohens_d']) < 0.2:
                print("  效应量: 小")
            elif abs(test_results['cohens_d']) < 0.5:
                print("  效应量: 中")
            elif abs(test_results['cohens_d']) < 0.8:
                print("  效应量: 大")
            else:
                print("  效应量: 非常大")
        
        all_comparisons[metric] = metric_comparisons
    
    # 生成可视化
    generate_comparison_plots(parsed_results, metrics, output_dir)
    
    # 保存结果到CSV
    save_comparison_results(all_comparisons, model_names, metrics, output_dir, alpha)
    
    # 生成报告
    generate_report(parsed_results, all_comparisons, model_names, metrics, output_dir, alpha)
    
    return all_comparisons


def generate_comparison_plots(parsed_results, metrics, output_dir):
    """生成比较图表"""
    model_names = list(parsed_results.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        return
    
    # 为每个指标生成箱线图和误差条图
    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准备数据
        data_to_plot = []
        labels = []
        means = []
        stds = []
        
        for model_name in model_names:
            if parsed_results[model_name]:
                values = extract_metric_values(parsed_results[model_name], metric)
                data_to_plot.append(values)
                labels.append(model_name)
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1))
        
        # 箱线图
        bp = axes[0].boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        axes[0].set_ylabel(metric.replace('_', ' ').title())
        axes[0].set_title(f'{metric.replace("_", " ").title()} - Box Plot')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 误差条图
        x_pos = np.arange(len(labels))
        axes[1].errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, markersize=8, linewidth=2)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_ylabel(metric.replace('_', ' ').title())
        axes[1].set_title(f'{metric.replace("_", " ").title()} - Mean ± Std')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        metric_filename = metric.replace('_', '_')
        plt.savefig(os.path.join(output_dir, f'{metric_filename}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 生成综合比较图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data_to_plot = []
        labels = []
        
        for model_name in model_names:
            if parsed_results[model_name]:
                values = extract_metric_values(parsed_results[model_name], metric)
                data_to_plot.append(values)
                labels.append(model_name)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_comparison_results(all_comparisons, model_names, metrics, output_dir, alpha):
    """保存比较结果到CSV"""
    rows = []
    
    for metric in metrics:
        for comparison_key, test_results in all_comparisons[metric].items():
            model1, model2 = comparison_key.split('_vs_')
            row = {
                'Metric': metric,
                'Model1': model1,
                'Model2': model2,
                'Mean1': test_results['mean1'],
                'Mean2': test_results['mean2'],
                'Std1': test_results['std1'],
                'Std2': test_results['std2'],
                'Difference': test_results['mean1'] - test_results['mean2'],
            }
            
            if test_results['is_paired']:
                row['Test_Type'] = 'Paired'
                row['T_Statistic'] = test_results['t_statistic']
                row['T_Pvalue'] = test_results['t_pvalue']
                row['T_Significant'] = test_results['significant_t']
                if test_results['wilcoxon_pvalue'] is not None:
                    row['Wilcoxon_Statistic'] = test_results['wilcoxon_statistic']
                    row['Wilcoxon_Pvalue'] = test_results['wilcoxon_pvalue']
                    row['Wilcoxon_Significant'] = test_results['significant_wilcoxon']
            else:
                row['Test_Type'] = 'Independent'
                row['T_Statistic'] = test_results['t_statistic']
                row['T_Pvalue'] = test_results['t_pvalue']
                row['T_Significant'] = test_results['significant_t']
                if test_results.get('mannwhitney_pvalue') is not None:
                    row['MannWhitney_U'] = test_results['mannwhitney_u']
                    row['MannWhitney_Pvalue'] = test_results['mannwhitney_pvalue']
                    row['MannWhitney_Significant'] = test_results['significant_mannwhitney']
            
            row['Cohens_d'] = test_results['cohens_d']
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'significance_test_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n比较结果已保存到: {csv_path}")


def generate_report(parsed_results, all_comparisons, model_names, metrics, output_dir, alpha):
    """生成文本报告"""
    report_path = os.path.join(output_dir, 'significance_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("模型间Grade分类性能显著性分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"显著性水平: α = {alpha}\n")
        f.write(f"比较的模型: {', '.join(model_names)}\n")
        f.write(f"比较的指标: {', '.join(metrics)}\n\n")
        
        # 各模型基本统计
        f.write("="*80 + "\n")
        f.write("各模型基本统计\n")
        f.write("="*80 + "\n\n")
        
        for model_name in model_names:
            if parsed_results[model_name]:
                f.write(f"{model_name}:\n")
                for metric in metrics:
                    values = extract_metric_values(parsed_results[model_name], metric)
                    f.write(f"  {metric}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}\n")
                f.write("\n")
        
        # 显著性检验结果
        f.write("="*80 + "\n")
        f.write("显著性检验结果\n")
        f.write("="*80 + "\n\n")
        
        for metric in metrics:
            f.write(f"\n指标: {metric.upper().replace('_', ' ')}\n")
            f.write("-"*80 + "\n")
            
            for comparison_key, test_results in all_comparisons[metric].items():
                model1, model2 = comparison_key.split('_vs_')
                f.write(f"\n{model1} vs {model2}:\n")
                f.write(f"  均值: {test_results['mean1']:.4f} vs {test_results['mean2']:.4f}\n")
                f.write(f"  标准差: {test_results['std1']:.4f} vs {test_results['std2']:.4f}\n")
                f.write(f"  差值: {test_results['mean1'] - test_results['mean2']:.4f}\n")
                
                if test_results['is_paired']:
                    f.write(f"  配对t检验: t={test_results['t_statistic']:.4f}, p={test_results['t_pvalue']:.4f}")
                    if test_results['significant_t']:
                        f.write(f" (显著, p < {alpha})\n")
                    else:
                        f.write(f" (不显著, p >= {alpha})\n")
                    
                    if test_results['wilcoxon_pvalue'] is not None:
                        f.write(f"  Wilcoxon检验: statistic={test_results['wilcoxon_statistic']:.4f}, p={test_results['wilcoxon_pvalue']:.4f}")
                        if test_results['significant_wilcoxon']:
                            f.write(f" (显著, p < {alpha})\n")
                        else:
                            f.write(f" (不显著, p >= {alpha})\n")
                else:
                    f.write(f"  独立样本t检验: t={test_results['t_statistic']:.4f}, p={test_results['t_pvalue']:.4f}")
                    if test_results['significant_t']:
                        f.write(f" (显著, p < {alpha})\n")
                    else:
                        f.write(f" (不显著, p >= {alpha})\n")
                    
                    if test_results.get('mannwhitney_pvalue') is not None:
                        f.write(f"  Mann-Whitney U检验: U={test_results['mannwhitney_u']:.4f}, p={test_results['mannwhitney_pvalue']:.4f}")
                        if test_results['significant_mannwhitney']:
                            f.write(f" (显著, p < {alpha})\n")
                        else:
                            f.write(f" (不显著, p >= {alpha})\n")
                
                f.write(f"  Cohen's d: {test_results['cohens_d']:.4f}\n")
                if abs(test_results['cohens_d']) < 0.2:
                    f.write("    效应量: 小\n")
                elif abs(test_results['cohens_d']) < 0.5:
                    f.write("    效应量: 中\n")
                elif abs(test_results['cohens_d']) < 0.8:
                    f.write("    效应量: 大\n")
                else:
                    f.write("    效应量: 非常大\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"分析报告已保存到: {report_path}")


if __name__ == "__main__":
    # 示例用法
    # 定义模型结果文件路径
    model_results = {
        'CIDIS': 'results/20251203-CIDIS-Test-b-1/cidis_results_summary_banana_combined.txt',
        'FruitVision-非冻结': 'results/20251203-FruitVision-Test-非冻结-b-1/fruit_vision_results_summary_banana_combined_非冻结.txt',
        'FruitVision-冻结': 'results/20251203-FruitVision-Test-冻结-b-1/fruit_vision_results_summary_banana_combined_冻结.txt',
        # 添加你自己的模型结果文件路径
        # 'YourModel': 'results/your_model_results.txt',
    }
    
    # 执行比较分析
    comparisons = compare_models(
        model_results_dict=model_results,
        output_dir='significance_analysis',
        alpha=0.05
    )
    
    print("\n分析完成！")

