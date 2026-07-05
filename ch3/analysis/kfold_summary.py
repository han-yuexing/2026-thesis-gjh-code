#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K折交叉验证结果汇总脚本

从各折的test.log文件中提取指标，计算均值和标准差
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path


def parse_log_file(log_file_path):
    """
    解析test.log文件，提取指标数据
    
    参数:
        log_file_path: log文件路径
    
    返回:
        包含所有指标的字典
    """
    metrics = {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取Weight R2
        weight_r2_match = re.search(r'Weight R2: ([\d.]+)', content)
        if weight_r2_match:
            metrics['weight_r2'] = float(weight_r2_match.group(1))
        
        # 提取Weight RMSE
        weight_rmse_match = re.search(r'Weight RMSE: ([\d.]+)', content)
        if weight_rmse_match:
            metrics['weight_rmse'] = float(weight_rmse_match.group(1))
        
        # 提取Curvature R2 (黄瓜)
        curvature_r2_match = re.search(r'Curvature R2: ([\d.]+)', content)
        if curvature_r2_match:
            metrics['curvature_r2'] = float(curvature_r2_match.group(1))
        
        # 提取Curvature RMSE (黄瓜)
        curvature_rmse_match = re.search(r'Curvature RMSE: ([\d.]+)', content)
        if curvature_rmse_match:
            metrics['curvature_rmse'] = float(curvature_rmse_match.group(1))
        
        # 提取Ripeness R2 (香蕉)
        ripeness_r2_match = re.search(r'Ripeness R2: ([\d.]+)', content)
        if ripeness_r2_match:
            metrics['ripeness_r2'] = float(ripeness_r2_match.group(1))
        
        # 提取Ripeness RMSE (香蕉)
        ripeness_rmse_match = re.search(r'Ripeness RMSE: ([\d.]+)', content)
        if ripeness_rmse_match:
            metrics['ripeness_rmse'] = float(ripeness_rmse_match.group(1))
        
        # 提取分类指标
        level_precision_match = re.search(r'Level Precision: ([\d.]+)', content)
        if level_precision_match:
            metrics['level_precision'] = float(level_precision_match.group(1))
        
        level_recall_match = re.search(r'Level Recall: ([\d.]+)', content)
        if level_recall_match:
            metrics['level_recall'] = float(level_recall_match.group(1))
        
        level_f1_match = re.search(r'Level F1 Score: ([\d.]+)', content)
        if level_f1_match:
            metrics['level_f1'] = float(level_f1_match.group(1))
        
        level_accuracy_match = re.search(r'Level Accuracy: ([\d.]+)', content)
        if level_accuracy_match:
            metrics['level_accuracy'] = float(level_accuracy_match.group(1))
        
        # 提取Level mAP (如果有)
        level_map_match = re.search(r'Level mAP: ([\d.]+)', content)
        if level_map_match:
            metrics['level_map'] = float(level_map_match.group(1))
        
    except Exception as e:
        print(f"解析文件 {log_file_path} 时出错: {e}")
    
    return metrics


def summarize_kfold_results(base_path, output_file=None):
    """
    汇总K折交叉验证结果
    
    参数:
        base_path: 结果文件夹路径（包含figures/fold1/test/test_fold1.log等）
        output_file: 输出文件路径（可选）
    
    返回:
        汇总结果字典
    """
    base_path = Path(base_path)
    figures_path = base_path / 'figures'
    
    if not figures_path.exists():
        print(f"错误: 找不到figures文件夹: {figures_path}")
        return None
    
    # 收集所有折的数据
    all_fold_metrics = []
    
    # 遍历所有折
    for fold_num in range(1, 6):
        fold_name = f'fold{fold_num}'
        log_file = figures_path / fold_name / 'test' / f'test_{fold_name}.log'
        
        if log_file.exists():
            print(f"读取 {fold_name} 的结果...")
            metrics = parse_log_file(log_file)
            if metrics:
                metrics['fold'] = fold_num
                all_fold_metrics.append(metrics)
                print(f"  {fold_name}: 成功提取 {len(metrics)-1} 个指标")
            else:
                print(f"  {fold_name}: 未能提取指标")
        else:
            print(f"警告: 找不到文件 {log_file}")
    
    if not all_fold_metrics:
        print("错误: 没有找到任何有效的结果数据")
        return None
    
    # 确定指标名称（排除'fold'）
    metric_names = [k for k in all_fold_metrics[0].keys() if k != 'fold']
    
    # 提取各指标的值
    summary = {}
    summary['folds'] = len(all_fold_metrics)
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_fold_metrics if metric_name in m]
        if values:
            summary[metric_name] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # 生成报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("K折交叉验证结果汇总")
    report_lines.append("="*80)
    report_lines.append(f"\n数据来源: {base_path}")
    report_lines.append(f"折数: {summary['folds']}")
    report_lines.append("\n" + "="*80)
    report_lines.append("指标统计（均值 ± 标准差）")
    report_lines.append("="*80)
    
    # 按类别组织指标
    regression_metrics = []
    classification_metrics = []
    
    for metric_name in sorted(metric_names):
        if metric_name in summary:
            mean_val = summary[metric_name]['mean']
            std_val = summary[metric_name]['std']
            min_val = summary[metric_name]['min']
            max_val = summary[metric_name]['max']
            
            # 格式化指标名称
            display_name = metric_name.replace('_', ' ').title()
            
            report_lines.append(f"\n{display_name}:")
            report_lines.append(f"  均值: {mean_val:.4f} ± {std_val:.4f}")
            report_lines.append(f"  范围: [{min_val:.4f}, {max_val:.4f}]")
            report_lines.append(f"  各折值: {[f'{v:.4f}' for v in summary[metric_name]['values']]}")
            
            if 'r2' in metric_name.lower() or 'rmse' in metric_name.lower():
                regression_metrics.append(metric_name)
            else:
                classification_metrics.append(metric_name)
    
    # 详细表格
    report_lines.append("\n" + "="*80)
    report_lines.append("各折详细指标")
    report_lines.append("="*80)
    
    # 创建表格头部
    header = f"{'Fold':<6}"
    for metric_name in sorted(metric_names):
        display_name = metric_name.replace('_', ' ').title()
        # 截断过长的名称
        if len(display_name) > 12:
            display_name = display_name[:12]
        header += f" {display_name:<12}"
    report_lines.append(header)
    report_lines.append("-" * (6 + 13 * len(metric_names)))
    
    # 添加各折数据
    for fold_metrics in all_fold_metrics:
        row = f"{fold_metrics['fold']:<6}"
        for metric_name in sorted(metric_names):
            if metric_name in fold_metrics:
                row += f" {fold_metrics[metric_name]:<12.4f}"
            else:
                row += f" {'N/A':<12}"
        report_lines.append(row)
    
    # 添加均值行
    row = f"{'Mean':<6}"
    for metric_name in sorted(metric_names):
        if metric_name in summary:
            row += f" {summary[metric_name]['mean']:<12.4f}"
        else:
            row += f" {'N/A':<12}"
    report_lines.append(row)
    
    # 添加标准差行
    row = f"{'Std':<6}"
    for metric_name in sorted(metric_names):
        if metric_name in summary:
            row += f" {summary[metric_name]['std']:<12.4f}"
        else:
            row += f" {'N/A':<12}"
    report_lines.append(row)
    
    report_lines.append("\n" + "="*80)
    report_lines.append("报告结束")
    report_lines.append("="*80)
    
    # 保存报告
    if output_file is None:
        output_file = base_path / 'kfold_summary.txt'
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n汇总报告已保存到: {output_file}")
    
    # 保存CSV文件
    csv_file = base_path / 'kfold_summary.csv'
    save_to_csv(all_fold_metrics, summary, metric_names, csv_file)
    print(f"CSV文件已保存到: {csv_file}")
    
    return summary


def save_to_csv(all_fold_metrics, summary, metric_names, csv_file):
    """保存结果到CSV文件"""
    rows = []
    
    # 添加各折数据
    for fold_metrics in all_fold_metrics:
        row = {'Fold': fold_metrics['fold']}
        for metric_name in sorted(metric_names):
            if metric_name in fold_metrics:
                row[metric_name] = fold_metrics[metric_name]
            else:
                row[metric_name] = None
        rows.append(row)
    
    # 添加均值行
    row = {'Fold': 'Mean'}
    for metric_name in sorted(metric_names):
        if metric_name in summary:
            row[metric_name] = summary[metric_name]['mean']
        else:
            row[metric_name] = None
    rows.append(row)
    
    # 添加标准差行
    row = {'Fold': 'Std'}
    for metric_name in sorted(metric_names):
        if metric_name in summary:
            row[metric_name] = summary[metric_name]['std']
        else:
            row[metric_name] = None
    rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, encoding='utf-8')


if __name__ == "__main__":
    import sys
    
    # 默认路径
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        # 使用Cucumber的K折结果
        base_path = "results/20251031-CucumberModel-KFold-c-1"
    
    print(f"汇总K折交叉验证结果: {base_path}")
    print("="*80)
    
    summary = summarize_kfold_results(base_path)
    
    if summary:
        print("\n汇总完成！")
        print(f"共处理 {summary['folds']} 折数据")
    else:
        print("\n汇总失败！")

