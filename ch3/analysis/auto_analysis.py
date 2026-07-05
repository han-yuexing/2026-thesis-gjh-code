import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error


def parse_log_file(log_file_path):
    """解析test.log文件，提取指标数据"""
    data = {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取true_weights
        true_weights_match = re.search(r'true_weights: \[(.*?)\]', content)
        if true_weights_match:
            data['true_weights'] = [float(x.strip()) for x in true_weights_match.group(1).split(',')]
            
        # 提取pred_weights
        pred_weights_match = re.search(r'pred_weights: \[(.*?)\]', content)
        if pred_weights_match:
            data['pred_weights'] = [float(x.strip()) for x in pred_weights_match.group(1).split(',')]
        
        # 提取true_curvatures（黄瓜）
        true_curvatures_match = re.search(r'true_curvatures: \[(.*?)\]', content)
        if true_curvatures_match:
            data['true_curvatures'] = [float(x.strip()) for x in true_curvatures_match.group(1).split(',')]
            
        # 提取pred_curvatures（黄瓜）
        pred_curvatures_match = re.search(r'pred_curvatures: \[(.*?)\]', content)
        if pred_curvatures_match:
            data['pred_curvatures'] = [float(x.strip()) for x in pred_curvatures_match.group(1).split(',')]
        
        # 提取true_ripeness（香蕉）
        true_ripeness_match = re.search(r'true_ripeness: \[(.*?)\]', content)
        if true_ripeness_match:
            data['true_ripeness'] = [float(x.strip()) for x in true_ripeness_match.group(1).split(',')]
        
        # 提取pred_ripeness（香蕉）
        pred_ripeness_match = re.search(r'pred_ripeness: \[(.*?)\]', content)
        if pred_ripeness_match:
            data['pred_ripeness'] = [float(x.strip()) for x in pred_ripeness_match.group(1).split(',')]
            
        # 提取true_levels
        true_levels_match = re.search(r'true_levels: \[(.*?)\]', content)
        if true_levels_match:
            data['true_levels'] = [int(x.strip()) for x in true_levels_match.group(1).split(',')]
            
        # 提取pred_levels
        pred_levels_match = re.search(r'pred_levels: \[(.*?)\]', content)
        if pred_levels_match:
            data['pred_levels'] = [int(x.strip()) for x in pred_levels_match.group(1).split(',')]
            
        # 提取指标值（重量）
        weight_r2_match = re.search(r'Weight R2: ([\d.]+)', content)
        if weight_r2_match:
            data['weight_r2'] = float(weight_r2_match.group(1))
            
        weight_rmse_match = re.search(r'Weight RMSE: ([\d.]+)', content)
        if weight_rmse_match:
            data['weight_rmse'] = float(weight_rmse_match.group(1))
            
        # 提取指标值（曲率-黄瓜）
        curvature_r2_match = re.search(r'Curvature R2: ([\d.]+)', content)
        if curvature_r2_match:
            data['curvature_r2'] = float(curvature_r2_match.group(1))
            
        curvature_rmse_match = re.search(r'Curvature RMSE: ([\d.]+)', content)
        if curvature_rmse_match:
            data['curvature_rmse'] = float(curvature_rmse_match.group(1))
        
        # 提取指标值（成熟度-香蕉）
        ripeness_r2_match = re.search(r'Ripeness R2: ([\d.]+)', content)
        if ripeness_r2_match:
            data['ripeness_r2'] = float(ripeness_r2_match.group(1))
            
        ripeness_rmse_match = re.search(r'Ripeness RMSE: ([\d.]+)', content)
        if ripeness_rmse_match:
            data['ripeness_rmse'] = float(ripeness_rmse_match.group(1))
            
        # 提取分类指标
        level_precision_match = re.search(r'Level Precision: ([\d.]+)', content)
        if level_precision_match:
            data['level_precision'] = float(level_precision_match.group(1))
            
        level_recall_match = re.search(r'Level Recall: ([\d.]+)', content)
        if level_recall_match:
            data['level_recall'] = float(level_recall_match.group(1))
            
        level_f1_match = re.search(r'Level F1 Score: ([\d.]+)', content)
        if level_f1_match:
            data['level_f1'] = float(level_f1_match.group(1))
            
        level_accuracy_match = re.search(r'Level Accuracy: ([\d.]+)', content)
        if level_accuracy_match:
            data['level_accuracy'] = float(level_accuracy_match.group(1))
            
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
        
    return data

def calculate_metrics_from_data(true_vals, pred_vals, true_labels, pred_labels):
    """从原始数据计算指标"""
    # 回归指标
    r2 = r2_score(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    
    # 分类指标
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    return r2, rmse, precision, recall, f1, accuracy

def main():
    # 基础路径
    base_path = "save/0707-b-resnet18-train1-weight_only"
    
    # 创建final文件夹
    final_path = os.path.join(base_path, "final")
    os.makedirs(final_path, exist_ok=True)
    print(f"结果将保存到: {final_path}")
    
    # 存储所有折的数据
    all_fold_data = []
    all_metrics = []
    
    # 动态辅助标签（黄瓜: Curvature；香蕉: Ripeness）
    aux_label = None
    
    # 读取每一折的数据
    for fold in range(1, 6):
        log_file = os.path.join(base_path, str(fold), "test.log")
        print(f"Reading fold {fold} from: {log_file}")
        
        if os.path.exists(log_file):
            fold_data = parse_log_file(log_file)
            if fold_data:
                all_fold_data.append(fold_data)
                
                # 计算指标
                if all(k in fold_data for k in ['true_weights', 'pred_weights', 'true_levels', 'pred_levels']):
                    w_r2, w_rmse, precision, recall, f1, accuracy = calculate_metrics_from_data(
                        fold_data['true_weights'], fold_data['pred_weights'], 
                        fold_data['true_levels'], fold_data['pred_levels']
                    )
                    
                    # 选择辅助目标（优先使用已有字段）
                    # 黄瓜（曲率）
                    if ('true_curvatures' in fold_data and 'pred_curvatures' in fold_data) or \
                       ('curvature_r2' in fold_data and 'curvature_rmse' in fold_data):
                        if aux_label is None:
                            aux_label = 'Curvature'
                        # 使用log文件中的曲率指标，如果没有则计算
                        if 'curvature_r2' in fold_data and 'curvature_rmse' in fold_data:
                            aux_r2 = fold_data['curvature_r2']
                            aux_rmse = fold_data['curvature_rmse']
                        else:
                            aux_r2, aux_rmse, _, _, _, _ = calculate_metrics_from_data(
                                fold_data['true_curvatures'], fold_data['pred_curvatures'],
                                fold_data['true_levels'], fold_data['pred_levels']
                            )
                    # 香蕉（成熟度）
                    elif ('true_ripeness' in fold_data and 'pred_ripeness' in fold_data) or \
                         ('ripeness_r2' in fold_data and 'ripeness_rmse' in fold_data):
                        if aux_label is None:
                            aux_label = 'Ripeness'
                        if 'ripeness_r2' in fold_data and 'ripeness_rmse' in fold_data:
                            aux_r2 = fold_data['ripeness_r2']
                            aux_rmse = fold_data['ripeness_rmse']
                        else:
                            aux_r2, aux_rmse, _, _, _, _ = calculate_metrics_from_data(
                                fold_data['true_ripeness'], fold_data['pred_ripeness'],
                                fold_data['true_levels'], fold_data['pred_levels']
                            )
                    else:
                        # 兜底：如果无法识别，使用0并标记
                        if aux_label is None:
                            aux_label = 'Aux'
                        aux_r2, aux_rmse = 0.0, 0.0
                    
                    # 使用log文件中weight指标，如果存在则覆盖
                    if 'weight_r2' in fold_data:
                        w_r2 = fold_data['weight_r2']
                    if 'weight_rmse' in fold_data:
                        w_rmse = fold_data['weight_rmse']
                    
                    # 使用log文件中的level指标，如果存在则覆盖
                    if 'level_precision' in fold_data:
                        precision = fold_data['level_precision']
                    if 'level_recall' in fold_data:
                        recall = fold_data['level_recall']
                    if 'level_f1' in fold_data:
                        f1 = fold_data['level_f1']
                    if 'level_accuracy' in fold_data:
                        accuracy = fold_data['level_accuracy']
                    
                    metrics = {
                        'fold': fold,
                        'weight_r2': w_r2,
                        'weight_rmse': w_rmse,
                        'aux_r2': aux_r2,
                        'aux_rmse': aux_rmse,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy
                    }
                    all_metrics.append(metrics)
                    print(f"Fold {fold} metrics loaded successfully")
                else:
                    print(f"Fold {fold} missing required data")
            else:
                print(f"Fold {fold} data parsing failed")
        else:
            print(f"Log file not found: {log_file}")
    
    if not all_metrics:
        print("No valid data found!")
        return
    
    # 转换为numpy数组
    metrics_array = np.array([[m['weight_r2'], m['weight_rmse'], m['aux_r2'], m['aux_rmse'], 
                              m['precision'], m['recall'], m['f1'], m['accuracy']] for m in all_metrics])
    
    # 计算均值和标准差
    means = np.mean(metrics_array, axis=0)
    stds = np.std(metrics_array, axis=0)
    
    # 打印指标统计
    print("\n" + "="*50)
    print("指标统计（均值±标准差）")
    print("="*50)
    print(f"Weight R²: {means[0]:.4f} ± {stds[0]:.4f}")
    print(f"Weight RMSE: {means[1]:.4f} ± {stds[1]:.4f}")
    print(f"{aux_label} R²: {means[2]:.4f} ± {stds[2]:.4f}")
    print(f"{aux_label} RMSE: {means[3]:.4f} ± {stds[3]:.4f}")
    print(f"Level Precision: {means[4]:.4f} ± {stds[4]:.4f}")
    print(f"Level Recall: {means[5]:.4f} ± {stds[5]:.4f}")
    print(f"Level F1 Score: {means[6]:.4f} ± {stds[6]:.4f}")
    print(f"Level Accuracy: {means[7]:.4f} ± {stds[7]:.4f}")
    
    # 绘图
    plt.rcParams['font.size'] = 10
    
    # 1. Weight R²拟合图（以第1折为例）
    if all_fold_data:
        plt.figure(figsize=(6,6))
        plt.scatter(all_fold_data[0]['true_weights'], all_fold_data[0]['pred_weights'], 
                   label='Predicted', color='b', alpha=0.7)
        true_w = all_fold_data[0]['true_weights']
        plt.plot([min(true_w), max(true_w)], [min(true_w), max(true_w)], 'r--', label='Ideal Fit')
        plt.xlabel('True Weight')
        plt.ylabel('Predicted Weight')
        plt.title('Weight R² Fitting Plot (Fold 1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(final_path, 'weight_r2_fitting.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 辅助目标 R²拟合图（以第1折为例）
        if aux_label == 'Curvature' and 'true_curvatures' in all_fold_data[0] and 'pred_curvatures' in all_fold_data[0]:
            true_aux = all_fold_data[0]['true_curvatures']
            pred_aux = all_fold_data[0]['pred_curvatures']
        elif aux_label == 'Ripeness' and 'true_ripeness' in all_fold_data[0] and 'pred_ripeness' in all_fold_data[0]:
            true_aux = all_fold_data[0]['true_ripeness']
            pred_aux = all_fold_data[0]['pred_ripeness']
        else:
            true_aux, pred_aux = None, None
        
        if true_aux is not None and pred_aux is not None:
            plt.figure(figsize=(6,6))
            plt.scatter(true_aux, pred_aux, 
                       label='Predicted', color='g', alpha=0.7)
            plt.plot([min(true_aux), max(true_aux)], [min(true_aux), max(true_aux)], 'r--', label='Ideal Fit')
            plt.xlabel(f'True {aux_label}')
            plt.ylabel(f'Predicted {aux_label}')
            plt.title(f'{aux_label} R² Fitting Plot (Fold 1)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            aux_fit_name = f"{aux_label.lower()}_r2_fitting.png"
            plt.savefig(os.path.join(final_path, aux_fit_name), dpi=300, bbox_inches='tight')
            plt.show()
    
    # 3. 所有指标的errorbar图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Weight R²
    axes[0,0].errorbar(range(1, len(all_metrics)+1), metrics_array[:, 0], yerr=stds[0], 
                       fmt='o-', capsize=5, color='blue', markersize=8)
    axes[0,0].set_xlabel('Fold')
    axes[0,0].set_ylabel('Weight R²')
    axes[0,0].set_title('Weight R² Mean ± Std')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0.9, 1.0)
    
    # 辅助目标 R²
    axes[0,1].errorbar(range(1, len(all_metrics)+1), metrics_array[:, 2], yerr=stds[2], 
                       fmt='o-', capsize=5, color='green', markersize=8)
    axes[0,1].set_xlabel('Fold')
    axes[0,1].set_ylabel(f'{aux_label} R²')
    axes[0,1].set_title(f'{aux_label} R² Mean ± Std')
    axes[0,1].grid(True, alpha=0.3)
    
    # Weight RMSE
    axes[1,0].errorbar(range(1, len(all_metrics)+1), metrics_array[:, 1], yerr=stds[1], 
                       fmt='o-', capsize=5, color='red', markersize=8)
    axes[1,0].set_xlabel('Fold')
    axes[1,0].set_ylabel('Weight RMSE')
    axes[1,0].set_title('Weight RMSE Mean ± Std')
    axes[1,0].grid(True, alpha=0.3)
    
    # 辅助目标 RMSE
    axes[1,1].errorbar(range(1, len(all_metrics)+1), metrics_array[:, 3], yerr=stds[3], 
                       fmt='o-', capsize=5, color='orange', markersize=8)
    axes[1,1].set_xlabel('Fold')
    axes[1,1].set_ylabel(f'{aux_label} RMSE')
    axes[1,1].set_title(f'{aux_label} RMSE Mean ± Std')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(final_path, 'regression_metrics_errorbar.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Level分类指标errorbar
    plt.figure(figsize=(10,6))
    metrics_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    level_means = means[4:8]
    level_stds = stds[4:8]
    
    plt.errorbar(range(len(metrics_names)), level_means, yerr=level_stds, 
                fmt='o-', capsize=5, color='purple', markersize=10, linewidth=2)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Level Classification Metrics Mean ± Std')
    plt.xticks(range(len(metrics_names)), metrics_names)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(final_path, 'classification_metrics_errorbar.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 混淆矩阵（以第1折为例）
    if all_fold_data:
        cm = confusion_matrix(all_fold_data[0]['true_levels'], all_fold_data[0]['pred_levels'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Level Classification Confusion Matrix (Fold 1)')
        plt.savefig(os.path.join(final_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. 折间比较图
    plt.figure(figsize=(15, 8))
    
    # 创建子图
    plt.subplot(2, 4, 1)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 0], color='blue', alpha=0.7)
    plt.title('Weight R² by Fold')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 2)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 2], color='green', alpha=0.7)
    plt.title(f'{aux_label} R² by Fold')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 3)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 1], color='red', alpha=0.7)
    plt.title('Weight RMSE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 4)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 3], color='orange', alpha=0.7)
    plt.title(f'{aux_label} RMSE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 5)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 4], color='purple', alpha=0.7)
    plt.title('Precision by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 6)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 5], color='brown', alpha=0.7)
    plt.title('Recall by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 7)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 6], color='pink', alpha=0.7)
    plt.title('F1 Score by Fold')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 8)
    plt.bar(range(1, len(all_metrics)+1), metrics_array[:, 7], color='gray', alpha=0.7)
    plt.title('Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(final_path, 'fold_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n分析完成！共处理了 {len(all_metrics)} 折数据。")
    
    # 保存结果到文件
    results_file = os.path.join(final_path, "analysis_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("多折交叉验证结果分析\n")
        f.write("="*50 + "\n\n")
        f.write(f"数据来源: {base_path}\n")
        f.write(f"处理折数: {len(all_metrics)}\n\n")
        
        f.write("指标统计（均值±标准差）\n")
        f.write("-" * 30 + "\n")
        f.write(f"Weight R²: {means[0]:.4f} ± {stds[0]:.4f}\n")
        f.write(f"Weight RMSE: {means[1]:.4f} ± {stds[1]:.4f}\n")
        f.write(f"{aux_label} R²: {means[2]:.4f} ± {stds[2]:.4f}\n")
        f.write(f"{aux_label} RMSE: {means[3]:.4f} ± {stds[3]:.4f}\n")
        f.write(f"Level Precision: {means[4]:.4f} ± {stds[4]:.4f}\n")
        f.write(f"Level Recall: {means[5]:.4f} ± {stds[5]:.4f}\n")
        f.write(f"Level F1 Score: {means[6]:.4f} ± {stds[6]:.4f}\n")
        f.write(f"Level Accuracy: {means[7]:.4f} ± {stds[7]:.4f}\n\n")
        
        f.write("各折详细指标\n")
        f.write("-" * 30 + "\n")
        for i, m in enumerate(all_metrics):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Weight R²: {m['weight_r2']:.4f}\n")
            f.write(f"  Weight RMSE: {m['weight_rmse']:.4f}\n")
            f.write(f"  {aux_label} R²: {m['aux_r2']:.4f}\n")
            f.write(f"  {aux_label} RMSE: {m['aux_rmse']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall: {m['recall']:.4f}\n")
            f.write(f"  F1 Score: {m['f1']:.4f}\n")
            f.write(f"  Accuracy: {m['accuracy']:.4f}\n\n")
    
    print(f"详细结果已保存到: {results_file}")
    
    # 创建汇总表格
    print("\n" + "="*80)
    print("各折指标汇总表")
    print("="*80)
    header = f"{'Fold':<6} {'Weight R²':<12} {'Weight RMSE':<12} {aux_label+' R²':<12} {aux_label+' RMSE':<14} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Accuracy':<10}"
    print(header)
    print("-" * 80)
    for i, m in enumerate(all_metrics):
        print(f"{i+1:<6} {m['weight_r2']:<12.4f} {m['weight_rmse']:<12.4f} {m['aux_r2']:<12.4f} "
              f"{m['aux_rmse']:<14.4f} {m['precision']:<10.4f} {m['recall']:<8.4f} "
              f"{m['f1']:<8.4f} {m['accuracy']:<10.4f}")
    
    print("-" * 80)
    print(f"{'Mean':<6} {means[0]:<12.4f} {means[1]:<12.4f} {means[2]:<12.4f} {means[3]:<14.4f} "
          f"{means[4]:<10.4f} {means[5]:<8.4f} {means[6]:<8.4f} {means[7]:<10.4f}")
    print(f"{'Std':<6} {stds[0]:<12.4f} {stds[1]:<12.4f} {stds[2]:<12.4f} {stds[3]:<14.4f} "
          f"{stds[4]:<10.4f} {stds[5]:<8.4f} {stds[6]:<8.4f} {stds[7]:<10.4f}")
    
    # 保存汇总表格到CSV文件
    import pandas as pd
    summary_data = []
    for i, m in enumerate(all_metrics):
        summary_data.append({
            'Fold': i+1,
            'Weight_R2': m['weight_r2'],
            'Weight_RMSE': m['weight_rmse'],
            f'{aux_label}_R2': m['aux_r2'],
            f'{aux_label}_RMSE': m['aux_rmse'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1_Score': m['f1'],
            'Accuracy': m['accuracy']
        })
    
    # 添加均值行
    summary_data.append({
        'Fold': 'Mean',
        'Weight_R2': means[0],
        'Weight_RMSE': means[1],
        f'{aux_label}_R2': means[2],
        f'{aux_label}_RMSE': means[3],
        'Precision': means[4],
        'Recall': means[5],
        'F1_Score': means[6],
        'Accuracy': means[7]
    })
    
    # 添加标准差行
    summary_data.append({
        'Fold': 'Std',
        'Weight_R2': stds[0],
        'Weight_RMSE': stds[1],
        f'{aux_label}_R2': stds[2],
        f'{aux_label}_RMSE': stds[3],
        'Precision': stds[4],
        'Recall': stds[5],
        'F1_Score': stds[6],
        'Accuracy': stds[7]
    })
    
    df = pd.DataFrame(summary_data)
    csv_file = os.path.join(final_path, "metrics_summary.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"汇总表格已保存到: {csv_file}")
    
    # 创建README文件
    readme_content = f"""# 多折交叉验证结果分析报告

## 实验信息
- 数据来源: {base_path}
- 处理折数: {len(all_metrics)}
- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 主要结果
- Weight R²: {means[0]:.4f} ± {stds[0]:.4f}
- Weight RMSE: {means[1]:.4f} ± {stds[1]:.4f}
- {aux_label} R²: {means[2]:.4f} ± {stds[2]:.4f}
- {aux_label} RMSE: {means[3]:.4f} ± {stds[3]:.4f}
- Level Precision: {means[4]:.4f} ± {stds[4]:.4f}
- Level Recall: {means[5]:.4f} ± {stds[5]:.4f}
- Level F1 Score: {means[6]:.4f} ± {stds[6]:.4f}
- Level Accuracy: {means[7]:.4f} ± {stds[7]:.4f}

## 文件说明
- `analysis_results.txt`: 详细的分析结果
- `metrics_summary.csv`: 指标汇总表格
- `weight_r2_fitting.png`: Weight R²拟合图
- `{aux_label.lower()}_r2_fitting.png`: {aux_label} R²拟合图
- `regression_metrics_errorbar.png`: 回归指标错误条图
- `classification_metrics_errorbar.png`: 分类指标错误条图
- `confusion_matrix.png`: 混淆矩阵
- `fold_comparison.png`: 折间比较图

## 分析结论
基于{len(all_metrics)}折交叉验证的结果，模型在以下方面表现：
1. Weight预测: R² = {means[0]:.4f} ± {stds[0]:.4f}，表现{'优秀' if means[0] > 0.95 else '良好' if means[0] > 0.9 else '一般'}
2. {aux_label}预测: R² = {means[2]:.4f} ± {stds[2]:.4f}，表现{'优秀' if means[2] > 0.9 else '良好' if means[2] > 0.8 else '一般'}
3. Level分类: Accuracy = {means[7]:.4f} ± {stds[7]:.4f}，表现{'优秀' if means[7] > 0.9 else '良好' if means[7] > 0.8 else '一般'}
"""
    
    readme_file = os.path.join(final_path, "README.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README文件已保存到: {readme_file}")
    print(f"\n所有结果已保存到: {final_path}")
    print("包含以下文件:")
    for file in os.listdir(final_path):
        print(f"  - {file}")

if __name__ == "__main__":
    main() 