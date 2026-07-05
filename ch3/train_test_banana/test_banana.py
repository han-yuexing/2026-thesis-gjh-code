import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import os
import sys
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.banana import BananaModel
from data import BananaDataset

# 加载yaml配置文件
def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 类别名称
    class_names = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(true_labels, pred_probs, save_path):
    """绘制ROC曲线"""
    true_labels_binary = np.eye(4)[true_labels]
    plt.figure(figsize=(10, 8))
    
    for i in range(4):
        fpr, tpr, _ = roc_curve(true_labels_binary[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Level {i} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class Classification')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_auc_bar(true_labels, pred_probs, save_path):
    """绘制AUC柱状图"""
    true_labels_binary = np.eye(4)[true_labels]
    auc_scores = []
    
    for i in range(4):
        fpr, tpr, _ = roc_curve(true_labels_binary[:, i], pred_probs[:, i])
        auc_scores.append(auc(fpr, tpr))
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Level 0', 'Level 1', 'Level 2', 'Level 3'], auc_scores, color='skyblue')
    plt.title('AUC Scores for Each Class')
    plt.ylabel('AUC Score')
    plt.ylim(0, 1)
    
    # 在柱子上添加数值标签
    for bar, score in zip(bars, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_r2_scatter(true_values, pred_values, title, save_path):
    """绘制R²散点图（真实值vs预测值，带斜线）"""
    plt.figure(figsize=(8, 6))
    
    # 绘制散点图
    plt.scatter(true_values, pred_values, alpha=0.6, color='blue')
    
    # 计算R²
    r2 = r2_score(true_values, pred_values)
    
    # 绘制理想斜线（y=x）
    min_val = min(min(true_values), min(pred_values))
    max_val = max(max(true_values), max(pred_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(test_dataset, model_path, pretrainpath, feature_fusion_type="cross_attention", 
               fpn_ablation_type="ripe_only", combine_type="lka", device=None, logger=None, save_dir=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    model = BananaModel(pretrainpath, feature_fusion_type=feature_fusion_type,
                        fpn_ablation_type=fpn_ablation_type, combine_type=combine_type).to(device)
    state_dict = torch.load(model_path, map_location=device)
    load_info = model.load_state_dict(state_dict, strict=False)
    missing_keys, unexpected_keys = load_info.missing_keys, load_info.unexpected_keys
    if (missing_keys or unexpected_keys) and logger:
        logger.warning(f"加载模型时存在差异 missing={missing_keys}, unexpected={unexpected_keys}")
    model.eval()
    
    true_weights = []
    pred_weights = []
    true_ripeness = []
    pred_ripeness = []
    true_levels = []
    pred_levels = []
    pred_probs = []
    
    with torch.no_grad():
        for step, data in enumerate(test_data_loader):
            inputs, weight, ripeness, level = data
            inputs, weight, ripeness, level = inputs.to(device), weight.to(device), ripeness.to(device), level.to(device)
            weight_pred, ripeness_pred, level_pred = model(inputs)
            
            true_weights.extend(weight.cpu().numpy())
            pred_weights.extend(weight_pred.squeeze().cpu().numpy())
            true_ripeness.extend(ripeness.cpu().numpy())
            pred_ripeness.extend(ripeness_pred.squeeze().cpu().numpy())
            true_levels.extend(level.cpu().numpy())
            pred_levels.extend(torch.argmax(level_pred, dim=1).cpu().numpy())
            pred_probs.extend(torch.nn.functional.softmax(level_pred, dim=1).cpu().numpy())
    
    print("true_weights", true_weights)
    print("pred_weights", pred_weights)
    print("true_ripeness", true_ripeness)
    print("pred_ripeness", pred_ripeness)
    print("true_levels", true_levels)
    print("pred_levels", pred_levels)
    
    if logger:
        logger.info(f"true_weights: {true_weights}")
        logger.info(f"pred_weights: {pred_weights}")
        logger.info(f"true_ripeness: {true_ripeness}")
        logger.info(f"pred_ripeness: {pred_ripeness}")
        logger.info(f"true_levels: {true_levels}")
        logger.info(f"pred_levels: {pred_levels}")
    
    # 计算回归指标
    weight_r2 = r2_score(true_weights, pred_weights)
    ripeness_r2 = r2_score(true_ripeness, pred_ripeness)
    weight_rmse = np.sqrt(mean_squared_error(true_weights, pred_weights))
    ripeness_rmse = np.sqrt(mean_squared_error(true_ripeness, pred_ripeness))
    
    # 计算分类指标
    level_precision = precision_score(true_levels, pred_levels, average='weighted', zero_division=0)
    level_recall = recall_score(true_levels, pred_levels, average='weighted', zero_division=0)
    level_f1 = f1_score(true_levels, pred_levels, average='weighted', zero_division=0)
    level_accuracy = accuracy_score(true_levels, pred_levels)
    true_levels_binary = np.eye(4)[true_levels]
    level_map = average_precision_score(true_levels_binary, np.array(pred_probs), average='weighted')
    
    print(f'Weight R2: {weight_r2}')
    print(f'Weight RMSE: {weight_rmse}')
    print(f'Ripeness R2: {ripeness_r2}')
    print(f'Ripeness RMSE: {ripeness_rmse}')
    print(f'Level Precision: {level_precision}')
    print(f'Level Recall: {level_recall}')
    print(f'Level F1 Score: {level_f1}')
    print(f'Level Accuracy: {level_accuracy}')
    print(f'Level mAP: {level_map}')
    
    if logger:
        logger.info(f'Weight R2: {weight_r2}')
        logger.info(f'Weight RMSE: {weight_rmse}')
        logger.info(f'Ripeness R2: {ripeness_r2}')
        logger.info(f'Ripeness RMSE: {ripeness_rmse}')
        logger.info(f'Level Precision: {level_precision}')
        logger.info(f'Level Recall: {level_recall}')
        logger.info(f'Level F1 Score: {level_f1}')
        logger.info(f'Level Accuracy: {level_accuracy}')
        logger.info(f'Level mAP: {level_map}')
    
    # 生成图像
    model_name = os.path.basename(model_path).replace('.pth', '')
    # 如果没有指定save_dir，使用当前目录
    if save_dir is None:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)
    # 混淆矩阵
    cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plot_confusion_matrix(true_levels, pred_levels, cm_path)
    print(f'Confusion matrix saved to: {cm_path}')
    if logger:
        logger.info(f'Confusion matrix saved to: {cm_path}')
    
    # ROC曲线
    roc_path = os.path.join(save_dir, f'roc_curves_{model_name}.png')
    plot_roc_curves(true_levels, np.array(pred_probs), roc_path)
    print(f'ROC curves saved to: {roc_path}')
    if logger:
        logger.info(f'ROC curves saved to: {roc_path}')
    
    # AUC柱状图
    auc_path = os.path.join(save_dir, f'auc_scores_{model_name}.png')
    plot_auc_bar(true_levels, np.array(pred_probs), auc_path)
    print(f'AUC scores saved to: {auc_path}')
    if logger:
        logger.info(f'AUC scores saved to: {auc_path}')
    
    # R²散点图 - 重量预测
    weight_r2_path = os.path.join(save_dir, f'weight_r2_scatter_{model_name}.png')
    plot_r2_scatter(true_weights, pred_weights, 'Weight Prediction', weight_r2_path)
    print(f'Weight R² scatter plot saved to: {weight_r2_path}')
    if logger:
        logger.info(f'Weight R² scatter plot saved to: {weight_r2_path}')
    
    # R²散点图 - 成熟度预测
    ripeness_r2_path = os.path.join(save_dir, f'ripeness_r2_scatter_{model_name}.png')
    plot_r2_scatter(true_ripeness, pred_ripeness, 'Ripeness Prediction', ripeness_r2_path)
    print(f'Ripeness R² scatter plot saved to: {ripeness_r2_path}')
    if logger:
        logger.info(f'Ripeness R² scatter plot saved to: {ripeness_r2_path}')

    metrics = {
        'weight_r2': weight_r2,
        'weight_rmse': weight_rmse,
        'ripeness_r2': ripeness_r2,
        'ripeness_rmse': ripeness_rmse,
        'level_precision': level_precision,
        'level_recall': level_recall,
        'level_f1': level_f1,
        'level_accuracy': level_accuracy,
        'level_map': level_map
    }

    return metrics

if __name__ == "__main__":
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录是父目录
    config_path = os.path.join(current_dir, 'config_kfold.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='测试Banana模型（K-Fold交叉验证）')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--result_dir', type=str, default=None,
                       help='训练结果文件夹路径（如 results/20251028-BananaModel-KFold-b-1），不指定则在当前目录保存')
    parser.add_argument('--fold', type=int, default=None, help='指定fold编号（如1-5），如果指定则在该fold下创建test子文件夹')
    parser.add_argument('--pretrained', type=str, default=config['model']['pretrained_path'],
                       help='预训练模型路径')
    parser.add_argument('--img_dir', type=str, default=config['dataset']['img_dir'],
                       help='数据集目录')
    parser.add_argument('--feature_fusion_type', type=str, default=config['model']['feature_fusion_type'],
                       help='特征融合类型')
    parser.add_argument('--fpn_ablation_type', type=str, default=config['model']['fpn_ablation_type'],
                       help='FPN消融类型')
    parser.add_argument('--combine_type', type=str, default=config['model']['combine_type'],
                       help='组合类型')
    
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载
    if args.config != config_path:
        config = load_config(args.config)
    
    # 将相对路径转换为绝对路径
    if not os.path.isabs(args.img_dir):
        args.img_dir = os.path.join(project_root, args.img_dir)
    if not os.path.isabs(args.pretrained):
        args.pretrained = os.path.join(project_root, args.pretrained)
    
    # 创建test文件夹
    if args.result_dir:
        # 在指定的结果文件夹下创建test子文件夹
        if args.fold:
            # 如果指定了fold，在对应fold的figures目录下创建test文件夹
            test_dir = os.path.join(args.result_dir, 'figures', f'fold{args.fold}', 'test')
        else:
            # 否则在根目录的figures下创建test文件夹
            test_dir = os.path.join(args.result_dir, 'figures', 'test')
    else:
        # 默认在当前目录创建test文件夹
        test_dir = os.path.join(current_dir, 'test')
    
    os.makedirs(test_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"测试结果将保存到: {test_dir}")
    
    # 设置日志（保存到test文件夹）
    log_file = os.path.join(test_dir, 'test.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f'测试模型: {args.model_path}')
    logger.info(f'测试结果保存到: {test_dir}')
    logger.info(f'feature_fusion_type: {args.feature_fusion_type}')
    logger.info(f'fpn_ablation_type: {args.fpn_ablation_type}')
    logger.info(f'combine_type: {args.combine_type}')
    print(f'测试模型: {args.model_path}')
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor(),
    ])
    
    img_dir_test = os.path.join(args.img_dir, 'test')
    test_dataset = BananaDataset(img_dir_test, mode="test", transform=transform)
    
    # 测试模型
    test_model(
        test_dataset=test_dataset,
        model_path=args.model_path,
        pretrainpath=args.pretrained,
        feature_fusion_type=args.feature_fusion_type,
        fpn_ablation_type=args.fpn_ablation_type,
        combine_type=args.combine_type,
        device=device,
        logger=logger,
        save_dir=test_dir
    )
    
    print(f"\n{'='*80}")
    print(f"测试完成!")
    print(f"测试结果已保存到: {test_dir}")
    print(f"{'='*80}")
    
    logger.info(f"测试完成! 结果保存到: {test_dir}")
