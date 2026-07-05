import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import os
import sys
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import glob
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入共享主干版本的模型
import importlib.util
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_shared_backbone_dir = os.path.join(project_root, 'model-SharedBackbone')
# 将 model-SharedBackbone 目录添加到 sys.path，以便 cucumber.py 能找到 modules.py
if model_shared_backbone_dir not in sys.path:
    sys.path.insert(0, model_shared_backbone_dir)
model_path = os.path.join(model_shared_backbone_dir, 'cucumber.py')
spec = importlib.util.spec_from_file_location("cucumber_shared", model_path)
cucumber_shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cucumber_shared)
CucumberModel = cucumber_shared.CucumberModel

from data import CucumberDataset

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
    class_names = ['super', 'first', 'second', 'third']
    
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

def find_latest_best_epoch(models_dir, fold_num=None):
    """
    查找每个fold下编号最大的best_epoch模型文件
    
    Args:
        models_dir: models目录路径
        fold_num: 如果指定，只查找该fold的模型，否则查找所有fold
    
    Returns:
        dict: {fold_number: model_path} 或 {fold_number: model_path} (如果指定fold_num)
    """
    model_files = {}
    
    if fold_num:
        # 只查找指定的fold
        fold_dir = os.path.join(models_dir, f'fold{fold_num}')
        if os.path.exists(fold_dir):
            best_epoch_files = glob.glob(os.path.join(fold_dir, 'best_epoch*.pth'))
            if best_epoch_files:
                # 提取epoch编号并找到最大的
                epochs = []
                for f in best_epoch_files:
                    match = re.search(r'best_epoch(\d+)\.pth', os.path.basename(f))
                    if match:
                        epochs.append((int(match.group(1)), f))
                if epochs:
                    max_epoch, latest_file = max(epochs, key=lambda x: x[0])
                    model_files[fold_num] = latest_file
    else:
        # 查找所有fold
        fold_dirs = glob.glob(os.path.join(models_dir, 'fold*'))
        for fold_dir in fold_dirs:
            match = re.search(r'fold(\d+)', os.path.basename(fold_dir))
            if match:
                fold_number = int(match.group(1))
                best_epoch_files = glob.glob(os.path.join(fold_dir, 'best_epoch*.pth'))
                if best_epoch_files:
                    # 提取epoch编号并找到最大的
                    epochs = []
                    for f in best_epoch_files:
                        match = re.search(r'best_epoch(\d+)\.pth', os.path.basename(f))
                        if match:
                            epochs.append((int(match.group(1)), f))
                    if epochs:
                        max_epoch, latest_file = max(epochs, key=lambda x: x[0])
                        model_files[fold_number] = latest_file
    
    return model_files

def test_model(test_dataset, model_path, pretrainpath, feature_fusion_type="cross_attention", 
               fpn_ablation_type="curv_only", combine_type="lka", device=None, logger=None, save_dir=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    model = CucumberModel(pretrainpath, feature_fusion_type=feature_fusion_type, fpn_ablation_type=fpn_ablation_type, combine_type=combine_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_weights = []
    pred_weights = []
    true_curvatures = []
    pred_curvatures = []
    true_levels = []
    pred_levels = []
    pred_probs = []

    with torch.no_grad():
        for step, data in enumerate(test_data_loader):
            inputs, weight, _, arch, level = data
            inputs, weight, arch, level = inputs.to(device), weight.to(device), arch.to(device), level.to(device)
            weight_pred, arch_pred, level_pred = model(inputs)

            true_weights.extend(weight.cpu().numpy())
            pred_weights.extend(weight_pred.squeeze().cpu().numpy())
            true_curvatures.extend(arch.cpu().numpy())
            pred_curvatures.extend(arch_pred.squeeze().cpu().numpy())
            true_levels.extend(level.cpu().numpy())
            pred_levels.extend(torch.argmax(level_pred, dim=1).cpu().numpy())
            pred_probs.extend(torch.nn.functional.softmax(level_pred, dim=1).cpu().numpy())

    print("true_weights", true_weights)
    print("pred_weights", pred_weights)
    print("true_curvatures", true_curvatures)
    print("pred_curvatures", pred_curvatures)
    print("true_levels", true_levels)
    print("pred_levels", pred_levels)

    if logger:
        logger.info(f"true_weights: {true_weights}")
        logger.info(f"pred_weights: {pred_weights}")
        logger.info(f"true_curvatures: {true_curvatures}")
        logger.info(f"pred_curvatures: {pred_curvatures}")
        logger.info(f"true_levels: {true_levels}")
        logger.info(f"pred_levels: {pred_levels}")

    # Calculate R2 and RMSE for weight and curvature regression
    weight_r2 = r2_score(true_weights, pred_weights)
    curvature_r2 = r2_score(true_curvatures, pred_curvatures)
    weight_rmse = np.sqrt(mean_squared_error(true_weights, pred_weights))
    curvature_rmse = np.sqrt(mean_squared_error(true_curvatures, pred_curvatures))

    # Calculate classification metrics
    level_precision = precision_score(true_levels, pred_levels, average='weighted', zero_division=0)
    level_recall = recall_score(true_levels, pred_levels, average='weighted', zero_division=0)
    level_f1 = f1_score(true_levels, pred_levels, average='weighted', zero_division=0)
    level_accuracy = accuracy_score(true_levels, pred_levels)

    print(f'Weight R2: {weight_r2}')
    print(f'Weight RMSE: {weight_rmse}')
    print(f'Curvature R2: {curvature_r2}')
    print(f'Curvature RMSE: {curvature_rmse}')
    print(f'Level Precision: {level_precision}')
    print(f'Level Recall: {level_recall}')
    print(f'Level F1 Score: {level_f1}')
    print(f'Level Accuracy: {level_accuracy}')

    if logger:
        logger.info(f'Weight R2: {weight_r2}')
        logger.info(f'Weight RMSE: {weight_rmse}')
        logger.info(f'Curvature R2: {curvature_r2}')
        logger.info(f'Curvature RMSE: {curvature_rmse}')
        logger.info(f'Level Precision: {level_precision}')
        logger.info(f'Level Recall: {level_recall}')
        logger.info(f'Level F1 Score: {level_f1}')
        logger.info(f'Level Accuracy: {level_accuracy}')

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
    
    # R²散点图 - 曲率预测
    curvature_r2_path = os.path.join(save_dir, f'curvature_r2_scatter_{model_name}.png')
    plot_r2_scatter(true_curvatures, pred_curvatures, 'Curvature Prediction', curvature_r2_path)
    print(f'Curvature R² scatter plot saved to: {curvature_r2_path}')
    if logger:
        logger.info(f'Curvature R² scatter plot saved to: {curvature_r2_path}')

if __name__ == "__main__":
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录是父目录
    config_path = os.path.join(current_dir, 'config_kfold.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='测试Cucumber模型（K-Fold交叉验证）')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='模型路径。如果不指定但指定了--result_dir，将自动使用每个fold的最后一个best_epoch')
    parser.add_argument('--result_dir', type=str, default=None,
                       help='训练结果文件夹路径（如 results/20251028-CucumberModel-KFold-c-1），不指定则在当前目录保存')
    parser.add_argument('--fold', type=int, default=None, 
                       help='指定fold编号（如1-5）。如果指定且未指定--model_path，将使用该fold的最后一个best_epoch')
    parser.add_argument('--test_all_folds', action='store_true',
                       help='如果指定了--result_dir且未指定--model_path，将测试所有fold的最后一个best_epoch模型')
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
    
    # 如果没有指定model_path，但指定了result_dir，尝试自动查找模型
    if args.model_path is None and args.result_dir:
        if not os.path.isabs(args.result_dir):
            args.result_dir = os.path.join(project_root, args.result_dir)
        
        models_dir = os.path.join(args.result_dir, 'models')
        if os.path.exists(models_dir):
            if args.test_all_folds or (args.fold is None):
                # 查找所有fold的最后一个best_epoch
                model_files = find_latest_best_epoch(models_dir)
                print(f"找到 {len(model_files)} 个fold的模型:")
                for fold_num in sorted(model_files.keys()):
                    print(f"  Fold {fold_num}: {os.path.basename(model_files[fold_num])}")
            elif args.fold:
                # 只查找指定fold的最后一个best_epoch
                model_files = find_latest_best_epoch(models_dir, fold_num=args.fold)
                if args.fold in model_files:
                    print(f"找到 Fold {args.fold} 的模型: {os.path.basename(model_files[args.fold])}")
                else:
                    print(f"错误: 未找到 Fold {args.fold} 的模型")
                    sys.exit(1)
            else:
                print("错误: 请指定 --model_path 或使用 --test_all_folds 或 --fold")
                sys.exit(1)
        else:
            print(f"错误: 未找到models目录: {models_dir}")
            sys.exit(1)
    elif args.model_path is None:
        print("错误: 必须指定 --model_path 或 --result_dir")
        sys.exit(1)
    else:
        # 使用指定的单个模型
        if args.test_all_folds or (args.fold is None and args.result_dir):
            print("错误: 指定了 --model_path 时不能使用 --test_all_folds")
            sys.exit(1)
        model_files = {0: args.model_path}  # 使用0作为key表示单个模型
    
    # 准备数据集（只需要一次）
    transform = transforms.Compose([
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor(),
    ])
    
    img_dir_test = os.path.join(args.img_dir, 'test')
    test_dataset = CucumberDataset(img_dir_test, mode="test", transform=transform)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试每个模型
    for fold_num in sorted(model_files.keys()):
        model_path = model_files[fold_num]
        
        # 创建test文件夹
        if args.result_dir:
            if fold_num == 0:
                # 单个模型，使用原始逻辑
                if args.fold:
                    test_dir = os.path.join(args.result_dir, 'figures', f'fold{args.fold}', 'test')
                else:
                    test_dir = os.path.join(args.result_dir, 'figures', 'test')
            else:
                # 每个fold的测试结果保存在各自的fold目录下
                test_dir = os.path.join(args.result_dir, 'figures', f'fold{fold_num}', 'test')
        else:
            test_dir = os.path.join(current_dir, 'test')
        
        os.makedirs(test_dir, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"测试 Fold {fold_num if fold_num > 0 else 'N/A'} 的模型: {os.path.basename(model_path)}")
        print(f"测试结果将保存到: {test_dir}")
        print(f"{'='*80}")
        
        # 设置日志（为每个fold创建单独的日志）
        if fold_num > 0:
            log_file = os.path.join(test_dir, f'test_fold{fold_num}.log')
        else:
            log_file = os.path.join(test_dir, 'test.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True  # 强制重新配置，避免多个fold日志混乱
        )
        logger = logging.getLogger(__name__)
        
        logger.info(f'测试模型: {model_path}')
        logger.info(f'测试结果保存到: {test_dir}')
        logger.info(f'feature_fusion_type: {args.feature_fusion_type}')
        logger.info(f'fpn_ablation_type: {args.fpn_ablation_type}')
        logger.info(f'combine_type: {args.combine_type}')
        
        # 测试模型
        test_model(
            test_dataset=test_dataset,
            model_path=model_path,
            pretrainpath=args.pretrained,
            feature_fusion_type=args.feature_fusion_type,
            fpn_ablation_type=args.fpn_ablation_type,
            combine_type=args.combine_type,
            device=device,
            logger=logger,
            save_dir=test_dir
        )
        
        print(f"\nFold {fold_num if fold_num > 0 else 'N/A'} 测试完成! 结果保存到: {test_dir}")
        logger.info(f"测试完成! 结果保存到: {test_dir}")
    
    print(f"\n{'='*80}")
    print(f"所有测试完成!")
    print(f"{'='*80}")

