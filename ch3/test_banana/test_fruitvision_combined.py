import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
from datetime import datetime
import yaml
import cv2
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from data.dataset import BananaCombinedDataset
from FruitVision.fruit_vision import FruitVision

# 加载yaml配置文件
def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 创建带日期和序号的结果文件夹
def create_result_folder(project_root, model_name, dataset_type='b'):
    """
    创建结果文件夹，命名格式: 日期-模型-b-序号
    dataset_type: 'b' for banana
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取当前日期
    date_str = datetime.now().strftime('%Y%m%d')
    
    # 数据集标识
    dataset_abbr = dataset_type
    
    # 查找当前日期下最大的序号
    existing_folders = [f for f in os.listdir(results_dir) if f.startswith(f"{date_str}-{model_name}-{dataset_abbr}-")]
    if existing_folders:
        max_index = max([int(f.split('-')[-1]) for f in existing_folders if f.split('-')[-1].isdigit()])
        new_index = max_index + 1
    else:
        new_index = 1
    
    # 创建新文件夹
    folder_name = f"{date_str}-{model_name}-{dataset_abbr}-{new_index}"
    folder_path = os.path.join(results_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # 创建子文件夹
    logs_dir = os.path.join(folder_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    return folder_path, logs_dir

class CLAHEOtsuTransform:
    """CLAHE对比度增强和Otsu阈值法分割预处理"""
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def __call__(self, img):
        # 转换为numpy数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # 转换为BGR格式（OpenCV格式）
        if len(img.shape) == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
            
        # 调整尺寸到224x224
        img_resized = cv2.resize(img_bgr, self.target_size)
        
        # CLAHE对比度增强
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 转换为灰度图进行Otsu阈值分割
        gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 将mask应用到原图
        mask_3channel = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
        img_masked = cv2.bitwise_and(img_clahe, mask_3channel)
        
        # 转换回RGB格式
        img_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL Image并归一化到[0,1]
        img_pil = Image.fromarray(img_rgb)
        
        # 转换为tensor并归一化
        import torchvision.transforms as transforms
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img_pil)
        
        return img_tensor

def extract_labels_from_batch(batch):
    """从批次数据中提取标签（最后一个元素）"""
    if isinstance(batch, (list, tuple)):
        # 对于 BananaDataset: (image, weight, ripeness, level)
        return batch[-1]  # level 是最后一个元素
    else:
        return batch

def calculate_specificity(y_true, y_pred, class_names):
    """计算每个类别的specificity"""
    cm = confusion_matrix(y_true, y_pred)
    specificity_scores = {}
    
    for i, class_name in enumerate(class_names):
        # 计算真阴性（TN）和假阳性（FP）
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # 计算specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_scores[class_name] = specificity
    
    return specificity_scores

def test_model(model_path, test_dataset, batch_size=16, freeze_backbone=True, device=None, logger=None):
    """
    测试单个FruitVision模型
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = FruitVision(num_classes=4, pretrained=False, freeze_backbone=freeze_backbone).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            labels = extract_labels_from_batch(batch).to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    
    class_names = ['super', 'first', 'second', 'third']
    
    # 计算precision, recall, f1-score
    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(4), zero_division=0
    )
    
    # 计算specificity
    specificity_scores = calculate_specificity(all_labels, all_predictions, class_names)
    
    # 计算宏平均和加权平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    avg_specificity = np.mean(list(specificity_scores.values()))
    
    results = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_specificity': avg_specificity,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'specificity_scores': specificity_scores
    }
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='测试FruitVision模型')
    parser.add_argument('--result_folder', type=str, required=True, 
                       help='结果文件夹路径，例如: results/20251028-FruitVision-b-1')
    args = parser.parse_args()
    
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 结果文件夹路径
    result_folder_path = os.path.join(project_root, args.result_folder)
    if not os.path.exists(result_folder_path):
        print(f"错误: 结果文件夹不存在: {result_folder_path}")
        return
    
    # 加载配置文件
    config_path = os.path.join(result_folder_path, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    config = load_config(config_path)
    
    # 从配置中读取参数
    batch_size = config.get('batch_size', 16)
    device_str = config.get('device', 'cuda:0')
    freeze_backbone = config.get('freeze_backbone', True)
    seeds = config.get('seeds', [24, 42, 2024, 2025, 3407, 1234, 5678, 9999, 7777, 8888])
    n_runs = config.get('n_runs', 10)
    
    # 设置设备
    if device_str.startswith('cuda'):
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"使用设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"冻结骨干网络: {freeze_backbone}")
    print(f"运行次数: {n_runs}")
    print(f"随机种子: {seeds[:n_runs]}")
    
    # 模型目录
    models_dir = os.path.join(result_folder_path, 'models')
    
    # 数据目录
    banana_test_dir = os.path.join(project_root, 'data', 'banana')
    banana_add_dir = os.path.join(project_root, 'data', 'banana_add_70')
    
    # 创建新的结果文件夹
    model_name = f"FruitVision-Test-{'冻结' if freeze_backbone else '非冻结'}"
    new_result_folder, logs_dir = create_result_folder(project_root, model_name, dataset_type='b')
    print(f"结果将保存到: {new_result_folder}")
    
    # 设置日志
    log_file = os.path.join(logs_dir, 'test_combined.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 记录配置信息
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"设备: {device}")
    logger.info(f"冻结骨干网络: {freeze_backbone}")
    logger.info(f"运行次数: {n_runs}")
    logger.info(f"随机种子: {seeds[:n_runs]}")
    
    # 保存配置到结果文件夹
    config_save_path = os.path.join(new_result_folder, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'batch_size': batch_size,
            'device': str(device),
            'freeze_backbone': freeze_backbone,
            'n_runs': n_runs,
            'seeds': seeds[:n_runs],
            'test_dataset': 'banana (原始test + banana_add_70)',
            'model_source': args.result_folder
        }, f, allow_unicode=True)
    
    # 创建合并测试数据集
    transform = CLAHEOtsuTransform(target_size=(224, 224))
    print("创建合并测试数据集...")
    test_dataset = BananaCombinedDataset(
        banana_dir=banana_test_dir,
        banana_add_dir=banana_add_dir,
        transform=transform,
        include_test=True,
        include_add=True
    )
    print(f"测试数据集大小: {len(test_dataset)}")
    logger.info(f"测试数据集大小: {len(test_dataset)}")
    
    # 定义种子和对应的模型文件
    model_files = [
        'best_fruit_vision_banana_seed24_run1.pth',
        'best_fruit_vision_banana_seed42_run2.pth',
        'best_fruit_vision_banana_seed2024_run3.pth',
        'best_fruit_vision_banana_seed2025_run4.pth',
        'best_fruit_vision_banana_seed3407_run5.pth',
        'best_fruit_vision_banana_seed1234_run6.pth',
        'best_fruit_vision_banana_seed5678_run7.pth',
        'best_fruit_vision_banana_seed9999_run8.pth',
        'best_fruit_vision_banana_seed7777_run9.pth',
        'best_fruit_vision_banana_seed8888_run10.pth'
    ]
    
    # 确保种子和模型文件数量匹配
    if len(seeds) < n_runs:
        seeds = seeds[:n_runs]
    if len(model_files) < n_runs:
        model_files = model_files[:n_runs]
    
    # 存储所有测试结果
    all_results = []
    
    # 测试每个模型
    print(f"\n{'='*80}")
    print(f"开始测试 {len(model_files)} 个模型")
    print(f"{'='*80}")
    
    for i, (seed, model_file) in enumerate(zip(seeds, model_files)):
        model_path = os.path.join(models_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            logger.warning(f"模型文件不存在: {model_path}")
            continue
        
        print(f"\n--- 测试运行 {i+1} (种子: {seed}) ---")
        logger.info(f"测试运行 {i+1} (种子: {seed})")
        
        results = test_model(
            model_path=model_path,
            test_dataset=test_dataset,
            batch_size=batch_size,
            freeze_backbone=freeze_backbone,
            device=device,
            logger=logger
        )
        
        all_results.append(results)
        
        print(f"测试准确率: {results['accuracy']:.4f}")
        print(f"宏平均 - Precision: {results['macro_precision']:.4f}, Recall: {results['macro_recall']:.4f}, F1: {results['macro_f1']:.4f}, Specificity: {results['macro_specificity']:.4f}")
        print(f"加权平均 - Precision: {results['weighted_precision']:.4f}, Recall: {results['weighted_recall']:.4f}, F1: {results['weighted_f1']:.4f}")
    
    if len(all_results) == 0:
        print("错误: 没有成功测试任何模型")
        logger.error("没有成功测试任何模型")
        return
    
    # 计算统计指标
    test_accuracies = [r['accuracy'] for r in all_results]
    macro_precisions = [r['macro_precision'] for r in all_results]
    macro_recalls = [r['macro_recall'] for r in all_results]
    macro_f1s = [r['macro_f1'] for r in all_results]
    macro_specificities = [r['macro_specificity'] for r in all_results]
    weighted_precisions = [r['weighted_precision'] for r in all_results]
    weighted_recalls = [r['weighted_recall'] for r in all_results]
    weighted_f1s = [r['weighted_f1'] for r in all_results]
    
    # 计算平均值和标准差
    mean_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies, ddof=1)
    mean_macro_precision = np.mean(macro_precisions)
    std_macro_precision = np.std(macro_precisions, ddof=1)
    mean_macro_recall = np.mean(macro_recalls)
    std_macro_recall = np.std(macro_recalls, ddof=1)
    mean_macro_f1 = np.mean(macro_f1s)
    std_macro_f1 = np.std(macro_f1s, ddof=1)
    mean_macro_specificity = np.mean(macro_specificities)
    std_macro_specificity = np.std(macro_specificities, ddof=1)
    mean_weighted_precision = np.mean(weighted_precisions)
    std_weighted_precision = np.std(weighted_precisions, ddof=1)
    mean_weighted_recall = np.mean(weighted_recalls)
    std_weighted_recall = np.std(weighted_recalls, ddof=1)
    mean_weighted_f1 = np.mean(weighted_f1s)
    std_weighted_f1 = np.std(weighted_f1s, ddof=1)
    
    # 打印最终结果摘要
    print(f"\n{'='*80}")
    print(f"最终测试结果摘要")
    print(f"{'='*80}")
    print(f"测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"各次运行: {', '.join([f'{acc:.4f}' for acc in test_accuracies])}")
    print(f"\n宏平均指标:")
    print(f"  Precision: {mean_macro_precision:.4f} ± {std_macro_precision:.4f}")
    print(f"  Recall:    {mean_macro_recall:.4f} ± {std_macro_recall:.4f}")
    print(f"  F1-Score:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"  Specificity: {mean_macro_specificity:.4f} ± {std_macro_specificity:.4f}")
    print(f"\n加权平均指标:")
    print(f"  Precision: {mean_weighted_precision:.4f} ± {std_weighted_precision:.4f}")
    print(f"  Recall:    {mean_weighted_recall:.4f} ± {std_weighted_recall:.4f}")
    print(f"  F1-Score:  {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
    
    # 保存结果到文件
    freeze_str = '冻结' if freeze_backbone else '非冻结'
    results_summary_path = os.path.join(new_result_folder, f'fruit_vision_results_summary_banana_combined_{freeze_str}.txt')
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FruitVision模型 - BANANA数据集（合并测试集） - 多次运行结果摘要\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"实验配置:\n")
        f.write(f"  数据集: banana (原始test + banana_add_70)\n")
        f.write(f"  测试集大小: {len(test_dataset)}\n")
        f.write(f"  批次大小: {batch_size}\n")
        f.write(f"  设备: {device}\n")
        f.write(f"  冻结骨干网络: {freeze_backbone}\n")
        f.write(f"  运行次数: {len(all_results)}\n")
        f.write(f"  随机种子: {seeds[:len(all_results)]}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"测试集结果:\n")
        f.write(f"{'='*80}\n")
        f.write(f"测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}\n")
        f.write(f"各次运行: {', '.join([f'{acc:.4f}' for acc in test_accuracies])}\n\n")
        f.write(f"宏平均指标:\n")
        f.write(f"  Precision: {mean_macro_precision:.4f} ± {std_macro_precision:.4f}\n")
        f.write(f"  Recall:    {mean_macro_recall:.4f} ± {std_macro_recall:.4f}\n")
        f.write(f"  F1-Score:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}\n")
        f.write(f"  Specificity: {mean_macro_specificity:.4f} ± {std_macro_specificity:.4f}\n\n")
        f.write(f"加权平均指标:\n")
        f.write(f"  Precision: {mean_weighted_precision:.4f} ± {std_weighted_precision:.4f}\n")
        f.write(f"  Recall:    {mean_weighted_recall:.4f} ± {std_weighted_recall:.4f}\n")
        f.write(f"  F1-Score:  {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"详细结果 (各次运行):\n")
        f.write(f"{'='*80}\n")
        for i in range(len(all_results)):
            f.write(f"\n运行 {i+1} (种子: {seeds[i]}):\n")
            f.write(f"  测试准确率: {test_accuracies[i]:.4f}\n")
            f.write(f"  宏平均Precision: {macro_precisions[i]:.4f}\n")
            f.write(f"  宏平均Recall: {macro_recalls[i]:.4f}\n")
            f.write(f"  宏平均F1-Score: {macro_f1s[i]:.4f}\n")
            f.write(f"  宏平均Specificity: {macro_specificities[i]:.4f}\n")
            f.write(f"  加权平均Precision: {weighted_precisions[i]:.4f}\n")
            f.write(f"  加权平均Recall: {weighted_recalls[i]:.4f}\n")
            f.write(f"  加权平均F1-Score: {weighted_f1s[i]:.4f}\n")
    
    print(f"\n结果摘要已保存到: {results_summary_path}")
    logger.info(f"结果摘要已保存到: {results_summary_path}")
    
    completion_info = f"测试完成! 测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}, 宏平均F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}"
    print(f"\n{completion_info}")
    logger.info(completion_info)

if __name__ == "__main__":
    main()

