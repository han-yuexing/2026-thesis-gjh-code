import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import os
import sys
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
import numpy as np
import yaml
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.separate import SeparateModule
from data.dataset import MixedDataset

# 加载yaml配置文件
def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_model(test_dataset, model_path, pretrainpath, device=None, logger=None, save_dir=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    model = SeparateModule(pretrainpath).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_categories = []
    pred_categories = []
    pred_probs = []

    with torch.no_grad():
        for step, data in enumerate(test_data_loader):
            inputs, _, _, category, _ = data
            inputs, category = inputs.to(device), category.to(device)
            outputs = model(inputs)

            true_categories.extend(category.cpu().numpy())
            pred_categories.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            pred_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    print("true_categories", true_categories)
    print("pred_categories", pred_categories)
    if logger:
        logger.info(f"true_categories: {true_categories}")
        logger.info(f"pred_categories: {pred_categories}")

    # Calculate F1 and mAP for category classification
    category_f1 = f1_score(true_categories, pred_categories, average='weighted')
    true_categories_binary = np.eye(2)[true_categories]  # Assuming 2 categories for binary classification
    category_map = average_precision_score(true_categories_binary, np.array(pred_probs), average='weighted')

    # Calculate accuracy for each category
    category_accuracy = accuracy_score(true_categories, pred_categories)

    print(f"\n{'='*80}")
    print(f'测试结果:')
    print(f"{'='*80}")
    print(f'Category Accuracy: {category_accuracy:.4f}')
    print(f'Category F1 Score: {category_f1:.4f}')
    print(f'Category mAP: {category_map:.4f}')
    print(f"{'='*80}")

    if logger:
        logger.info(f"{'='*80}")
        logger.info(f'测试结果:')
        logger.info(f"{'='*80}")
        logger.info(f'Category Accuracy: {category_accuracy:.4f}')
        logger.info(f'Category F1 Score: {category_f1:.4f}')
        logger.info(f'Category mAP: {category_map:.4f}')
        logger.info(f"{'='*80}")
    
    # 保存测试结果到文本文件
    if save_dir:
        model_name = os.path.basename(model_path).replace('.pth', '')
        results_file = os.path.join(save_dir, f'test_results_{model_name}.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Separate模型测试结果\n")
            f.write("="*80 + "\n\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"测试样本数: {len(true_categories)}\n\n")
            f.write("="*80 + "\n")
            f.write("测试指标:\n")
            f.write("="*80 + "\n")
            f.write(f"Accuracy:  {category_accuracy:.4f}\n")
            f.write(f"F1 Score:  {category_f1:.4f}\n")
            f.write(f"mAP:       {category_map:.4f}\n\n")
            f.write("="*80 + "\n")
            f.write("预测详情:\n")
            f.write("="*80 + "\n")
            f.write(f"True categories: {true_categories}\n")
            f.write(f"Pred categories: {pred_categories}\n")
        
        print(f"\n测试结果已保存到: {results_file}")
        if logger:
            logger.info(f"测试结果已保存到: {results_file}")
    
    return {
        'accuracy': category_accuracy,
        'f1_score': category_f1,
        'mAP': category_map
    }

if __name__ == "__main__":
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录是父目录
    config_path = os.path.join(current_dir, 'config.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='测试Separate模型')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--result_dir', type=str, default=None, 
                       help='训练结果文件夹路径（如 results/20251027-separate-m-1），不指定则在当前目录创建test文件夹')
    parser.add_argument('--pretrained', type=str, default=config['model']['pretrained_path'],
                       help='预训练模型路径')
    parser.add_argument('--img_dir', type=str, default=config['dataset']['img_dir'],
                       help='数据集目录')
    
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载
    if args.config != config_path:
        config = load_config(args.config)
    
    # 将相对路径转换为绝对路径（相对于脚本所在目录）
    if not os.path.isabs(args.img_dir):
        args.img_dir = os.path.join(project_root, args.img_dir)
    if not os.path.isabs(args.pretrained):
        args.pretrained = os.path.join(project_root, args.pretrained)
    
    # 创建test文件夹
    if args.result_dir:
        # 在指定的结果文件夹下创建test子文件夹
        test_dir = os.path.join(args.result_dir, 'test')
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
    print(f'测试模型: {args.model_path}')
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.RandomRotation(degrees=config['transform']['rotation_degrees']),
        transforms.ToTensor(),
    ])
    
    img_dir_test = os.path.join(args.img_dir, 'test')
    test_dataset = MixedDataset(img_dir_test, mode="test", transform=transform)
    
    # 测试模型
    test_results = test_model(
        test_dataset, 
        args.model_path, 
        args.pretrained, 
        device=device, 
        logger=logger,
        save_dir=test_dir
    )
    
    print(f"\n{'='*80}")
    print(f"测试完成!")
    print(f"准确率: {test_results['accuracy']:.4f}")
    print(f"F1分数: {test_results['f1_score']:.4f}")
    print(f"mAP: {test_results['mAP']:.4f}")
    print(f"测试结果已保存到: {test_dir}")
    print(f"{'='*80}")
    
    logger.info(f"测试完成! 准确率: {test_results['accuracy']:.4f}, F1: {test_results['f1_score']:.4f}, mAP: {test_results['mAP']:.4f}")