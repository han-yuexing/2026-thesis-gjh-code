import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from data.dataset import BananaDataset, CucumberDataset
from fruit_vision_multimodal import FruitVision_Multimodal, FruitVision_Multimodal_Simple

# 加载yaml配置文件
def load_config(config_path):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 创建带日期和序号的结果文件夹
def create_result_folder(project_root, model_name, dataset_type):
    """创建结果文件夹"""
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    dataset_abbr = 'c' if dataset_type == 'cucumber' else 'b'
    
    existing_folders = [f for f in os.listdir(results_dir) if f.startswith(f"{date_str}-{model_name}-{dataset_abbr}-")]
    if existing_folders:
        max_index = max([int(f.split('-')[-1]) for f in existing_folders if f.split('-')[-1].isdigit()])
        new_index = max_index + 1
    else:
        new_index = 1
    
    folder_name = f"{date_str}-{model_name}-{dataset_abbr}-{new_index}"
    folder_path = os.path.join(results_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    models_dir = os.path.join(folder_path, 'models')
    logs_dir = os.path.join(folder_path, 'logs')
    figures_dir = os.path.join(folder_path, 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    return folder_path, models_dir, logs_dir, figures_dir

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataset(dataset_type, split, transform):
    """获取数据集并返回形状特征数量"""
    if dataset_type == 'banana':
        DatasetClass = BananaDataset
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "banana", split)
        class_names = ['super', 'first', 'second', 'third']
        num_shape_features = 1  # ripeness
    elif dataset_type == 'cucumber':
        DatasetClass = CucumberDataset
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cucumber", split)
        class_names = ['super', 'first', 'second', 'third']
        num_shape_features = 1  # arch (弯曲度)
    else:
        raise ValueError('dataset_type must be "banana" or "cucumber"')
    dataset = DatasetClass(image_dir=data_dir, mode=split, transform=transform)
    return dataset, class_names, num_shape_features

def extract_features_from_batch(batch, dataset_type):
    """从batch中提取图像、重量、形状特征和标签"""
    if dataset_type == 'cucumber':
        # CucumberDataset返回: image, weight, curvature, arch, level
        # 只使用 arch (弯曲度)
        images = batch[0]
        weight = batch[1]
        arch = batch[3]
        labels = batch[4]
        shape_features = arch.unsqueeze(1)
    elif dataset_type == 'banana':
        # BananaDataset返回: image, weight, ripeness, level
        images = batch[0]
        weight = batch[1]
        ripeness = batch[2]
        labels = batch[3]
        shape_features = ripeness.unsqueeze(1)
    else:
        raise ValueError('dataset_type must be "banana" or "cucumber"')
    
    return images, weight, shape_features, labels

def train_model(dataset_type, transform, batch_size=16, epochs=50, lr=0.001, seed=42, 
                run_id=1, save_dirs=None, device=None, logger=None, use_simple_model=False,
                pretrained=True, freeze_backbone=True):
    """使用多模态输入训练FruitVision模型"""
    set_seed(seed)
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 获取训练数据集
    train_dataset, class_names, num_shape_features = get_dataset(dataset_type, 'train', transform)
    
    # 分为训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if logger:
        logger.info(f"==================== Starting run {run_id} with seed {seed} ====================")
        logger.info(f"Dataset: {dataset_type}, Shape features: {num_shape_features}")
        logger.info(f"Model type: {'Simple' if use_simple_model else 'Full'}")
    print(f"==================== Starting run {run_id} with seed {seed} ====================")
    print(f"数据集: {dataset_type}, 形状特征数: {num_shape_features}")
    print(f"模型类型: {'简化版' if use_simple_model else '完整版'}")
    
    # 初始化多模态模型
    if use_simple_model:
        model = FruitVision_Multimodal_Simple(
            num_classes=len(class_names), 
            num_shape_features=num_shape_features,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        ).to(device)
    else:
        model = FruitVision_Multimodal(
            num_classes=len(class_names), 
            num_shape_features=num_shape_features,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    
    # 训练历史记录
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images, weight, shape_features, labels = extract_features_from_batch(batch, dataset_type)
            
            images = images.to(device)
            weight = weight.to(device)
            shape_features = shape_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, weight, shape_features)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, weight, shape_features, labels = extract_features_from_batch(batch, dataset_type)
                
                images = images.to(device)
                weight = weight.to(device)
                shape_features = shape_features.to(device)
                labels = labels.to(device)
                
                outputs = model(images, weight, shape_features)
                loss = loss_function(outputs, labels)
                
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # 记录历史
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 打印信息
        epoch_info = f"Run {run_id}, Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        print(epoch_info)
        if logger:
            logger.info(epoch_info)
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
    
    # 保存最佳模型
    model_suffix = "simple" if use_simple_model else "full"
    if save_dirs and 'models' in save_dirs:
        best_model_path = os.path.join(save_dirs['models'], 
                                      f"best_fruitvision_multimodal_{model_suffix}_{dataset_type}_seed{seed}_run{run_id}.pth")
    else:
        best_model_path = f"best_fruitvision_multimodal_{model_suffix}_{dataset_type}_seed{seed}_run{run_id}.pth"
    torch.save(best_model_state, best_model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curve - Run {run_id} (Seed {seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy Curve - Run {run_id} (Seed {seed})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_dirs and 'figures' in save_dirs:
        curves_path = os.path.join(save_dirs['figures'], 
                                  f'fruitvision_multimodal_{model_suffix}_run{run_id}_seed{seed}_curves_{dataset_type}.png')
    else:
        curves_path = f'fruitvision_multimodal_{model_suffix}_run{run_id}_seed{seed}_curves_{dataset_type}.png'
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Run {run_id} completed! Best validation accuracy: {best_val_acc:.4f}")
    
    return best_model_path, class_names, num_shape_features, best_val_acc

def test_model(dataset_type, model_path, transform, batch_size=16, class_names=None, 
               num_shape_features=None, save_dirs=None, device=None, logger=None, 
               use_simple_model=False, pretrained=True, freeze_backbone=True):
    """测试多模态模型"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 获取测试数据集
    test_dataset, class_names, num_shape_features = get_dataset(dataset_type, 'test', transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    if use_simple_model:
        model = FruitVision_Multimodal_Simple(
            num_classes=len(class_names), 
            num_shape_features=num_shape_features,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        ).to(device)
    else:
        model = FruitVision_Multimodal(
            num_classes=len(class_names), 
            num_shape_features=num_shape_features,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    loss_function = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    final_predictions = []
    final_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, weight, shape_features, labels = extract_features_from_batch(batch, dataset_type)
            
            images = images.to(device)
            weight = weight.to(device)
            shape_features = shape_features.to(device)
            labels = labels.to(device)
            
            outputs = model(images, weight, shape_features)
            loss = loss_function(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            final_predictions.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    final_accuracy = accuracy_score(final_labels, final_predictions)
    
    # 计算precision, recall, f1-score
    precision, recall, f1_score, support = precision_recall_fscore_support(
        final_labels, final_predictions, average=None, labels=range(len(class_names))
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        final_labels, final_predictions, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        final_labels, final_predictions, average='weighted'
    )
    
    test_results = [
        f"Test Loss: {avg_test_loss:.4f}",
        f"Test Accuracy: {test_accuracy:.4f}",
        f"Final model accuracy on {dataset_type} test set: {final_accuracy:.4f}",
        f"Macro Average - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1-Score: {macro_f1:.4f}",
        f"Weighted Average - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1-Score: {weighted_f1:.4f}"
    ]
    
    # 打印每个类别的详细指标
    print("\n--- Per-Class Metrics ---")
    if logger:
        logger.info("--- Per-Class Metrics ---")
    for i, class_name in enumerate(class_names):
        class_metrics = f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1_score[i]:.4f}, Support={support[i]}"
        print(class_metrics)
        if logger:
            logger.info(class_metrics)
    
    # 打印平均指标
    for result in test_results:
        print(result)
        if logger:
            logger.info(result)
    
    print("\n--- Classification Report ---")
    if logger:
        logger.info("--- Classification Report ---")
    classification_rep = classification_report(final_labels, final_predictions, target_names=class_names)
    print(classification_rep)
    if logger:
        logger.info(f"\n{classification_rep}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(final_labels, final_predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Multimodal FruitVision')
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
    if save_dirs and 'figures' in save_dirs:
        confusion_matrix_path = os.path.join(save_dirs['figures'], 
                                            f'confusion_matrix_multimodal_{dataset_type}_{os.path.basename(model_path).replace(".pth","")}.png')
    else:
        confusion_matrix_path = f'confusion_matrix_multimodal_{dataset_type}_{os.path.basename(model_path).replace(".pth","")}.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    if logger:
        logger.info(f"Confusion matrix saved: {confusion_matrix_path}")
    
    # 返回详细的测试结果
    test_results_dict = {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    return test_results_dict


if __name__ == "__main__":
    import argparse
    
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(current_dir, 'config_multimodal.yaml')
    
    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(config_path):
        config_path = os.path.join(current_dir, 'config.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='FruitVision多模态模型训练和测试')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--dataset', type=str, default=config['dataset']['type'], 
                        choices=['banana', 'cucumber'], help='数据集类型')
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'], help='批次大小')
    parser.add_argument('--lr', type=float, default=config['training']['lr'], help='学习率')
    parser.add_argument('--n_runs', type=int, default=config['training']['n_runs'], 
                        help='运行次数（用于显著性检验）')
    parser.add_argument('--seeds', type=int, nargs='+', default=config['seeds'], 
                        help='随机种子列表')
    parser.add_argument('--simple', action='store_true', 
                        default=config.get('model', {}).get('type', 'full') == 'simple',
                        help='使用简化版模型 (直接拼接特征，不使用编码器)')
    parser.add_argument('--no-pretrained', action='store_true',
                        default=not config.get('model', {}).get('pretrained', True),
                        help='不使用预训练权重')
    parser.add_argument('--no-freeze', action='store_true',
                        default=not config.get('model', {}).get('freeze_backbone', True),
                        help='不冻结骨干网络')
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载并更新参数
    if args.config != config_path:
        config = load_config(args.config)
        # 如果命令行没有明确指定，则从新配置文件读取
        if '--simple' not in sys.argv and '--full' not in sys.argv:
            args.simple = config.get('model', {}).get('type', 'full') == 'simple'
        if '--no-pretrained' not in sys.argv:
            args.no_pretrained = not config.get('model', {}).get('pretrained', True)
        if '--no-freeze' not in sys.argv:
            args.no_freeze = not config.get('model', {}).get('freeze_backbone', True)
    
    # 确保种子数量与运行次数匹配
    if len(args.seeds) < args.n_runs:
        print(f"警告: 提供的种子数量({len(args.seeds)})少于运行次数({args.n_runs})，将使用默认种子")
        args.seeds = config['seeds'][:args.n_runs]
    
    # 创建结果文件夹
    model_name = "FruitVision-Multi" + ("-Simple" if args.simple else "")
    result_folder, models_dir, logs_dir, figures_dir = create_result_folder(
        project_root, model_name, args.dataset
    )
    
    # 创建保存目录字典
    save_dirs = {
        'root': result_folder,
        'models': models_dir,
        'logs': logs_dir,
        'figures': figures_dir
    }
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"结果将保存到: {result_folder}")
    print(f"模型类型: {'简化版 (直接拼接)' if args.simple else '完整版 (带编码器)'}")
    print(f"预训练权重: {'否' if args.no_pretrained else '是'}")
    print(f"冻结骨干网络: {'否' if args.no_freeze else '是'}")
    
    # 设置日志
    log_file = os.path.join(logs_dir, 'fruitvision_multimodal_training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 保存配置到结果文件夹
    config_save_path = os.path.join(result_folder, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'dataset': args.dataset,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'n_runs': args.n_runs,
            'seeds': args.seeds[:args.n_runs],
            'device': str(device),
            'model_type': 'simple' if args.simple else 'full',
            'multimodal': True,
            'pretrained': not args.no_pretrained,
            'freeze_backbone': not args.no_freeze
        }, f, allow_unicode=True)
    
    # 存储所有运行的结果
    all_models = []
    all_val_accs = []
    class_names = None
    num_shape_features = None
    
    logger.info(f"开始使用{args.n_runs}个不同种子进行训练，数据集: {args.dataset}")
    logger.info(f"结果保存路径: {result_folder}")
    logger.info(f"模型类型: {'简化版' if args.simple else '完整版'}")
    print(f"\n开始使用{args.n_runs}个不同种子进行多模态训练，数据集: {args.dataset}")
    print(f"使用的种子: {args.seeds[:args.n_runs]}")
    
    # 执行多次训练，每次使用不同的种子
    for i in range(args.n_runs):
        seed = args.seeds[i]
        print(f"\n{'='*80}")
        print(f"第 {i+1}/{args.n_runs} 次运行 (种子: {seed})")
        print(f"{'='*80}")
        
        model_path, class_names, num_shape_features, val_acc = train_model(
            dataset_type=args.dataset,
            transform=transform,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=seed,
            run_id=i+1,
            save_dirs=save_dirs,
            device=device,
            logger=logger,
            use_simple_model=args.simple,
            pretrained=not args.no_pretrained,
            freeze_backbone=not args.no_freeze
        )
        
        all_models.append(model_path)
        all_val_accs.append(val_acc)
    
    # 计算验证集的平均准确率和标准差
    mean_val_acc = np.mean(all_val_accs)
    std_val_acc = np.std(all_val_accs, ddof=1)
    
    print(f"\n{'='*80}")
    print(f"所有训练运行完成!")
    print(f"验证集准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"各次运行的验证准确率: {[f'{acc:.4f}' for acc in all_val_accs]}")
    print(f"{'='*80}")
    
    logger.info(f"所有训练运行完成! 平均验证准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    
    # 在测试集上测试所有模型
    print(f"\n{'='*80}")
    print(f"开始在测试集上测试所有模型")
    print(f"{'='*80}")
    test_results_list = []
    
    for i, model_path in enumerate(all_models):
        print(f"\n--- 测试第 {i+1} 次运行的最佳模型 (种子: {args.seeds[i]}) ---")
        test_results = test_model(
            args.dataset, model_path, transform, args.batch_size, class_names,
            num_shape_features=num_shape_features,
            save_dirs=save_dirs, device=device, logger=logger, use_simple_model=args.simple,
            pretrained=not args.no_pretrained, freeze_backbone=not args.no_freeze
        )
        test_results_list.append(test_results)
    
    # 收集所有测试指标
    test_accuracies = [result['accuracy'] for result in test_results_list]
    macro_precisions = [result['macro_precision'] for result in test_results_list]
    macro_recalls = [result['macro_recall'] for result in test_results_list]
    macro_f1s = [result['macro_f1'] for result in test_results_list]
    weighted_precisions = [result['weighted_precision'] for result in test_results_list]
    weighted_recalls = [result['weighted_recall'] for result in test_results_list]
    weighted_f1s = [result['weighted_f1'] for result in test_results_list]
    
    # 计算平均值和标准差
    mean_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies, ddof=1)
    mean_macro_precision = np.mean(macro_precisions)
    std_macro_precision = np.std(macro_precisions, ddof=1)
    mean_macro_recall = np.mean(macro_recalls)
    std_macro_recall = np.std(macro_recalls, ddof=1)
    mean_macro_f1 = np.mean(macro_f1s)
    std_macro_f1 = np.std(macro_f1s, ddof=1)
    mean_weighted_precision = np.mean(weighted_precisions)
    std_weighted_precision = np.std(weighted_precisions, ddof=1)
    mean_weighted_recall = np.mean(weighted_recalls)
    std_weighted_recall = np.std(weighted_recalls, ddof=1)
    mean_weighted_f1 = np.mean(weighted_f1s)
    std_weighted_f1 = np.std(weighted_f1s, ddof=1)
    
    # 打印最终结果摘要
    print(f"\n{'='*80}")
    print(f"多模态FruitVision模型 - 最终测试结果摘要")
    print(f"{'='*80}")
    print(f"测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"各次运行的测试准确率: {[f'{acc:.4f}' for acc in test_accuracies]}")
    print(f"\n宏平均指标:")
    print(f"  Precision: {mean_macro_precision:.4f} ± {std_macro_precision:.4f}")
    print(f"  Recall:    {mean_macro_recall:.4f} ± {std_macro_recall:.4f}")
    print(f"  F1-Score:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"\n加权平均指标:")
    print(f"  Precision: {mean_weighted_precision:.4f} ± {std_weighted_precision:.4f}")
    print(f"  Recall:    {mean_weighted_recall:.4f} ± {std_weighted_recall:.4f}")
    print(f"  F1-Score:  {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
    
    logger.info(f"{'='*80}")
    logger.info(f"多模态FruitVision模型 - 最终测试结果摘要")
    logger.info(f"{'='*80}")
    logger.info(f"测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    logger.info(f"宏平均 - Precision: {mean_macro_precision:.4f} ± {std_macro_precision:.4f}")
    logger.info(f"宏平均 - Recall: {mean_macro_recall:.4f} ± {std_macro_recall:.4f}")
    logger.info(f"宏平均 - F1-Score: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    logger.info(f"加权平均 - Precision: {mean_weighted_precision:.4f} ± {std_weighted_precision:.4f}")
    logger.info(f"加权平均 - Recall: {mean_weighted_recall:.4f} ± {std_weighted_recall:.4f}")
    logger.info(f"加权平均 - F1-Score: {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
    
    # 绘制所有运行的测试准确率对比图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    run_numbers = list(range(1, args.n_runs + 1))
    plt.bar(run_numbers, test_accuracies, color='skyblue', edgecolor='navy')
    plt.axhline(y=mean_test_acc, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_test_acc:.4f}')
    plt.title(f'{args.dataset.capitalize()} - 测试准确率 ({args.n_runs}次运行)\n多模态FruitVision', fontsize=12)
    plt.xlabel('运行次数')
    plt.ylabel('测试准确率')
    plt.xticks(run_numbers)
    plt.ylim([min(test_accuracies) - 0.02, max(test_accuracies) + 0.02])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(run_numbers, macro_f1s, color='lightcoral', edgecolor='darkred')
    plt.axhline(y=mean_macro_f1, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_macro_f1:.4f}')
    plt.title(f'{args.dataset.capitalize()} - 宏平均F1分数 ({args.n_runs}次运行)\n多模态FruitVision', fontsize=12)
    plt.xlabel('运行次数')
    plt.ylabel('宏平均F1分数')
    plt.xticks(run_numbers)
    plt.ylim([min(macro_f1s) - 0.02, max(macro_f1s) + 0.02])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    results_path = os.path.join(figures_dir, f'fruitvision_multimodal_multi_run_results_{args.dataset}.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"多次运行结果对比图已保存: {results_path}")
    
    # 保存结果摘要
    results_summary_path = os.path.join(result_folder, f'fruitvision_multimodal_results_summary_{args.dataset}.txt')
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"多模态FruitVision模型 - {args.dataset.upper()}数据集 - 多次运行结果摘要\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"实验配置:\n")
        f.write(f"  数据集: {args.dataset}\n")
        f.write(f"  模型类型: {'简化版 (直接拼接)' if args.simple else '完整版 (带编码器)'}\n")
        f.write(f"  预训练: {'否' if args.no_pretrained else '是'}\n")
        f.write(f"  冻结骨干: {'否' if args.no_freeze else '是'}\n")
        f.write(f"  形状特征数: {num_shape_features}\n")
        f.write(f"  运行次数: {args.n_runs}\n")
        f.write(f"  随机种子: {args.seeds[:args.n_runs]}\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  批次大小: {args.batch_size}\n")
        f.write(f"  学习率: {args.lr}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"验证集结果:\n")
        f.write(f"{'='*80}\n")
        f.write(f"平均准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}\n")
        f.write(f"各次运行: {', '.join([f'{acc:.4f}' for acc in all_val_accs])}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"测试集结果:\n")
        f.write(f"{'='*80}\n")
        f.write(f"测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}\n")
        f.write(f"各次运行: {', '.join([f'{acc:.4f}' for acc in test_accuracies])}\n\n")
        f.write(f"宏平均指标:\n")
        f.write(f"  Precision: {mean_macro_precision:.4f} ± {std_macro_precision:.4f}\n")
        f.write(f"  Recall:    {mean_macro_recall:.4f} ± {std_macro_recall:.4f}\n")
        f.write(f"  F1-Score:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}\n\n")
        f.write(f"加权平均指标:\n")
        f.write(f"  Precision: {mean_weighted_precision:.4f} ± {std_weighted_precision:.4f}\n")
        f.write(f"  Recall:    {mean_weighted_recall:.4f} ± {std_weighted_recall:.4f}\n")
        f.write(f"  F1-Score:  {mean_weighted_f1:.4f} ± {std_weighted_f1:.4f}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"详细结果 (各次运行):\n")
        f.write(f"{'='*80}\n")
        for i in range(args.n_runs):
            f.write(f"\n运行 {i+1} (种子: {args.seeds[i]}):\n")
            f.write(f"  验证准确率: {all_val_accs[i]:.4f}\n")
            f.write(f"  测试准确率: {test_accuracies[i]:.4f}\n")
            f.write(f"  宏平均Precision: {macro_precisions[i]:.4f}\n")
            f.write(f"  宏平均Recall: {macro_recalls[i]:.4f}\n")
            f.write(f"  宏平均F1-Score: {macro_f1s[i]:.4f}\n")
            f.write(f"  加权平均Precision: {weighted_precisions[i]:.4f}\n")
            f.write(f"  加权平均Recall: {weighted_recalls[i]:.4f}\n")
            f.write(f"  加权平均F1-Score: {weighted_f1s[i]:.4f}\n")
    
    print(f"\n结果摘要已保存到: {results_summary_path}")
    logger.info(f"结果摘要已保存到: {results_summary_path}")
    
    completion_info = f"多模态训练和测试完成! 测试准确率: {mean_test_acc:.4f} ± {std_test_acc:.4f}, 宏平均F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}"
    print(f"\n{completion_info}")
    logger.info(completion_info)

