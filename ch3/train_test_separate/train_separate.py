import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
import os
import sys
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
import math
from datetime import datetime
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

# 创建带日期和序号的结果文件夹
def create_result_folder(project_root, model_name, dataset_type='m'):
    """
    创建结果文件夹，命名格式: 日期-模型-m-序号
    dataset_type: 'm' for mixed dataset
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取当前日期
    date_str = datetime.now().strftime('%Y%m%d')
    
    # 数据集标识 (mixed -> 'm')
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
    models_dir = os.path.join(folder_path, 'models')
    logs_dir = os.path.join(folder_path, 'logs')
    figures_dir = os.path.join(folder_path, 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    return folder_path, models_dir, logs_dir, figures_dir

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        cosine_steps = current_step - num_warmup_steps
        cosine_total_steps = max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * cosine_steps / cosine_total_steps))
    
    return LambdaLR(optimizer, lr_lambda)

def set_seed(seed):
    random.seed(seed)  # Python 随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # PyTorch 随机数生成器（CPU）
    
    # 如果使用 GPU，需设置以下两项
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子（如果有多个）
    
    # 使用确定性的 cuDNN 算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_validate(dataset, num_epochs, batch_size, lr, modelpath, patience=10, seed=42, run_id=1, save_dirs=None, device=None, logger=None):
    # 设置随机种子
    set_seed(seed)
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if logger:
        logger.info(f"==================== Starting run {run_id} with seed {seed} ====================")
    print(f"==================== Starting run {run_id} with seed {seed} ====================")

    model = SeparateModule(modelpath).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_data_loader))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []

    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for step, data in enumerate(train_data_loader):
            inputs, _, _, category, _ = data
            inputs, category = inputs.to(device), category.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, category)
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == category).sum().item()
            total_train += category.size(0)

        avg_train_loss = total_train_loss / len(train_data_loader)
        train_acc = correct_train / total_train
        train_losses.append((avg_train_loss, train_acc))

        # 验证
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for step, data in enumerate(val_data_loader):
                inputs, _, _, category, _ = data
                inputs, category = inputs.to(device), category.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, category)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == category).sum().item()
                total_val += category.size(0)

        avg_val_loss = total_val_loss / len(val_data_loader)
        val_acc = correct_val / total_val
        val_losses.append((avg_val_loss, val_acc))

        # 日志和打印
        epoch_info = f'Run {run_id}, Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}'
        print(epoch_info)
        if logger:
            logger.info(epoch_info)

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stop_info = f"Early stopping triggered after {epoch + 1} epochs"
            print(stop_info)
            if logger:
                logger.info(stop_info)
            break

    # 保存最佳模型
    if save_dirs and 'models' in save_dirs:
        best_model_path = os.path.join(save_dirs['models'], f'best_separate_seed{seed}_run{run_id}.pth')
    else:
        best_model_path = f'best_separate_seed{seed}_run{run_id}.pth'
    torch.save(best_model_state, best_model_path)
    
    if logger:
        logger.info(f"Run {run_id} completed! Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.4f}")
    
    # 绘图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), [x[0] for x in train_losses], label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), [x[0] for x in val_losses], label='Validation Loss')
    plt.title(f'Loss Curve - Run {run_id} (Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), [x[1] for x in train_losses], label='Train Accuracy')
    plt.plot(range(1, len(val_losses) + 1), [x[1] for x in val_losses], label='Validation Accuracy')
    plt.title(f'Accuracy Curve - Run {run_id} (Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_dirs and 'figures' in save_dirs:
        curves_path = os.path.join(save_dirs['figures'], f'separate_run{run_id}_seed{seed}_curves.png')
    else:
        curves_path = f'separate_run{run_id}_seed{seed}_curves.png'
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_model_path, best_val_acc

if __name__ == "__main__":
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录是父目录
    config_path = os.path.join(current_dir, 'config.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器（可覆盖yaml配置）
    parser = argparse.ArgumentParser(description='训练和测试Separate模型')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'], help='批次大小')
    parser.add_argument('--lr', type=float, default=config['training']['lr'], help='学习率')
    parser.add_argument('--patience', type=int, default=config['training']['patience'], help='早停patience')
    parser.add_argument('--n_runs', type=int, default=config['training']['n_runs'], 
                       help='运行次数（用于显著性检验）')
    parser.add_argument('--seeds', type=int, nargs='+', default=config['seeds'], 
                       help='随机种子列表')
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
    
    # 确保种子数量与运行次数匹配
    if len(args.seeds) < args.n_runs:
        print(f"警告: 提供的种子数量({len(args.seeds)})少于运行次数({args.n_runs})，将使用默认种子")
        args.seeds = config['seeds'][:args.n_runs]
    
    # 创建结果文件夹
    model_name = config['save']['model_name']
    result_folder, models_dir, logs_dir, figures_dir = create_result_folder(
        project_root, model_name, dataset_type='m'
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
    
    # 设置日志
    log_file = os.path.join(logs_dir, 'separate_training.log')
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
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'patience': args.patience,
            'n_runs': args.n_runs,
            'seeds': args.seeds[:args.n_runs],
            'pretrained_path': args.pretrained,
            'img_dir': args.img_dir,
            'device': str(device)
        }, f, allow_unicode=True)
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.RandomRotation(degrees=config['transform']['rotation_degrees']),
        transforms.ToTensor(),
    ])
    
    img_dir_train = os.path.join(args.img_dir, 'train')
    dataset = MixedDataset(img_dir_train, mode="train", transform=transform)
    
    # 存储所有运行的结果
    all_models = []
    all_val_accs = []
    
    logger.info(f"开始使用{args.n_runs}个不同种子进行训练，数据集: mixed")
    logger.info(f"结果保存路径: {result_folder}")
    print(f"\n开始使用{args.n_runs}个不同种子进行训练，数据集: mixed")
    print(f"使用的种子: {args.seeds[:args.n_runs]}")
    
    # 执行多次训练，每次使用不同的种子
    for i in range(args.n_runs):
        seed = args.seeds[i]
        print(f"\n{'='*80}")
        print(f"第 {i+1}/{args.n_runs} 次运行 (种子: {seed})")
        print(f"{'='*80}")
        
        model_path, val_acc = train_and_validate(
            dataset=dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            modelpath=args.pretrained,
            patience=args.patience,
            seed=seed,
            run_id=i+1,
            save_dirs=save_dirs,
            device=device,
            logger=logger
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
    
    # 绘制所有运行的验证准确率对比图
    plt.figure(figsize=(10, 5))
    run_numbers = list(range(1, args.n_runs + 1))
    plt.bar(run_numbers, all_val_accs, color='skyblue', edgecolor='navy')
    plt.axhline(y=mean_val_acc, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val_acc:.4f}')
    plt.title(f'Separate Model - 验证准确率 ({args.n_runs}次运行)', fontsize=12)
    plt.xlabel('运行次数')
    plt.ylabel('验证准确率')
    plt.xticks(run_numbers)
    plt.ylim([min(all_val_accs) - 0.02, max(all_val_accs) + 0.02])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    results_path = os.path.join(figures_dir, f'separate_multi_run_results.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"多次运行结果对比图已保存: {results_path}")
    
    # 保存最终结果到文件
    results_summary_path = os.path.join(result_folder, f'separate_results_summary.txt')
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Separate模型 - MIXED数据集 - 多次运行结果摘要\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"实验配置:\n")
        f.write(f"  数据集: mixed\n")
        f.write(f"  运行次数: {args.n_runs}\n")
        f.write(f"  随机种子: {args.seeds[:args.n_runs]}\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  批次大小: {args.batch_size}\n")
        f.write(f"  学习率: {args.lr}\n")
        f.write(f"  早停patience: {args.patience}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"验证集结果:\n")
        f.write(f"{'='*80}\n")
        f.write(f"平均准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}\n")
        f.write(f"各次运行: {', '.join([f'{acc:.4f}' for acc in all_val_accs])}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"详细结果 (各次运行):\n")
        f.write(f"{'='*80}\n")
        for i in range(args.n_runs):
            f.write(f"\n运行 {i+1} (种子: {args.seeds[i]}):\n")
            f.write(f"  验证准确率: {all_val_accs[i]:.4f}\n")
            f.write(f"  模型路径: {all_models[i]}\n")
    
    print(f"\n结果摘要已保存到: {results_summary_path}")
    logger.info(f"结果摘要已保存到: {results_summary_path}")
    
    completion_info = f"训练完成! 验证准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}"
    print(f"\n{completion_info}")
    logger.info(completion_info)