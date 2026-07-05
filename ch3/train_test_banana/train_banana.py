import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import logging
import os
import sys
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
from datetime import datetime
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

# 创建带日期和序号的结果文件夹
def create_result_folder(project_root, model_name, dataset_type='b'):
    """
    创建结果文件夹，命名格式: 日期-模型-b-序号
    dataset_type: 'c' for cucumber, 'b' for banana
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


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 可学习的不确定性参数
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # weight 的 log(σ)
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # ripeness 的 log(σ)

    def forward(self, loss_weight, loss_ripeness):
        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)
        total_loss = (1 / (2 * sigma1**2)) * loss_weight + (1 / (2 * sigma2**2)) * loss_ripeness
        total_loss += self.log_sigma1 + self.log_sigma2  # 正则化项
        return total_loss


def train_and_cross_validate(dataset, num_epochs, batch_size, lr, model_factory, loss_type="multi_task",
                            n_folds=5, seed=42, save_dirs=None, device=None, logger=None, stage=1, gamma=0.1):
    """
    训练和交叉验证函数
    loss_type: 损失函数类型
        - "multi_task": 使用MultiTaskLoss + gamma * loss_level
        - "huber": 使用HuberLoss + HuberLoss + CrossEntropyLoss
        - "mse": 使用MSELoss + MSELoss + CrossEntropyLoss
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    all_train_loss = []
    all_val_loss = []

    num_training_steps = len(dataset) // batch_size * num_epochs  # 总训练步数
    num_warmup_steps = num_training_steps // 10  # 将 warm-up 设置为总步数的 10%


    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        
        # 为每个fold创建子文件夹
        if save_dirs and 'models' in save_dirs:
            foldsave_models = os.path.join(save_dirs['models'], f"fold{fold + 1}")
            foldsave_figures = os.path.join(save_dirs['figures'], f"fold{fold + 1}")
            os.makedirs(foldsave_models, exist_ok=True)
            os.makedirs(foldsave_figures, exist_ok=True)
        else:
            foldsave_models = f"fold{fold + 1}"
            foldsave_figures = f"fold{fold + 1}"
            os.makedirs(foldsave_models, exist_ok=True)
            os.makedirs(foldsave_figures, exist_ok=True)
        
        if logger:
            logger.info(f'==================== Starting fold {fold + 1} ====================')
        print(f'==================== Starting fold {fold + 1} ====================')
        train_losses = []
        val_losses = []

        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = model_factory()
        

        if stage == 2:
            statedict = "./save/1216-res18-train1/1/best_epoch98.pth"
            model.load_state_dict(torch.load(statedict))
            if logger:
                logger.info(f"model load {statedict}")
            
            # 冻结 weight_block 和 curvature_block 的参数
            for param in model.lraspp.parameters():
                param.requires_grad = False

            for param in model.fpn.parameters():
                param.requires_grad = False

            for param in model.feature_combine.parameters():
                param.requires_grad = False
            
            for param in model.weight_module.parameters():
                param.requires_grad = False

            for param in model.ripe_module.parameters():
                param.requires_grad = False
                            
            for param in model.weight_ffn.parameters():
                param.requires_grad = False

            for param in model.ripe_ffn.parameters():
                param.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
        
        # 根据损失函数类型选择不同的损失函数
        if loss_type == "multi_task":
            criterion_weight_ripeness = nn.HuberLoss()  # 用于重量和成熟度的损失
            criterion_class = nn.CrossEntropyLoss()  # 用于分类的损失
            multi_task_loss = MultiTaskLoss().to(device)
        elif loss_type == "huber":
            criterion_weight = nn.HuberLoss()  # 用于重量的损失
            criterion_ripeness = nn.HuberLoss()  # 用于成熟度的损失
            criterion_class = nn.CrossEntropyLoss()  # 用于分类的损失
        elif loss_type == "mse":
            criterion_weight = nn.MSELoss()  # 用于重量的损失
            criterion_ripeness = nn.MSELoss()  # 用于成熟度的损失
            criterion_class = nn.CrossEntropyLoss()  # 用于分类的损失
        
        best_val_loss = float('inf')

        if logger:
            logger.info(f"Fold {fold + 1} - loss_type: {loss_type}")
            logger.info(f"Fold {fold + 1} - gamma: {gamma}")


        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for step, data in enumerate(train_data_loader):
                inputs, weight, ripeness, level = data
                
                inputs, weight, ripeness, level = inputs.to(device), weight.to(device), ripeness.to(device), level.to(device)
                weight_pred, ripeness_pred, level_pred = model(inputs)
                
                # 计算损失
                if loss_type == "multi_task":
                    loss_weight = criterion_weight_ripeness(weight_pred.squeeze(), weight)
                    loss_ripeness = criterion_weight_ripeness(ripeness_pred.squeeze(), ripeness)
                    loss_level = criterion_class(level_pred, level)
                elif loss_type == "huber":
                    loss_weight = criterion_weight(weight_pred.squeeze(), weight)
                    loss_ripeness = criterion_ripeness(ripeness_pred.squeeze(), ripeness)
                    loss_level = criterion_class(level_pred, level)
                elif loss_type == "mse":
                    loss_weight = criterion_weight(weight_pred.squeeze(), weight)
                    loss_ripeness = criterion_ripeness(ripeness_pred.squeeze(), ripeness)
                    loss_level = criterion_class(level_pred, level)

                
                if stage == 1:
                    if loss_type == "multi_task":
                        # 使用MultiTaskLoss + gamma * loss_level
                        loss = multi_task_loss(loss_weight, loss_ripeness) + gamma * loss_level
                    elif loss_type == "huber":
                        # 使用HuberLoss + HuberLoss + CrossEntropyLoss
                        loss = loss_weight + loss_ripeness + loss_level
                    elif loss_type == "mse":
                        # 使用MSELoss + MSELoss + CrossEntropyLoss
                        loss = loss_weight + loss_ripeness + loss_level

                elif stage == 2:
                    loss = loss_level

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # 更新学习率

                total_train_loss += loss.item()

                if stage == 1:
                    _, predicted = torch.max(level_pred, 1)
                    correct = (predicted == level).sum().item()
                    accuracy_train = correct / level.size(0)

            avg_train_loss = total_train_loss / len(train_data_loader)
            train_losses.append(avg_train_loss)

            # 验证
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for step, data in enumerate(val_data_loader):
                    inputs, weight, ripeness, level = data
                    inputs, weight, ripeness, level = inputs.to(device), weight.to(device), ripeness.to(device), level.to(device)
                    weight_pred, ripeness_pred, level_pred = model(inputs)
                    
                    # 计算损失
                    if loss_type == "multi_task":
                        loss_weight = criterion_weight_ripeness(weight_pred.squeeze(), weight)
                        loss_ripeness = criterion_weight_ripeness(ripeness_pred.squeeze(), ripeness)
                        loss_level = criterion_class(level_pred, level)
                    elif loss_type == "huber":
                        loss_weight = criterion_weight(weight_pred.squeeze(), weight)
                        loss_ripeness = criterion_ripeness(ripeness_pred.squeeze(), ripeness)
                        loss_level = criterion_class(level_pred, level)
                    elif loss_type == "mse":
                        loss_weight = criterion_weight(weight_pred.squeeze(), weight)
                        loss_ripeness = criterion_ripeness(ripeness_pred.squeeze(), ripeness)
                        loss_level = criterion_class(level_pred, level)

                    if stage == 1:
                        if loss_type == "multi_task":
                            # 使用MultiTaskLoss + gamma * loss_level
                            loss = multi_task_loss(loss_weight, loss_ripeness) + gamma * loss_level
                        elif loss_type == "huber":
                            # 使用HuberLoss + HuberLoss + CrossEntropyLoss
                            loss = loss_weight + loss_ripeness + loss_level
                        elif loss_type == "mse":
                            # 使用MSELoss + MSELoss + CrossEntropyLoss
                            loss = loss_weight + loss_ripeness + loss_level

                    elif stage == 2:
                        loss = loss_level

                    total_val_loss += loss.item()

                    if stage == 1:
                        _, predicted = torch.max(level_pred, 1)
                        correct = (predicted == level).sum().item()
                        accuracy_val = correct / level.size(0)

            avg_val_loss = total_val_loss / len(val_data_loader)
            val_losses.append(avg_val_loss)

            # 日志和打印
            if stage == 2:
                if logger:
                    logger.info(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}')
                    logger.info(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                    logger.info(f'Loss Weight: {loss_weight.item():.4f}, Loss Ripeness: {loss_ripeness.item():.4f}, Loss Level: {loss_level.item():.4f}')
                print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}')
                print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'Loss Weight: {loss_weight.item():.4f}, Loss Ripeness: {loss_ripeness.item():.4f}, Loss Level: {loss_level.item():.4f}')

            elif stage == 1:
                if logger:
                    logger.info(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}')
                    logger.info(f'Train Loss: {avg_train_loss:.4f}, Train acc: {accuracy_train:.4f}')
                    logger.info(f'Val Loss: {avg_val_loss:.4f}, Val acc: {accuracy_val:.4f}')
                    logger.info(f'Loss Weight: {loss_weight.item():.4f}, Loss Ripe: {loss_ripeness.item():.4f}, Loss Level: {loss_level.item():.4f}')
                
                print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}')
                print(f'Train Loss: {avg_train_loss:.4f}, Train acc: {accuracy_train:.4f}')
                print(f'Val Loss: {avg_val_loss:.4f}, Val acc: {accuracy_val:.4f}')
                print(f'Loss Weight: {loss_weight.item():.4f}, Loss Ripe: {loss_ripeness.item():.4f}, Loss Level: {loss_level.item():.4f}')


            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = os.path.join(foldsave_models, f'best_epoch{epoch + 1}.pth')
                torch.save(model.state_dict(), model_save_path)
            
            if epoch == num_epochs - 1:
                model_save_path = os.path.join(foldsave_models, f'epoch{epoch + 1}.pth')
                torch.save(model.state_dict(), model_save_path)

        all_train_loss.append(train_losses)
        all_val_loss.append(val_losses)

        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
        plt.title(f'Loss curves for fold {fold + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(foldsave_figures, f'fold{fold + 1}_loss.png'))
        plt.close()

if __name__ == "__main__":
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录是父目录
    config_path = os.path.join(current_dir, 'config_kfold.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器（可覆盖yaml配置）
    parser = argparse.ArgumentParser(description='训练Banana模型（K-Fold交叉验证）')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'], help='批次大小')
    parser.add_argument('--lr', type=float, default=config['training']['lr'], help='学习率')
    parser.add_argument('--seed', type=int, default=config['training']['seed'], help='随机种子')
    parser.add_argument('--n_folds', type=int, default=config['training']['n_folds'], help='K-fold折数')
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
    parser.add_argument('--loss_type', type=str, default=config['loss']['type'],
                       help='损失函数类型')
    parser.add_argument('--gamma', type=float, default=config['loss']['gamma'],
                       help='loss_level的权重系数')
    parser.add_argument('--stage', type=int, default=config['stage'],
                       help='训练阶段: 1=全模型训练, 2=仅训练level分类器')
    
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载
    if args.config != config_path:
        config = load_config(args.config)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 将相对路径转换为绝对路径
    if not os.path.isabs(args.img_dir):
        args.img_dir = os.path.join(project_root, args.img_dir)
    if not os.path.isabs(args.pretrained):
        args.pretrained = os.path.join(project_root, args.pretrained)
    
    # 创建结果文件夹
    model_name = config['save']['model_name']
    result_folder, models_dir, logs_dir, figures_dir = create_result_folder(
        project_root, model_name, dataset_type='b'
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
    log_file = os.path.join(logs_dir, 'banana_kfold_training.log')
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
    logger.info(f"K-Fold交叉验证训练 - Seed: {args.seed}")
    logger.info(f"stage: {args.stage}")
    logger.info(f"num_epochs: {args.epochs}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"n_folds: {args.n_folds}")
    logger.info(f"feature_fusion_type: {args.feature_fusion_type}")
    logger.info(f"fpn_ablation_type: {args.fpn_ablation_type}")
    logger.info(f"combine_type: {args.combine_type}")
    logger.info(f"loss_type: {args.loss_type}")
    logger.info(f"gamma: {args.gamma}")
    
    # 保存配置到结果文件夹
    config_save_path = os.path.join(result_folder, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'stage': args.stage,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'n_folds': args.n_folds,
            'pretrained_path': args.pretrained,
            'img_dir': args.img_dir,
            'feature_fusion_type': args.feature_fusion_type,
            'fpn_ablation_type': args.fpn_ablation_type,
            'combine_type': args.combine_type,
            'loss_type': args.loss_type,
            'gamma': args.gamma,
            'device': str(device)
        }, f, allow_unicode=True)
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.RandomRotation(degrees=config['transform']['rotation_degrees']),
        transforms.ToTensor(),
    ])
    
    img_dir_train = os.path.join(args.img_dir, 'train')
    dataset = BananaDataset(img_dir_train, mode="train", transform=transform)

    def model_factory():
        return BananaModel(
            args.pretrained, 
            feature_fusion_type=args.feature_fusion_type, 
            fpn_ablation_type=args.fpn_ablation_type, 
            combine_type=args.combine_type
        ).to(device)

    train_and_cross_validate(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_factory=model_factory,
        loss_type=args.loss_type,
        n_folds=args.n_folds,
        seed=args.seed,
        save_dirs=save_dirs,
        device=device,
        logger=logger,
        stage=args.stage,
        gamma=args.gamma
    )
    
    logger.info("K-Fold交叉验证训练完成！")
    print("\nK-Fold交叉验证训练完成！")
    print(f"结果已保存到: {result_folder}")