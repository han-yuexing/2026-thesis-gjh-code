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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
from datetime import datetime
import yaml
import argparse
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score, accuracy_score, mean_squared_error, average_precision_score
import pandas as pd
from PIL import Image

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


class BananaMaskedPseudoDataset(torch.utils.data.Dataset):
    """
    生成图像 + 伪标签数据集。

    默认适配: FruVeg26/data/banana/masked/banana_masked.csv
    CSV列:
      - image: 例如 masked/01_00.png 或 01_00.png 或 绝对路径
      - weight, ripeness
      - grade: super/first/second/third (会映射到level)
    """

    def __init__(self, masked_dir: str, csv_path: str, transform=None):
        self.masked_dir = masked_dir
        self.csv_path = csv_path
        self.transform = transform
        self.level_mapping = {'super': 0, 'first': 1, 'second': 2, 'third': 3}

        df = pd.read_csv(csv_path)
        if "grade" in df.columns and "level" not in df.columns:
            df = df.rename(columns={"grade": "level"})

        required = {"image", "weight", "ripeness", "level"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"masked伪标签csv列不匹配，期望包含{sorted(required)}，实际={list(df.columns)}")

        self.df = df.reset_index(drop=True)

    def subset_by_ratio(self, ratio: float, seed: int):
        if ratio <= 0:
            self.df = self.df.iloc[0:0].copy().reset_index(drop=True)
            return self
        if ratio >= 1:
            return self
        n = len(self.df)
        k = int(round(n * ratio))
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        keep = np.sort(idx[:k])
        self.df = self.df.iloc[keep].reset_index(drop=True)
        return self

    def __len__(self):
        return len(self.df)

    def _resolve_img_path(self, img_field: str) -> str:
        p = str(img_field).strip()
        if os.path.isabs(p) and os.path.exists(p):
            return p

        # 常见情况: 'masked/xx.png'，其父目录就是banana根目录；这里masked_dir已经指向.../banana/masked
        # 所以如果p以'masked/'开头，取basename
        if p.startswith("masked/") or p.startswith("masked\\"):
            p = os.path.basename(p)

        candidate = os.path.join(self.masked_dir, p)
        if os.path.exists(candidate):
            return candidate

        # 兜底：直接拼接原字段（有些人把masked_dir设成banana根目录）
        candidate2 = os.path.join(self.masked_dir, str(img_field))
        return candidate2

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_img_path(row["image"])
        image = Image.open(img_path).convert("RGB")
        weight = torch.tensor(float(row["weight"]), dtype=torch.float32)
        ripeness = torch.tensor(float(row["ripeness"]), dtype=torch.float32)
        level_str = str(row["level"]).strip()
        level = torch.tensor(self.level_mapping[level_str], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, weight, ripeness, level


def _compute_composite_loss(
    loss_type,
    stage,
    gamma,
    multi_task_loss,
    criterion_weight_ripeness,
    criterion_weight,
    criterion_ripeness,
    criterion_class,
    weight_pred,
    ripeness_pred,
    level_pred,
    weight,
    ripeness,
    level,
):
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
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    if stage == 1:
        if loss_type == "multi_task":
            loss = multi_task_loss(loss_weight, loss_ripeness) + gamma * loss_level
        else:
            loss = loss_weight + loss_ripeness + loss_level
    elif stage == 2:
        loss = loss_level
    else:
        raise ValueError(f"unknown stage: {stage}")

    return loss, loss_weight, loss_ripeness, loss_level


def train_model(dataset, num_epochs, batch_size, lr, model_factory, loss_type="multi_task", 
                seed=3407, run_id=1, save_dirs=None, device=None, logger=None, stage=1, gamma=0.1,
                masked_dir=None, masked_csv=None, masked_ratio=0.0, masked_seed=3407, omega=0.3):
    """
    单次训练函数
    loss_type: 损失函数类型
        - "multi_task": 使用MultiTaskLoss + gamma * loss_level
        - "huber": 使用HuberLoss + HuberLoss + CrossEntropyLoss
        - "mse": 使用MSELoss + MSELoss + CrossEntropyLoss
    """
    # 设置当前运行的随机种子
    set_seed(seed)
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 将数据集分为训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 生成图像（masked）只加入训练集，不污染验证集
    masked_train_dataset = None
    if masked_ratio and masked_ratio > 0:
        if masked_dir is None or masked_csv is None:
            raise ValueError("masked_ratio>0时必须同时提供masked_dir与masked_csv")

        masked_ds = BananaMaskedPseudoDataset(
            masked_dir=masked_dir,
            csv_path=masked_csv,
            transform=getattr(dataset, "transform", None),
        )
        masked_ds = masked_ds.subset_by_ratio(float(masked_ratio), seed=int(masked_seed))
        masked_train_dataset = masked_ds
    
    # 为每次运行创建子文件夹
    if save_dirs and 'models' in save_dirs:
        runsave = os.path.join(save_dirs['root'], f"run{run_id}_seed{seed}")
        os.makedirs(runsave, exist_ok=True)
    else:
        runsave = f"run{run_id}_seed{seed}"
        os.makedirs(runsave, exist_ok=True)
    
    if logger:
        logger.info(f'==================== Starting run {run_id} with seed {seed} ====================')
    print(f'==================== Starting run {run_id} with seed {seed} ====================')
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 训练步数：以原始训练集为主（masked只做额外加权项）
    num_training_steps = len(train_dataset) // batch_size * num_epochs  # 总训练步数
    num_warmup_steps = num_training_steps // 10  # 将 warm-up 设置为总步数的 10%

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    masked_data_loader = None
    if masked_train_dataset is not None and len(masked_train_dataset) > 0:
        masked_data_loader = DataLoader(masked_train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model_factory()
    

    if stage == 2:
        statedict = "./save/1216-res18-train1/1/best_epoch98.pth"
        model.load_state_dict(torch.load(statedict))
        logging.info(f"model load {statedict}")
        
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

    # 根据损失函数类型选择不同的损失函数
    multi_task_loss_orig = None
    multi_task_loss_masked = None
    if loss_type == "multi_task":
        criterion_weight_ripeness = nn.HuberLoss()  # 用于重量和成熟度的损失
        criterion_class = nn.CrossEntropyLoss()  # 用于分类的损失
        # 为原始数据与masked伪标签数据分别维护一套可学习方差参数
        multi_task_loss_orig = MultiTaskLoss().to(device)
        multi_task_loss_masked = MultiTaskLoss().to(device)
        criterion_weight = None
        criterion_ripeness = None
    elif loss_type == "huber":
        criterion_weight = nn.HuberLoss()  # 用于重量的损失
        criterion_ripeness = nn.HuberLoss()  # 用于成熟度的损失
        criterion_class = nn.CrossEntropyLoss()  # 用于分类的损失
        criterion_weight_ripeness = None
        multi_task_loss_orig = None
        multi_task_loss_masked = None
    elif loss_type == "mse":
        criterion_weight = nn.MSELoss()  # 用于重量的损失
        criterion_ripeness = nn.MSELoss()  # 用于成熟度的损失
        criterion_class = nn.CrossEntropyLoss()  # 用于分类的损失
        criterion_weight_ripeness = None
        multi_task_loss_orig = None
        multi_task_loss_masked = None

    # multi_task 的可学习方差参数也需要进 optimizer，否则不会更新
    optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if multi_task_loss_orig is not None:
        optim_params += list(multi_task_loss_orig.parameters())
    if multi_task_loss_masked is not None:
        optim_params += list(multi_task_loss_masked.parameters())

    optimizer = optim.AdamW(optim_params, lr=lr)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_metrics = None  # 存储最佳epoch的所有指标

    if logger:
        logger.info(f"loss_type: {loss_type}")
        logger.info(f"gamma: {gamma}")
        logger.info(f"omega(masked): {omega}")


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        masked_iter = iter(masked_data_loader) if masked_data_loader is not None else None

        for step, data in enumerate(train_data_loader):
            # -------- 原始数据 batch --------
            inputs, weight, ripeness, level = data
            inputs = inputs.to(device)
            weight = weight.to(device)
            ripeness = ripeness.to(device)
            level = level.to(device)

            weight_pred, ripeness_pred, level_pred = model(inputs)
            loss_orig, loss_weight, loss_ripeness, loss_level = _compute_composite_loss(
                loss_type=loss_type,
                stage=stage,
                gamma=gamma,
                multi_task_loss=multi_task_loss_orig,
                criterion_weight_ripeness=criterion_weight_ripeness,
                criterion_weight=criterion_weight,
                criterion_ripeness=criterion_ripeness,
                criterion_class=criterion_class,
                weight_pred=weight_pred,
                ripeness_pred=ripeness_pred,
                level_pred=level_pred,
                weight=weight,
                ripeness=ripeness,
                level=level,
            )

            # -------- masked生成数据 batch（可选）--------
            if masked_iter is not None and omega and omega > 0:
                try:
                    m_inputs, m_weight, m_ripeness, m_level = next(masked_iter)
                except StopIteration:
                    masked_iter = iter(masked_data_loader)
                    m_inputs, m_weight, m_ripeness, m_level = next(masked_iter)

                m_inputs = m_inputs.to(device)
                m_weight = m_weight.to(device)
                m_ripeness = m_ripeness.to(device)
                m_level = m_level.to(device)

                m_weight_pred, m_ripeness_pred, m_level_pred = model(m_inputs)
                loss_masked, _, _, _ = _compute_composite_loss(
                    loss_type=loss_type,
                    stage=stage,
                    gamma=gamma,
                    multi_task_loss=multi_task_loss_masked,
                    criterion_weight_ripeness=criterion_weight_ripeness,
                    criterion_weight=criterion_weight,
                    criterion_ripeness=criterion_ripeness,
                    criterion_class=criterion_class,
                    weight_pred=m_weight_pred,
                    ripeness_pred=m_ripeness_pred,
                    level_pred=m_level_pred,
                    weight=m_weight,
                    ripeness=m_ripeness,
                    level=m_level,
                )
                loss = loss_orig + float(omega) * loss_masked
            else:
                loss = loss_orig

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if stage == 1:
                _, predicted = torch.max(level_pred, 1)
                train_correct += (predicted == level).sum().item()
                train_total += level.size(0)

        avg_train_loss = total_train_loss / len(train_data_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        # 收集所有预测值和真实值用于计算指标
        all_true_weights = []
        all_pred_weights = []
        all_true_ripeness = []
        all_pred_ripeness = []
        all_true_levels = []
        all_pred_levels = []
        all_pred_probs = []
        
        with torch.no_grad():
            for step, data in enumerate(val_data_loader):
                inputs, weight, ripeness, level = data
                inputs, weight, ripeness, level = inputs.to(device), weight.to(device), ripeness.to(device), level.to(device)
                weight_pred, ripeness_pred, level_pred = model(inputs)
                
                # 收集预测值和真实值
                all_true_weights.extend(weight.cpu().numpy())
                all_pred_weights.extend(weight_pred.squeeze().cpu().numpy())
                all_true_ripeness.extend(ripeness.cpu().numpy())
                all_pred_ripeness.extend(ripeness_pred.squeeze().cpu().numpy())
                all_true_levels.extend(level.cpu().numpy())
                all_pred_levels.extend(torch.argmax(level_pred, dim=1).cpu().numpy())
                all_pred_probs.extend(torch.nn.functional.softmax(level_pred, dim=1).cpu().numpy())
                
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
                        loss = multi_task_loss_orig(loss_weight, loss_ripeness) + gamma * loss_level
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
                    val_correct += (predicted == level).sum().item()
                    val_total += level.size(0)

        avg_val_loss = total_val_loss / len(val_data_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 计算所有指标（仅在stage==1时）
        if stage == 1:
            # 回归指标
            weight_r2 = r2_score(all_true_weights, all_pred_weights)
            weight_rmse = np.sqrt(mean_squared_error(all_true_weights, all_pred_weights))
            ripeness_r2 = r2_score(all_true_ripeness, all_pred_ripeness)
            ripeness_rmse = np.sqrt(mean_squared_error(all_true_ripeness, all_pred_ripeness))
            
            # 分类指标
            level_precision = precision_score(all_true_levels, all_pred_levels, average='weighted', zero_division=0)
            level_recall = recall_score(all_true_levels, all_pred_levels, average='weighted', zero_division=0)
            level_f1 = f1_score(all_true_levels, all_pred_levels, average='weighted', zero_division=0)
            level_accuracy = accuracy_score(all_true_levels, all_pred_levels)
            
            # mAP
            true_levels_binary = np.eye(4)[all_true_levels]
            level_map = average_precision_score(true_levels_binary, np.array(all_pred_probs), average='weighted')
            
            current_metrics = {
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
        else:
            current_metrics = None

        # 日志和打印
        if stage == 2:
            if logger:
                logger.info(f'------ Epoch {epoch + 1}/{num_epochs} ------')
                logger.info(f'Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
                logger.info(f'Loss Weight: {loss_weight.item()}, Loss Ripeness: {loss_ripeness.item()}, Loss Level: {loss_level.item()}')
            print(f'------ Epoch {epoch + 1}/{num_epochs} ------')
            print(f'Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
            print(f'Loss Weight: {loss_weight.item()}, Loss Ripeness: {loss_ripeness.item()}, Loss Level: {loss_level.item()}')

        elif stage == 1:
            if logger:
                logger.info(f'Run {run_id}, Epoch [{epoch + 1}/{num_epochs}]')
                logger.info(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
                logger.info(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
                logger.info(f'Loss Weight: {loss_weight.item():.4f}, Loss Ripeness: {loss_ripeness.item():.4f}, Loss Level: {loss_level.item():.4f}')
                if current_metrics:
                    logger.info(f'Weight R²: {current_metrics["weight_r2"]:.4f}, RMSE: {current_metrics["weight_rmse"]:.4f}')
                    logger.info(f'Ripeness R²: {current_metrics["ripeness_r2"]:.4f}, RMSE: {current_metrics["ripeness_rmse"]:.4f}')
                    logger.info(f'Level: P={current_metrics["level_precision"]:.4f}, R={current_metrics["level_recall"]:.4f}, F1={current_metrics["level_f1"]:.4f}, mAP={current_metrics["level_map"]:.4f}')
            
            print(f'Run {run_id}, Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            if current_metrics:
                print(f'Weight R²: {current_metrics["weight_r2"]:.4f}, RMSE: {current_metrics["weight_rmse"]:.4f}')
                print(f'Ripeness R²: {current_metrics["ripeness_r2"]:.4f}, RMSE: {current_metrics["ripeness_rmse"]:.4f}')
                print(f'Level: P={current_metrics["level_precision"]:.4f}, R={current_metrics["level_recall"]:.4f}, F1={current_metrics["level_f1"]:.4f}, mAP={current_metrics["level_map"]:.4f}')

        # 保存最佳模型（基于accuracy）
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_metrics = current_metrics.copy() if current_metrics else None
            model_save_path = os.path.join(runsave, f'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
        
        if epoch == num_epochs - 1:
            model_save_path = os.path.join(runsave, f'final_model.pth')
            torch.save(model.state_dict(), model_save_path)

    if logger:
        logger.info(f"Run {run_id} completed! Best validation accuracy: {best_val_acc:.4f}")
        if best_metrics:
            logger.info(f"Best metrics - Weight R²: {best_metrics['weight_r2']:.4f}, RMSE: {best_metrics['weight_rmse']:.4f}")
            logger.info(f"Best metrics - Ripeness R²: {best_metrics['ripeness_r2']:.4f}, RMSE: {best_metrics['ripeness_rmse']:.4f}")
            logger.info(f"Best metrics - Level: P={best_metrics['level_precision']:.4f}, R={best_metrics['level_recall']:.4f}, F1={best_metrics['level_f1']:.4f}, mAP={best_metrics['level_map']:.4f}")

    # 绘图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.title(f'Loss Curve - Run {run_id} (Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy Curve - Run {run_id} (Seed {seed})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{runsave}/training_curves.png')
    plt.close()
    
    return best_val_acc, model_save_path, best_metrics

if __name__ == "__main__":
    # 获取项目根目录和配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 项目根目录是父目录
    config_path = os.path.join(current_dir, 'config.yaml')
    
    # 加载yaml配置
    config = load_config(config_path)
    
    # 创建命令行参数解析器（可覆盖yaml配置）
    parser = argparse.ArgumentParser(description='训练和测试Banana模型')
    parser.add_argument('--config', type=str, default=config_path, help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'], help='批次大小')
    parser.add_argument('--lr', type=float, default=config['training']['lr'], help='学习率')
    parser.add_argument('--n_runs', type=int, default=config['training']['n_runs'], 
                       help='运行次数（用于显著性检验）')
    parser.add_argument('--seeds', type=int, nargs='+', default=config['seeds'], 
                       help='随机种子列表')
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
    parser.add_argument('--masked_ratio', type=float, default=0.0,
                       help='加入训练的生成(masked)图像比例: 0/0.25/0.5/0.75/1.0')
    parser.add_argument('--masked_seed', type=int, default=3407,
                       help='生成(masked)图像抽样随机种子')
    parser.add_argument('--omega', type=float, default=0.3,
                       help='生成(masked)分支权重: L总 = L原始 + omega * L生成')
    parser.add_argument('--masked_dir', type=str, default=None,
                       help='生成(masked)图像目录，例如 FruVeg26/data/banana/masked')
    parser.add_argument('--masked_csv', type=str, default=None,
                       help='生成(masked)伪标签csv，例如 FruVeg26/data/banana/masked/banana_masked.csv')
    parser.add_argument('--stage', type=int, default=config['stage'],
                       help='训练阶段: 1=全模型训练, 2=仅训练level分类器')
    
    args = parser.parse_args()
    
    # 如果指定了不同的配置文件，重新加载
    if args.config != config_path:
        config = load_config(args.config)
    
    # 将相对路径转换为绝对路径（相对于脚本所在目录）
    if not os.path.isabs(args.img_dir):
        args.img_dir = os.path.join(project_root, args.img_dir)
    if not os.path.isabs(args.pretrained):
        args.pretrained = os.path.join(project_root, args.pretrained)

    # masked默认路径（不依赖yaml，直接按你当前数据组织推断）
    if args.masked_dir is None:
        # 例如: <repo>/FruVeg26/data_/banana/train -> <repo>/FruVeg26/data/banana/masked
        # 这里优先用FruVeg26/data/banana/masked（与你提供的路径一致）
        args.masked_dir = os.path.join(project_root, "data", "banana", "masked")
    if args.masked_csv is None:
        args.masked_csv = os.path.join(args.masked_dir, "banana_masked.csv")

    if args.masked_ratio and args.masked_ratio > 0:
        if not os.path.isabs(args.masked_dir):
            args.masked_dir = os.path.join(project_root, args.masked_dir)
        if not os.path.isabs(args.masked_csv):
            args.masked_csv = os.path.join(project_root, args.masked_csv)
    
    # 确保种子数量与运行次数匹配
    if len(args.seeds) < args.n_runs:
        print(f"警告: 提供的种子数量({len(args.seeds)})少于运行次数({args.n_runs})，将使用默认种子")
        args.seeds = config['seeds'][:args.n_runs]
    
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"结果将保存到: {result_folder}")
    
    # 设置日志
    log_file = os.path.join(logs_dir, 'banana_training.log')
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
    logger.info(f"stage: {args.stage}")
    logger.info(f"num_epochs: {args.epochs}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"feature_fusion_type: {args.feature_fusion_type}")
    logger.info(f"fpn_ablation_type: {args.fpn_ablation_type}")
    logger.info(f"combine_type: {args.combine_type}")
    logger.info(f"loss_type: {args.loss_type}")
    logger.info(f"gamma: {args.gamma}")
    logger.info(f"omega(masked): {args.omega}")
    logger.info(f"masked_ratio: {args.masked_ratio}, masked_seed: {args.masked_seed}")
    logger.info(f"n_runs: {args.n_runs}")
    logger.info(f"seeds: {args.seeds[:args.n_runs]}")
    
    # 保存配置到结果文件夹
    config_save_path = os.path.join(result_folder, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'stage': args.stage,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'n_runs': args.n_runs,
            'seeds': args.seeds[:args.n_runs],
            'pretrained_path': args.pretrained,
            'img_dir': args.img_dir,
            'feature_fusion_type': args.feature_fusion_type,
            'fpn_ablation_type': args.fpn_ablation_type,
            'combine_type': args.combine_type,
            'loss_type': args.loss_type,
            'gamma': args.gamma,
            'omega': args.omega,
            'masked_ratio': args.masked_ratio,
            'masked_seed': args.masked_seed,
            'masked_dir': args.masked_dir,
            'masked_csv': args.masked_csv,
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

    # 存储所有运行的结果
    all_val_accs = []
    all_model_paths = []
    all_best_metrics = []  # 存储所有运行的最佳指标
    
    logger.info(f"开始使用{args.n_runs}个不同种子进行训练，数据集: banana")
    logger.info(f"结果保存路径: {result_folder}")
    print(f"\n开始使用{args.n_runs}个不同种子进行训练，数据集: banana")
    print(f"使用的种子: {args.seeds[:args.n_runs]}")
    
    # 执行多次训练，每次使用不同的种子
    for i in range(args.n_runs):
        seed = args.seeds[i]
        print(f"\n{'='*80}")
        print(f"第 {i+1}/{args.n_runs} 次运行 (种子: {seed})")
        print(f"{'='*80}")
        
        val_acc, model_path, best_metrics = train_model(
            dataset=dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_factory=model_factory,
            loss_type=args.loss_type,
            seed=seed,
            run_id=i+1,
            save_dirs=save_dirs,
            device=device,
            logger=logger,
            stage=args.stage,
            gamma=args.gamma,
            masked_dir=args.masked_dir,
            masked_csv=args.masked_csv,
            masked_ratio=args.masked_ratio,
            masked_seed=args.masked_seed,
            omega=args.omega
        )
        
        all_val_accs.append(val_acc)
        all_model_paths.append(model_path)
        all_best_metrics.append(best_metrics)
    
    # 计算验证集的平均准确率和标准差
    mean_val_acc = np.mean(all_val_accs)
    std_val_acc = np.std(all_val_accs, ddof=1)  # 使用样本标准差
    
    # 计算其他指标的平均值和标准差（如果存在）
    if all_best_metrics and all_best_metrics[0] is not None:
        all_weight_r2 = [m['weight_r2'] for m in all_best_metrics if m]
        all_weight_rmse = [m['weight_rmse'] for m in all_best_metrics if m]
        all_ripeness_r2 = [m['ripeness_r2'] for m in all_best_metrics if m]
        all_ripeness_rmse = [m['ripeness_rmse'] for m in all_best_metrics if m]
        all_level_precision = [m['level_precision'] for m in all_best_metrics if m]
        all_level_recall = [m['level_recall'] for m in all_best_metrics if m]
        all_level_f1 = [m['level_f1'] for m in all_best_metrics if m]
        all_level_map = [m['level_map'] for m in all_best_metrics if m]
        
        mean_weight_r2 = np.mean(all_weight_r2)
        std_weight_r2 = np.std(all_weight_r2, ddof=1)
        mean_weight_rmse = np.mean(all_weight_rmse)
        std_weight_rmse = np.std(all_weight_rmse, ddof=1)
        mean_ripeness_r2 = np.mean(all_ripeness_r2)
        std_ripeness_r2 = np.std(all_ripeness_r2, ddof=1)
        mean_ripeness_rmse = np.mean(all_ripeness_rmse)
        std_ripeness_rmse = np.std(all_ripeness_rmse, ddof=1)
        mean_level_precision = np.mean(all_level_precision)
        std_level_precision = np.std(all_level_precision, ddof=1)
        mean_level_recall = np.mean(all_level_recall)
        std_level_recall = np.std(all_level_recall, ddof=1)
        mean_level_f1 = np.mean(all_level_f1)
        std_level_f1 = np.std(all_level_f1, ddof=1)
        mean_level_map = np.mean(all_level_map)
        std_level_map = np.std(all_level_map, ddof=1)
    else:
        all_weight_r2 = all_weight_rmse = all_ripeness_r2 = all_ripeness_rmse = None
        all_level_precision = all_level_recall = all_level_f1 = all_level_map = None
        mean_weight_r2 = std_weight_r2 = mean_weight_rmse = std_weight_rmse = None
        mean_ripeness_r2 = std_ripeness_r2 = mean_ripeness_rmse = std_ripeness_rmse = None
        mean_level_precision = std_level_precision = mean_level_recall = std_level_recall = None
        mean_level_f1 = std_level_f1 = mean_level_map = std_level_map = None
    
    print(f"\n{'='*80}")
    print(f"所有训练运行完成!")
    print(f"验证集准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"各次运行的验证准确率: {[f'{acc:.4f}' for acc in all_val_accs]}")
    if all_best_metrics and all_best_metrics[0] is not None:
        print(f"\n其他指标平均值:")
        print(f"  Weight R²: {mean_weight_r2:.4f} ± {std_weight_r2:.4f}")
        print(f"  Weight RMSE: {mean_weight_rmse:.4f} ± {std_weight_rmse:.4f}")
        print(f"  Ripeness R²: {mean_ripeness_r2:.4f} ± {std_ripeness_r2:.4f}")
        print(f"  Ripeness RMSE: {mean_ripeness_rmse:.4f} ± {std_ripeness_rmse:.4f}")
        print(f"  Level Precision: {mean_level_precision:.4f} ± {std_level_precision:.4f}")
        print(f"  Level Recall: {mean_level_recall:.4f} ± {std_level_recall:.4f}")
        print(f"  Level F1: {mean_level_f1:.4f} ± {std_level_f1:.4f}")
        print(f"  Level mAP: {mean_level_map:.4f} ± {std_level_map:.4f}")
    print(f"{'='*80}")
    
    logger.info(f"所有训练运行完成! 平均验证准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    logger.info(f"各次运行的验证准确率: {all_val_accs}")
    if all_best_metrics and all_best_metrics[0] is not None:
        logger.info(f"平均 Weight R²: {mean_weight_r2:.4f} ± {std_weight_r2:.4f}")
        logger.info(f"平均 Ripeness R²: {mean_ripeness_r2:.4f} ± {std_ripeness_r2:.4f}")
        logger.info(f"平均 Level F1: {mean_level_f1:.4f} ± {std_level_f1:.4f}")
    
    # 绘制所有运行的验证准确率对比图
    plt.figure(figsize=(10, 6))
    run_numbers = list(range(1, args.n_runs + 1))
    plt.bar(run_numbers, all_val_accs, color='skyblue', edgecolor='navy')
    plt.axhline(y=mean_val_acc, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val_acc:.4f}')
    plt.title(f'Banana模型 - 验证准确率 ({args.n_runs}次运行)', fontsize=14)
    plt.xlabel('运行次数')
    plt.ylabel('验证准确率')
    plt.xticks(run_numbers)
    plt.ylim([min(all_val_accs) - 0.02, max(all_val_accs) + 0.02])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    results_path = os.path.join(figures_dir, 'multi_run_validation_results.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"多次运行结果对比图已保存: {results_path}")
    
    # 显著性检验说明
    print(f"\n{'='*80}")
    print("显著性检验说明:")
    print("如果您想要与其他模型进行显著性检验比较，请保存当前结果。")
    print("可以使用 analysis/p-analysis.py 中的 perform_significance_test() 函数进行比较。")
    print(f"当前结果: 验证准确率 = {all_val_accs}")
    print(f"{'='*80}")
    
    logger.info("显著性检验说明: 使用 analysis/p-analysis.py 中的 perform_significance_test() 函数进行模型间比较")
    logger.info(f"当前验证准确率: {all_val_accs}")
    
    # 保存最终结果到文件
    results_summary_path = os.path.join(result_folder, 'banana_results_summary.txt')
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Banana模型 - 多次运行结果摘要\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"实验配置:\n")
        f.write(f"  数据集: banana\n")
        f.write(f"  阶段: {args.stage}\n")
        f.write(f"  运行次数: {args.n_runs}\n")
        f.write(f"  随机种子: {args.seeds[:args.n_runs]}\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  批次大小: {args.batch_size}\n")
        f.write(f"  学习率: {args.lr}\n")
        f.write(f"  特征融合类型: {args.feature_fusion_type}\n")
        f.write(f"  FPN消融类型: {args.fpn_ablation_type}\n")
        f.write(f"  组合类型: {args.combine_type}\n")
        f.write(f"  损失函数类型: {args.loss_type}\n")
        f.write(f"  Gamma: {args.gamma}\n")
        f.write(f"  Omega(masked): {args.omega}\n")
        f.write(f"  masked_ratio: {args.masked_ratio}, masked_seed: {args.masked_seed}\n\n")
        f.write(f"{'='*80}\n")
        f.write(f"验证集结果 (总体统计):\n")
        f.write(f"{'='*80}\n")
        f.write(f"平均准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}\n")
        f.write(f"各次运行准确率: {', '.join([f'{acc:.4f}' for acc in all_val_accs])}\n")
        if all_best_metrics and all_best_metrics[0] is not None:
            f.write(f"\n回归任务:\n")
            f.write(f"  Weight R²: {mean_weight_r2:.4f} ± {std_weight_r2:.4f}\n")
            f.write(f"  Weight RMSE (g): {mean_weight_rmse:.4f} ± {std_weight_rmse:.4f}\n")
            f.write(f"  Ripeness R²: {mean_ripeness_r2:.4f} ± {std_ripeness_r2:.4f}\n")
            f.write(f"  Ripeness RMSE: {mean_ripeness_rmse:.4f} ± {std_ripeness_rmse:.4f}\n")
            f.write(f"\n分类任务:\n")
            f.write(f"  Precision: {mean_level_precision:.4f} ± {std_level_precision:.4f}\n")
            f.write(f"  Recall: {mean_level_recall:.4f} ± {std_level_recall:.4f}\n")
            f.write(f"  F1-Score: {mean_level_f1:.4f} ± {std_level_f1:.4f}\n")
            f.write(f"  mAP: {mean_level_map:.4f} ± {std_level_map:.4f}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"详细结果 (各次运行):\n")
        f.write(f"{'='*80}\n")
        for i in range(args.n_runs):
            f.write(f"\n运行 {i+1} (种子: {args.seeds[i]}):\n")
            f.write(f"  验证准确率: {all_val_accs[i]:.4f}\n")
            if all_best_metrics[i] is not None:
                m = all_best_metrics[i]
                f.write(f"  Weight R²: {m['weight_r2']:.4f}, RMSE: {m['weight_rmse']:.4f}\n")
                f.write(f"  Ripeness R²: {m['ripeness_r2']:.4f}, RMSE: {m['ripeness_rmse']:.4f}\n")
                f.write(f"  Level Precision: {m['level_precision']:.4f}, Recall: {m['level_recall']:.4f}\n")
                f.write(f"  Level F1: {m['level_f1']:.4f}, mAP: {m['level_map']:.4f}\n")
            f.write(f"  模型路径: {all_model_paths[i]}\n")
    
    print(f"\n结果摘要已保存到: {results_summary_path}")
    logger.info(f"结果摘要已保存到: {results_summary_path}")
    
    completion_info = f"训练完成! 验证准确率: {mean_val_acc:.4f} ± {std_val_acc:.4f}"
    print(f"\n{completion_info}")
    logger.info(completion_info)