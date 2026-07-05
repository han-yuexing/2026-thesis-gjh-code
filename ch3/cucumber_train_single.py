import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, accuracy_score, f1_score, average_precision_score
from data import CucumberDataset
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
import math
from model.simple_model import SimpleModel

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
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子（如果有多个）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_and_validate(dataset, num_epochs, batch_size, lr, save, modelpath, switch):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleModel(modelpath, switch=switch).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_data_loader))

    if switch == 1 or switch == 2:
        criterion = nn.HuberLoss()
    elif switch == 3:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_checkpoint_path = None
    no_improve_count = 0
    patience = 10

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        true_train_values = []
        pred_train_values = []

        for step, data in enumerate(train_data_loader):
            inputs, weight, curvature, arch, level = data
            inputs = inputs.to(device)
            weight = weight.to(device)
            curvature = curvature.to(device)
            arch = arch.to(device)
            level = level.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            if switch == 1:
                loss = criterion(output.squeeze(), weight)
                true_train_values.extend(weight.cpu().numpy())
                pred_train_values.extend(output.squeeze().detach().cpu().numpy())  # 使用 detach
            elif switch == 2:
                loss = criterion(output.squeeze(), arch)
                true_train_values.extend(arch.cpu().numpy())
                pred_train_values.extend(output.squeeze().detach().cpu().numpy())  # 使用 detach
            elif switch == 3:
                loss = criterion(output, level)
                _, predicted = torch.max(output, 1)
                correct_train += (predicted == level).sum().item()
                total_train += level.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_data_loader)
        train_losses.append(avg_train_loss)

        if switch == 3:
            train_acc = correct_train / total_train if total_train > 0 else 0
        else:
            train_acc = 0
            if true_train_values and pred_train_values:
                train_r2 = r2_score(true_train_values, pred_train_values)
            else:
                train_r2 = 0

        # 验证
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        true_val_values = []
        pred_val_values = []

        with torch.no_grad():
            for step, data in enumerate(val_data_loader):
                inputs, weight, curvature, arch, level = data
                inputs = inputs.to(device)
                weight = weight.to(device)
                curvature = curvature.to(device)
                arch = arch.to(device)
                level = level.to(device)

                output = model(inputs)

                if switch == 1:
                    loss = criterion(output.squeeze(), weight)
                    true_val_values.extend(weight.cpu().numpy())
                    pred_val_values.extend(output.squeeze().detach().cpu().numpy())  # 使用 detach
                elif switch == 2:
                    loss = criterion(output.squeeze(), arch)
                    true_val_values.extend(arch.cpu().numpy())
                    pred_val_values.extend(output.squeeze().detach().cpu().numpy())  # 使用 detach
                elif switch == 3:
                    loss = criterion(output, level)
                    _, predicted = torch.max(output, 1)
                    correct_val += (predicted == level).sum().item()
                    total_val += level.size(0)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_data_loader)
        val_losses.append(avg_val_loss)

        if switch == 3:
            val_acc = correct_val / total_val if total_val > 0 else 0
        else:
            val_acc = 0
            if true_val_values and pred_val_values:
                val_r2 = r2_score(true_val_values, pred_val_values)
            else:
                val_r2 = 0

        # 日志和打印
        logging.info(f'------ Epoch {epoch + 1}/{num_epochs} ------')
        print(f'------ Epoch {epoch + 1}/{num_epochs} ------')
        if switch == 3:
            logging.info(f'Train Loss: {avg_train_loss}, Train acc: {train_acc}')
            logging.info(f'Val Loss: {avg_val_loss}, Val acc: {val_acc}')
            print(f'Train Loss: {avg_train_loss}, Train acc: {train_acc}')
            print(f'Val Loss: {avg_val_loss}, Val acc: {val_acc}')
        else:
            logging.info(f'Train Loss: {avg_train_loss}, Train R2: {train_r2}')
            logging.info(f'Val Loss: {avg_val_loss}, Val R2: {val_r2}')
            print(f'Train Loss: {avg_train_loss}, Train R2: {train_r2}')
            print(f'Val Loss: {avg_val_loss}, Val R2: {val_r2}')

        # 早停机制
        if switch == 1 or switch == 2:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0
                best_checkpoint_path = os.path.join(save, f'best_epoch{epoch + 1}.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    print(f"Early stopping at epoch {epoch + 1}")
                    if best_checkpoint_path is not None:
                        model.load_state_dict(torch.load(best_checkpoint_path))
                        logging.info(f"Loaded best model from {best_checkpoint_path}")
                    break
        elif switch == 3:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0
                best_checkpoint_path = os.path.join(save, f'best_epoch{epoch + 1}.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    print(f"Early stopping at epoch {epoch + 1}")
                    if best_checkpoint_path is not None:
                        model.load_state_dict(torch.load(best_checkpoint_path))
                        logging.info(f"Loaded best model from {best_checkpoint_path}")
                    break

        if epoch == num_epochs - 1:
            model_save_path = os.path.join(save, f'epoch{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Loss curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save}/loss.png')
    plt.close()

    # 训练结束后加载验证集上表现最好的模型
    if best_checkpoint_path is not None:
        model.load_state_dict(torch.load(best_checkpoint_path))
        logging.info(f"Final best model loaded from {best_checkpoint_path}")

    return model

def test_model(test_dataset, model, switch, save):
    test_data_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    model.eval()

    true_values = []
    pred_values = []
    true_levels = []
    pred_levels = []
    pred_probs = []

    with torch.no_grad():
        for step, data in enumerate(test_data_loader):
            inputs, weight, curvature, arch, level = data
            inputs = inputs.to(device)
            weight = weight.to(device)
            curvature = curvature.to(device)
            arch = arch.to(device)
            level = level.to(device)

            output = model(inputs)

            if switch == 1:
                true_values.extend(weight.cpu().numpy())
                pred_values.extend(output.squeeze().detach().cpu().numpy())  # 使用 detach
            elif switch == 2:
                true_values.extend(arch.cpu().numpy())
                pred_values.extend(output.squeeze().detach().cpu().numpy())  # 使用 detach
            elif switch == 3:
                true_levels.extend(level.cpu().numpy())
                pred_levels.extend(torch.argmax(output, dim=1).cpu().numpy())
                pred_probs.extend(torch.nn.functional.softmax(output, dim=1).cpu().numpy())

    if switch == 1:
        weight_r2 = r2_score(true_values, pred_values)
        print(f'Weight R2: {weight_r2}')
        logging.info(f'Weight R2: {weight_r2}')
    elif switch == 2:
        arch_r2 = r2_score(true_values, pred_values)
        print(f'Arch R2: {arch_r2}')
        logging.info(f'Arch R2: {arch_r2}')
    elif switch == 3:
        level_f1 = f1_score(true_levels, pred_levels, average='weighted')
        level_map = average_precision_score(np.eye(4)[true_levels], np.array(pred_probs), average='weighted')
        level_accuracy = accuracy_score(true_levels, pred_levels)
        print(f'Level F1 Score: {level_f1}')
        print(f'Level mAP: {level_map}')
        print(f'Level Accuracy: {level_accuracy}')
        logging.info(f'Level F1 Score: {level_f1}')
        logging.info(f'Level mAP: {level_map}')
        logging.info(f'Level Accuracy: {level_accuracy}')

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    model_name = 'vit_b_16'
    img_dir = "./data/cucumber/train"
    test_img_dir = "./data/cucumber/test"
    modelpath = f"./model/pretrained/{model_name}.pth"
    num_epochs = 100
    batch_size = 20
    lr = 0.00005

    # 第一次训练：预测 weight
    save_weight = f"./save/{model_name}/simple_train_weight/"
    os.makedirs(save_weight, exist_ok=True)
    logging.basicConfig(filename=save_weight+'train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f"Training for weight prediction")
    dataset = CucumberDataset(img_dir, mode="train", transform=transform)
    test_dataset = CucumberDataset(test_img_dir, mode="test", transform=transform)
    model_weight = train_and_validate(dataset, num_epochs, batch_size, lr, save_weight, modelpath, switch=1)
    test_model(test_dataset, model_weight, switch=1, save=save_weight)

    # 第二次训练：预测 shape
    save_shape = f"./save/{model_name}/simple_train_shape/"
    os.makedirs(save_shape, exist_ok=True)
    logging.basicConfig(filename=save_shape+'train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f"Training for shape prediction")
    dataset = CucumberDataset(img_dir, mode="train", transform=transform)
    test_dataset = CucumberDataset(test_img_dir, mode="test", transform=transform)
    model_shape = train_and_validate(dataset, num_epochs, batch_size, lr, save_shape, modelpath, switch=2)
    test_model(test_dataset, model_shape, switch=2, save=save_shape)

    # 第三次训练：分类 level
    save_level = f"./save/{model_name}/simple_train_level/"
    os.makedirs(save_level, exist_ok=True)
    logging.basicConfig(filename=save_level+'train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f"Training for level classification")
    dataset = CucumberDataset(img_dir, mode="train", transform=transform)
    test_dataset = CucumberDataset(test_img_dir, mode="test", transform=transform)
    model_level = train_and_validate(dataset, num_epochs, batch_size, lr, save_level, modelpath, switch=3)
    test_model(test_dataset, model_level, switch=3, save=save_level)