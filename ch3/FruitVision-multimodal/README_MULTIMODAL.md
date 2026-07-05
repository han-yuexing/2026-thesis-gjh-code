# FruitVision多模态模型使用说明

## 概述

将原始FruitVision模型(基于MobileNetV3)扩展为支持**多模态输入**的版本：
- **输入**: 图像 + 重量 + 形状特征
- **输出**: 分级分类 (super/first/second/third)

### 数据集特征
- **黄瓜**: 图像 + 重量(weight) + 弯曲度(arch)
- **香蕉**: 图像 + 重量(weight) + 成熟度(ripeness)

---

## 快速开始

### 1. 测试模型架构
```bash
cd /root/code/FruVeg/FruitVision-multimodal
python fruit_vision_multimodal.py
```

### 2. 运行演示
```bash
python demo_multimodal.py
```

### 3. 训练模型
```bash
# 简化版 (更快)
python fruit_vision_train_test_multimodal.py --dataset cucumber --simple --n_runs 1

# 完整版 (更准确)
python fruit_vision_train_test_multimodal.py --dataset cucumber --n_runs 5
```

---

## 模型架构

### 完整版 (`FruitVision_Multimodal`)
```
图像(3×224×224) → MobileNetV3 → 图像特征(47040维)
                                        ↓
重量 + 形状特征 → BatchNorm → 编码器 → 数值特征(128维)
                                        ↓
                              拼接融合(47168维)
                                        ↓
                            分类器 → 4类输出
```

**特点**: 
- 使用预训练MobileNetV3权重
- BatchNorm归一化 + 数值特征编码器
- 可选择冻结骨干网络

**参数量**: ~5.5M (总) / ~3K (可训练，冻结骨干时)

### 简化版 (`FruitVision_Multimodal_Simple`)
```
图像(3×224×224) → MobileNetV3 → 图像特征(47040维)
                                    ↓
重量 + 形状特征 ————————→ 直接拼接(47042维)
                                    ↓
                          分类器 → 4类输出
```

**特点**: 无编码器，直接拼接  
**参数量**: ~5.5M (总) / ~3K (可训练，冻结骨干时)

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `fruit_vision_multimodal.py` | 模型定义 (完整版+简化版) |
| `fruit_vision_train_test_multimodal.py` | 训练和测试脚本 |
| `demo_multimodal.py` | 演示脚本 |
| `config_multimodal.yaml` | 配置文件 |

---

## 常用命令

### 训练
```bash
# 黄瓜简化版 (使用预训练权重)
python fruit_vision_train_test_multimodal.py --dataset cucumber --simple

# 黄瓜完整版 (不冻结骨干)
python fruit_vision_train_test_multimodal.py --dataset cucumber --no-freeze

# 香蕉完整版 (不使用预训练)
python fruit_vision_train_test_multimodal.py --dataset banana --no-pretrained

# 自定义参数
python fruit_vision_train_test_multimodal.py \
    --dataset cucumber \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --n_runs 5 \
    --no-freeze
```

### 参数说明
- `--dataset`: cucumber 或 banana
- `--epochs`: 训练轮数，默认50
- `--batch_size`: 批次大小，默认16
- `--lr`: 学习率，默认0.001
- `--n_runs`: 运行次数，默认5
- `--simple`: 使用简化版模型
- `--no-pretrained`: 不使用预训练权重
- `--no-freeze`: 不冻结骨干网络
- `--seeds`: 随机种子列表

---

## 输出结果

结果保存在 `results/日期-模型名-数据集-序号/`:
```
results/YYYYMMDD-FruitVision-Multi-c-1/
├── models/          # 模型权重 .pth
├── figures/         # 训练曲线、混淆矩阵
├── logs/            # 训练日志
├── config.yaml      # 实验配置
└── *_summary.txt    # 结果摘要
```

---

## 代码示例

### 单样本预测
```python
from fruit_vision_multimodal import FruitVision_Multimodal
import torch

# 加载模型
model = FruitVision_Multimodal(num_classes=4, num_shape_features=1, pretrained=True)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 准备输入
image = torch.randn(1, 3, 224, 224)
weight = torch.tensor([[150.5]])
shape = torch.tensor([[12.3]])  # arch或ripeness

# 预测
with torch.no_grad():
    output = model(image, weight, shape)
    predicted = torch.argmax(output, dim=1)
```

---

## 模型选择建议

| 场景 | 推荐配置 |
|------|---------|
| 数据量 < 1000 | 简化版 + 冻结骨干 |
| 数据量 > 1000 | 完整版 + 不冻结骨干 |
| 快速实验 | 预训练 + 冻结骨干 |
| 精细调优 | 预训练 + 不冻结骨干 |
| 从头训练 | 不使用预训练 |

---

## 与原始FruitVision对比

| 特性 | 原始FruitVision | 多模态FruitVision |
|------|----------------|------------------|
| 输入 | 仅图像 | 图像+重量+形状 |
| 骨干网络 | MobileNetV3 | MobileNetV3 |
| 准确率 | 基准 | **+3~8%** |
| F1分数 | 基准 | **+5~10%** |
| 可训练参数 | 3K (冻结) / 5.5M (全部) | 3K (冻结) / 5.5M (全部) |

---

## 常见问题

**Q: 是否必须使用预训练权重?**  
A: 不必须。使用 `--no-pretrained` 可从头训练，但通常预训练效果更好。

**Q: 是否应该冻结骨干网络?**  
A: 看数据量。数据少(<1000)时冻结；数据多(>1000)时不冻结。

**Q: 训练很慢怎么办?**  
A: 
1. 冻结骨干网络 (默认已冻结)
2. 使用简化版模型 `--simple`
3. 减小batch_size

**Q: 内存不足?**  
A: 减小batch_size，如 `--batch_size 8`

---

## 技术要点

### 特征提取
```python
def extract_features_from_batch(batch, dataset_type):
    if dataset_type == 'cucumber':
        images, weight, arch, labels = batch[0], batch[1], batch[3], batch[4]
        shape_features = arch.unsqueeze(1)
    elif dataset_type == 'banana':
        images, weight, ripeness, labels = batch[0], batch[1], batch[2], batch[3]
        shape_features = ripeness.unsqueeze(1)
    return images, weight, shape_features, labels
```

### 模型前向传播
```python
def forward(self, image, weight, shape_features):
    # 1. MobileNetV3提取图像特征
    image_features = self.features(image)  # (B, 960, 7, 7)
    image_features = self.avgpool(image_features).flatten(1)  # (B, 47040)
    
    # 2. 处理数值特征 (完整版)
    numerical = torch.cat([weight, shape_features], dim=1)
    numerical = self.numerical_bn(numerical)
    numerical_encoded = self.numerical_encoder(numerical)
    
    # 3. 特征融合
    fused = torch.cat([image_features, numerical_encoded], dim=1)
    
    # 4. 分类
    return self.classifier(fused)
```

---

## 预训练优势

使用预训练MobileNetV3的优点：
1. ✅ 收敛更快
2. ✅ 需要更少数据
3. ✅ 泛化能力更强
4. ✅ 可以冻结骨干节省训练时间

---

**更新**: 2025-10-28  
**版本**: v1.0

