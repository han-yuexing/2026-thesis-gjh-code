import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List

class FruitVision(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout=0.5, freeze_backbone=True):
        super(FruitVision, self).__init__()
        
        # 加载 MobileNetV3-Large 作为特征提取器
        if pretrained:
            print("正在加载预训练的 MobileNetV3-Large 权重...")
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            mobilenet = models.mobilenet_v3_large(weights=weights)
        else:
            mobilenet = models.mobilenet_v3_large(weights=None)
            
        # 去除 MobileNetV3 的分类头，仅保留特征提取部分
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV3-Large 的输出通道数为960
        # 按论文结构，用 flatten，之后接2层全连接
        self.flatten = nn.Flatten()
        # 按论文 Table 2, Flatten 后的特征维度是 47040 (7*7*960)
        self.fc1 = nn.Linear(7*7*960, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

        # 根据论文，在训练过程中可以选择性地冻结骨干网络的权重
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)              # shape: (B, 960, 7, 7)
        x = self.flatten(x)               # shape: (B, 47040)
        x = F.relu(self.fc1(x))           # 按论文未特别指定激活，用relu
        x = self.dropout(x)
        x = self.fc2(x)                   # shape: (B, num_classes)
        return x

# --- 使用示例 ---
if __name__ == '__main__':
    # 创建模型实例
    # 论文提到他们使用了预训练模型，所以 pretrained=True
    # 根据论文，通常冻结骨干网络，只训练分类头
    fruit_vision_model = FruitVision(
        num_classes=4, 
        pretrained=True, 
        dropout=0.5,
        freeze_backbone=True
    )
    
    # 打印模型结构，以验证其正确性
    print("\n--- FruitVision 模型结构 ---")
    print(fruit_vision_model)

    # 模拟一个输入张量 (batch_size=2, channels=3, height=224, width=224)
    # MobileNetV3 预设输入尺寸为 224x224
    dummy_input = torch.randn(2, 3, 224, 224)

    # 执行一次前向传播
    with torch.no_grad():
        output = fruit_vision_model(dummy_input)

    print(f"\n模型输入尺寸: {dummy_input.shape}")
    print(f"模型输出尺寸: {output.shape}")
    print(f"模型输出示例: \n{output}")

    # 验证参数数量是否与论文大致相符
    total_params = sum(p.numel() for p in fruit_vision_model.parameters())
    trainable_params = sum(p.numel() for p in fruit_vision_model.parameters() if p.requires_grad)
    
    print(f"\n模型总参数: {total_params:,}")
    print(f"模型可训练参数: {trainable_params:,}")
    
    # 检查哪些层被冻结了
    print(f"\n冻结的层:")
    for name, param in fruit_vision_model.named_parameters():
        if not param.requires_grad:
            print(f"  {name}: {param.numel():,} 参数")