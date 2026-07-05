import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

class CIDIS_Model(nn.Module):
    """
    根据论文图6复现的CIDIS模型PyTorch实现。
    """
    def __init__(self, num_classes=4):
        super(CIDIS_Model, self).__init__()
        
        # 特征提取器 (Feature Extractor)
        # 对应论文中的三个"卷积-卷积-池化"块
        self.features = nn.Sequential(
            # --- 第一个卷积块 ---
            # 论文描述: 两个3x3卷积层，一个4x4最大池化层
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),

            # --- 第二个卷积块 ---
            # 论文描述: 两个3x3卷积层，一个4x4最大池化层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),

            # --- 第三个卷积块 ---
            # 论文描述: 两个3x3卷积层，一个3x3最大池化层
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        
        # 分类器 (Classifier)
        # 对应论文中的全连接层部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 手动计算特征图尺寸:
            # 输入: 224x224
            # 第一个池化: 224/4 = 56x56
            # 第二个池化: 56/4 = 14x14  
            # 第三个池化: 14/3 = 4.67 -> 4x4 (向下取整)
            # 最终特征图: 4x4x32 = 512
            nn.Linear(in_features=4*4*32, out_features=1024),
            nn.ReLU(inplace=True),
            # 根据论文表5，最优模型使用了Dropout层 
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 创建并查看模型摘要 ---
# 需要安装 torchinfo: pip install torchinfo
input_size = (1, 3, 224, 224) # (batch_size, channels, height, width)
cidis_model = CIDIS_Model()
print("--- CIDIS 模型架构 ---")
summary(cidis_model, input_size=input_size)