import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

class FruitVision_Multimodal(nn.Module):
    """
    多模态FruitVision模型 - 支持图片+重量+形状特征的输入
    输入:
      - 图像: (batch_size, 3, 224, 224)
      - 重量: (batch_size, 1)
      - 形状特征: (batch_size, 1)
        - 黄瓜: arch(弯曲度)
        - 香蕉: ripeness(成熟度)
    输出:
      - 分级分类结果: (batch_size, num_classes)
    """
    def __init__(self, num_classes=4, num_shape_features=1, pretrained=True, 
                 dropout=0.5, freeze_backbone=True):
        """
        参数:
            num_classes: 分类类别数 (默认4: super, first, second, third)
            num_shape_features: 形状特征数量 (默认1: 黄瓜arch或香蕉ripeness)
            pretrained: 是否使用预训练的MobileNetV3权重
            dropout: Dropout概率
            freeze_backbone: 是否冻结骨干网络
        """
        super(FruitVision_Multimodal, self).__init__()
        
        # ============ 图像特征提取器 (MobileNetV3-Large) ============
        if pretrained:
            print("正在加载预训练的 MobileNetV3-Large 权重...")
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            mobilenet = models.mobilenet_v3_large(weights=weights)
        else:
            mobilenet = models.mobilenet_v3_large(weights=None)
        
        self.features = mobilenet.features
        
        # 图像特征维度: 7*7*960 = 47040
        self.image_feature_dim = 7 * 7 * 960
        
        # 冻结骨干网络
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # ============ 数值特征处理 ============
        # 重量(1) + 形状特征(num_shape_features)
        self.num_numerical_features = 1 + num_shape_features
        
        # 数值特征归一化层
        self.numerical_bn = nn.BatchNorm1d(self.num_numerical_features)
        
        # 数值特征编码器
        self.numerical_encoder = nn.Sequential(
            nn.Linear(self.num_numerical_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        self.numerical_feature_dim = 128
        
        # ============ 融合分类器 ============
        self.fused_feature_dim = self.image_feature_dim + self.numerical_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fused_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, weight, shape_features):
        """
        前向传播
        参数:
            image: (batch_size, 3, 224, 224) - 图像
            weight: (batch_size, 1) 或 (batch_size,) - 重量
            shape_features: (batch_size, num_shape_features) - 形状特征
        返回:
            logits: (batch_size, num_classes) - 分类输出
        """
        # 1. 提取图像特征
        image_features = self.features(image)     # (batch_size, 960, 7, 7)
        image_features = image_features.view(image_features.size(0), -1)  # (batch_size, 47040)
        
        # 2. 处理重量特征
        if weight.dim() == 1:
            weight = weight.unsqueeze(1)
        
        # 3. 拼接所有数值特征
        numerical_features = torch.cat([weight, shape_features], dim=1)
        
        # 4. 数值特征归一化和编码
        numerical_features = self.numerical_bn(numerical_features)
        numerical_features_encoded = self.numerical_encoder(numerical_features)
        
        # 5. 融合图像特征和数值特征
        fused_features = torch.cat([image_features, numerical_features_encoded], dim=1)
        
        # 6. 分类
        output = self.classifier(fused_features)
        
        return output


class FruitVision_Multimodal_Simple(nn.Module):
    """
    简化版多模态FruitVision模型 - 直接拼接数值特征
    """
    def __init__(self, num_classes=4, num_shape_features=1, pretrained=True,
                 dropout=0.5, freeze_backbone=True):
        super(FruitVision_Multimodal_Simple, self).__init__()
        
        # 图像特征提取器
        if pretrained:
            print("正在加载预训练的 MobileNetV3-Large 权重...")
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            mobilenet = models.mobilenet_v3_large(weights=weights)
        else:
            mobilenet = models.mobilenet_v3_large(weights=None)
        
        self.features = mobilenet.features
        
        self.image_feature_dim = 7 * 7 * 960
        self.num_numerical_features = 1 + num_shape_features
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # 直接拼接，不使用编码器
        self.fused_feature_dim = self.image_feature_dim + self.num_numerical_features
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fused_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, weight, shape_features):
        # 提取图像特征
        image_features = self.features(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # 处理重量
        if weight.dim() == 1:
            weight = weight.unsqueeze(1)
        
        # 拼接所有特征
        numerical_features = torch.cat([weight, shape_features], dim=1)
        fused_features = torch.cat([image_features, numerical_features], dim=1)
        
        # 分类
        output = self.classifier(fused_features)
        return output


# ============ 测试和演示 ============
if __name__ == "__main__":
    print("="*80)
    print("多模态FruitVision模型架构测试")
    print("="*80)
    
    # 测试黄瓜数据集 (1个形状特征: arch弯曲度)
    print("\n【黄瓜数据集 - 完整版模型】")
    cucumber_model = FruitVision_Multimodal(
        num_classes=4, 
        num_shape_features=1,
        pretrained=False,  # 测试时不加载预训练权重
        freeze_backbone=False
    )
    
    # 创建模拟输入
    batch_size = 8
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_weight = torch.randn(batch_size, 1)
    dummy_shape = torch.randn(batch_size, 1)  # arch弯曲度
    
    output = cucumber_model(dummy_image, dummy_weight, dummy_shape)
    print(f"输入: 图像{dummy_image.shape}, 重量{dummy_weight.shape}, 形状{dummy_shape.shape}")
    print(f"输出: {output.shape}")
    
    # 打印模型摘要
    print("\n模型架构摘要:")
    summary(cucumber_model, 
            input_data=[dummy_image, dummy_weight, dummy_shape],
            col_names=["input_size", "output_size", "num_params"],
            depth=3)
    
    # 测试香蕉数据集 (1个形状特征: ripeness)
    print("\n" + "="*80)
    print("【香蕉数据集 - 简化版模型】")
    banana_model = FruitVision_Multimodal_Simple(
        num_classes=4, 
        num_shape_features=1,
        pretrained=False,
        freeze_backbone=False
    )
    
    dummy_shape_banana = torch.randn(batch_size, 1)  # ripeness
    output_banana = banana_model(dummy_image, dummy_weight, dummy_shape_banana)
    print(f"输入: 图像{dummy_image.shape}, 重量{dummy_weight.shape}, 形状{dummy_shape_banana.shape}")
    print(f"输出: {output_banana.shape}")
    
    print("\n模型架构摘要:")
    summary(banana_model,
            input_data=[dummy_image, dummy_weight, dummy_shape_banana],
            col_names=["input_size", "output_size", "num_params"],
            depth=3)
    
    print("\n" + "="*80)
    print("✅ 模型测试完成!")
    print("="*80)

