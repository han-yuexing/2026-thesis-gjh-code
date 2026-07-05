import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

class CIDIS_Model_Multimodal(nn.Module):
    """
    多模态CIDIS模型 - 支持图片+重量+形状特征的输入
    输入:
      - 图像: (batch_size, 3, 224, 224)
      - 重量: (batch_size, 1)
      - 形状特征: (batch_size, 1)
        - 黄瓜: arch(弯曲度)
        - 香蕉: ripeness(成熟度)
    输出:
      - 分级分类结果: (batch_size, num_classes)
    """
    def __init__(self, num_classes=4, num_shape_features=1):
        """
        参数:
            num_classes: 分类类别数 (默认4: super, first, second, third)
            num_shape_features: 形状特征数量 (默认1: 黄瓜arch或香蕉ripeness)
        """
        super(CIDIS_Model_Multimodal, self).__init__()
        
        # ============ 图像特征提取器 (与原CIDIS相同) ============
        self.features = nn.Sequential(
            # --- 第一个卷积块 ---
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),

            # --- 第二个卷积块 ---
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),

            # --- 第三个卷积块 ---
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        
        # 图像特征维度: 4x4x32 = 512
        self.image_feature_dim = 4 * 4 * 32
        
        # ============ 数值特征处理 ============
        # 重量(1) + 形状特征(num_shape_features)
        self.num_numerical_features = 1 + num_shape_features
        
        # 数值特征归一化层 (BatchNorm1d)
        self.numerical_bn = nn.BatchNorm1d(self.num_numerical_features)
        
        # 数值特征编码器 (可选: 将数值特征映射到更高维空间)
        self.numerical_encoder = nn.Sequential(
            nn.Linear(self.num_numerical_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        self.numerical_feature_dim = 128
        
        # ============ 融合分类器 ============
        # 拼接图像特征和数值特征
        self.fused_feature_dim = self.image_feature_dim + self.numerical_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.fused_feature_dim, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, image, weight, shape_features):
        """
        前向传播
        参数:
            image: (batch_size, 3, 224, 224) - 图像
            weight: (batch_size, 1) 或 (batch_size,) - 重量
            shape_features: (batch_size, num_shape_features) - 形状特征
                黄瓜: [curvature, arch]
                香蕉: [ripeness]
        返回:
            logits: (batch_size, num_classes) - 分类输出
        """
        # 1. 提取图像特征
        image_features = self.features(image)  # (batch_size, 32, 4, 4)
        image_features = image_features.view(image_features.size(0), -1)  # (batch_size, 512)
        
        # 2. 处理重量特征
        if weight.dim() == 1:
            weight = weight.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
        
        # 3. 拼接所有数值特征
        numerical_features = torch.cat([weight, shape_features], dim=1)  # (batch_size, 1+num_shape)
        
        # 4. 数值特征归一化和编码
        numerical_features = self.numerical_bn(numerical_features)
        numerical_features_encoded = self.numerical_encoder(numerical_features)  # (batch_size, 128)
        
        # 5. 融合图像特征和数值特征
        fused_features = torch.cat([image_features, numerical_features_encoded], dim=1)
        
        # 6. 分类
        output = self.classifier(fused_features)
        
        return output


class CIDIS_Model_Multimodal_Simple(nn.Module):
    """
    简化版多模态CIDIS模型 - 直接拼接数值特征，不使用编码器
    适合数据量较小或特征较少的情况
    """
    def __init__(self, num_classes=4, num_shape_features=2):
        super(CIDIS_Model_Multimodal_Simple, self).__init__()
        
        # 图像特征提取器 (与原CIDIS相同)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        
        self.image_feature_dim = 4 * 4 * 32  # 512
        self.num_numerical_features = 1 + num_shape_features
        
        # 直接拼接，不使用编码器
        self.fused_feature_dim = self.image_feature_dim + self.num_numerical_features
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.fused_feature_dim, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, image, weight, shape_features):
        # 提取图像特征
        image_features = self.features(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # 处理重量
        if weight.dim() == 1:
            weight = weight.unsqueeze(1)
        
        # 拼接所有特征 (图像特征 + 重量 + 形状特征)
        numerical_features = torch.cat([weight, shape_features], dim=1)
        fused_features = torch.cat([image_features, numerical_features], dim=1)
        
        # 分类
        output = self.classifier(fused_features)
        return output


# ============ 测试和演示 ============
if __name__ == "__main__":
    print("="*80)
    print("多模态CIDIS模型架构测试")
    print("="*80)
    
    # 测试黄瓜数据集 (1个形状特征: arch弯曲度)
    print("\n【黄瓜数据集 - 完整版模型】")
    cucumber_model = CIDIS_Model_Multimodal(num_classes=4, num_shape_features=1)
    
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
    banana_model = CIDIS_Model_Multimodal_Simple(num_classes=4, num_shape_features=1)
    
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


