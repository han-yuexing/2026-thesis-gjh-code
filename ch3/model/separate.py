import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
# from torchsummary import summary

class SeparateModule(nn.Module):
    def __init__(self, modelpath="model/pretrained/resnet18.pth"):
        super(SeparateModule, self).__init__()
        
        # 加载预训练ResNet18（默认使用ImageNet预训练）
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
        
        # 二分类全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),  # ResNet18最终特征维度512
            # nn.Sigmoid()      # 如果用BCELoss需要加Sigmoid
        )

        # 加载自定义预训练权重（如果存在）
        if os.path.exists(modelpath):
            state_dict = torch.load(modelpath)
            # 处理标准ResNet18权重到backbone的映射
            backbone_state_dict = {}
            for key, value in state_dict.items():
                # 跳过fc层（因为我们使用自定义的classifier）
                if key.startswith('fc.'):
                    continue
                # 其他层直接加载到backbone
                backbone_state_dict[key] = value
            
            # 只加载backbone部分，使用strict=False允许缺少fc层
            self.backbone.load_state_dict(backbone_state_dict, strict=False)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    

if __name__ == "__main__":
    # 测试前向传播
    model = SeparateModule()
    dummy_input = torch.randn(2, 3, 224, 224)  # (batch, channel, H, W)
    output = model(dummy_input)
    print(output.shape)  # 预期输出 torch.Size([2, 2])

    # summary(model, (3, 224, 224))