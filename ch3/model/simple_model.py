import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SimpleModel(nn.Module):
    def __init__(self, modelpath="resnet18.pth", switch=1):
        super(SimpleModel, self).__init__()
        self.switch = switch
        
        # 从模型路径中提取模型名称
        model_name = modelpath.split('/')[-1].split('.')[0]
        self.model_name = model_name
        
        # 根据模型名称创建基础模型
        if 'resnet18' in model_name:
            self.base_model = timm.create_model('resnet18', pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        elif 'resnet34' in model_name:
            self.base_model = timm.create_model('resnet34', pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        elif 'resnet50' in model_name:
            self.base_model = timm.create_model('resnet50', pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        elif 'vgg11' in model_name:
            self.base_model = timm.create_model('vgg11', pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        elif 'vgg16' in model_name:
            self.base_model = timm.create_model('vgg16', pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        elif 'vit_tiny' in model_name:
            # 对于 vit_tiny，取消 features_only 参数
            self.base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        else:
            raise ValueError(f"Unsupported model name in modelpath: {model_name}")
        
        # 加载状态字典
        state_dict = torch.load(modelpath, weights_only=True)
        
        # 针对 vgg 模型，修改状态字典的键名并删除 classifier 部分的权重
        if 'vgg' in model_name:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('features.'):
                    new_k = k.replace('features.', 'features_')
                    new_state_dict[new_k] = v
                elif k.startswith('classifier'):
                    continue
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # 针对 vit_tiny，如果需要调整状态字典，可添加相应处理（此处暂不修改）
        if 'vit_tiny' in model_name:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k] = v
            state_dict = new_state_dict
        
        # 移除 fc 相关的权重
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        
        # 加载修改后的状态字典
        self.base_model.load_state_dict(state_dict, strict=False)
        
        # 保存特征维度信息
        if not ('vit_tiny' in model_name):
            self.out_channels = [info["num_chs"] for info in self.base_model.feature_info]
        else:
            # 对于 vit_tiny，通常使用模型的 embed_dim 作为特征维度
            self.out_channels = [self.base_model.embed_dim]
        
        # 根据 switch 添加不同任务的全连接层
        if switch == 1:
            self.weight_ffn = nn.Sequential(
                nn.Linear(self.out_channels[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        elif switch == 2:
            self.shape_ffn = nn.Sequential(
                nn.Linear(self.out_channels[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        elif switch == 3:
            self.level_ffn = nn.Sequential(
                nn.Linear(self.out_channels[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 4)
            )

    def forward(self, x):
        if 'vit_tiny' in self.model_name:
            # 对于 vit_tiny，调用 forward_features 获取特征向量
            final_feature = self.base_model.forward_features(x)
        else:
            features = self.base_model(x)
            final_feature = F.adaptive_avg_pool2d(features[-1], 1).reshape(x.shape[0], -1)
        
        if self.switch == 1:
            weight = self.weight_ffn(final_feature)
            return weight
        elif self.switch == 2:
            shape = self.shape_ffn(final_feature)
            return shape
        elif self.switch == 3:
            logits = self.level_ffn(final_feature)
            return logits
        else:
            raise ValueError("Switch must be 1, 2, or 3")