import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import re

try:
    from .modules import CrossAttention, LKA_Attention, FPN
except ImportError:
    from modules import CrossAttention, LKA_Attention, FPN


__all__ = ["WeightModule", "CurvModule", "CucumberModel"]


# base
class WeightModule(nn.Module):
    def __init__(self, modelpath="resnet50.pth"):
        super(WeightModule, self).__init__()
        model_name = os.path.splitext(os.path.basename(modelpath))[0]
        
        if model_name.startswith("resnet"):
            self.base_model = timm.create_model(
                model_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4)
            )
            state_dict = torch.load(modelpath, weights_only=True)
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            self.base_model.load_state_dict(state_dict)

        elif model_name.startswith("vgg"):
            # 加载权重并处理键名不匹配问题
            self.base_model = timm.create_model(
                model_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3)
            )
            state_dict = torch.load(modelpath, weights_only=True)
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if "classifier" not in k
            }
            new_state_dict = {}
            for k, v in filtered_state_dict.items():
                # 动态处理键名，支持 features 和 classifier 的两种格式
                new_key = k.replace(".", "_")
                new_key = new_key.replace("_weight", ".weight").replace("_bias", ".bias")
                new_state_dict[new_key] = v
            self.base_model.load_state_dict(new_state_dict, strict=False)
        
        # Store the channel numbers of the feature maps
        self.out_channels = [info["num_chs"] for info in self.base_model.feature_info]

    def forward(self, x):
        features = self.base_model(x)
        return features


# base
class CurvModule(nn.Module):
    def __init__(self, modelpath="resnet50.pth"):
        super(CurvModule, self).__init__()
        model_name = os.path.splitext(os.path.basename(modelpath))[0]
        
        if model_name.startswith("resnet"):
            self.base_model = timm.create_model(
                model_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4)
            )
            state_dict = torch.load(modelpath, weights_only=True)
            state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
            self.base_model.load_state_dict(state_dict)

        elif model_name.startswith("vgg"):
            # 加载权重并处理键名不匹配问题
            self.base_model = timm.create_model(
                model_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3)
            )
            state_dict = torch.load(modelpath, weights_only=True)
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if "classifier" not in k
            }
            new_state_dict = {}
            for k, v in filtered_state_dict.items():
                # 动态处理键名，支持 features 和 classifier 的两种格式
                new_key = k.replace(".", "_")
                new_key = new_key.replace("_weight", ".weight").replace("_bias", ".bias")
                new_state_dict[new_key] = v
            self.base_model.load_state_dict(new_state_dict, strict=False)
        
        self.out_channels = [info["num_chs"] for info in self.base_model.feature_info]

    def forward(self, x):
        features = self.base_model(x)
        return features


class FFN(nn.Module):
    def __init__(self, in_channels=256, out_channels=4):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 128)  # Concatenate along channel dimension
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits
    

class FeatureCombineModule(nn.Module):
    def __init__(self, channels, combine_type="lka"):
        super(FeatureCombineModule, self).__init__()
        self.combine_type = combine_type
        self.out_channels = channels

        # 支持混合类型，如conv3_lka1、conv2_lka2、conv1_lka3
        if combine_type == "lka":
            self.attentions = nn.ModuleList([LKA_Attention(ch) for ch in channels])
        elif combine_type == "conv":
            self.attentions = nn.ModuleList([nn.Conv2d(ch, ch * 2, 3, padding=1) for ch in channels])
        elif combine_type.startswith("conv") and "_lka" in combine_type:
            # 解析如conv2_lka2
            m = re.match(r"conv(\d+)_lka(\d+)", combine_type)
            if not m:
                raise ValueError(f"Unknown combine_type: {combine_type}")
            conv_num = int(m.group(1))
            lka_num = int(m.group(2))
            if conv_num + lka_num != len(channels):
                raise ValueError(f"conv_num + lka_num 必须等于通道层数, got {conv_num}+{lka_num}!={len(channels)}")
            self.attentions = nn.ModuleList([
                *(nn.Conv2d(ch, ch * 2, 3, padding=1) for ch in channels[:conv_num]),
                *(LKA_Attention(ch) for ch in channels[conv_num:])
            ])
        else:
            raise ValueError(f"Unknown combine_type: {combine_type}")

    def forward(self, features1, features2):
        combined_features = []
        combined_feature = features1[0] + features2[0]
        combined_feature = self.attentions[0](combined_feature)
        combined_features.append(combined_feature)
        for i in range(1, len(features1)):
            combined_feature = F.interpolate(combined_feature, size=features1[i].shape[-2:], mode="nearest")
            combined_feature = combined_feature + features1[i] + features2[i]
            combined_feature = self.attentions[i](combined_feature)
            combined_features.append(combined_feature)
        return combined_features
    

class CucumberModel(nn.Module):
    def __init__(self, modelpath="resnet50.pth", feature_fusion_type="cross_attention", fpn_ablation_type="curv_only", combine_type="lka"):
        super(CucumberModel, self).__init__()
        
        self.weight_module = WeightModule(modelpath)
        self.curv_module = CurvModule(modelpath)
        
        # 保存模型名称，避免在forward中重复解析
        self.model_name = os.path.splitext(os.path.basename(modelpath))[0]
        
        # 特征融合类型控制
        self.feature_fusion_type = feature_fusion_type  # "cross_attention", "add", "multiply", "concat"
        
        # FPN消融实验类型控制
        self.fpn_ablation_type = fpn_ablation_type  # "none", "both", "curv_only", "weight_only"
        
        # 获取backbone的通道数
        backbone_channels = self.curv_module.out_channels
        
        # 根据模型类型选择不同的特征层
        if self.model_name.startswith("vgg"):
            # VGG模型：使用所有4层特征
            selected_channels = backbone_channels  # [64, 128, 256, 512]
        else:
            # ResNet模型：选择后4层特征（跳过第一层）
            selected_channels = backbone_channels[-4:]  # [256, 512, 1024, 2048]
        
        # 根据FPN消融类型初始化FPN模块
        if self.fpn_ablation_type in ["both", "curv_only", "weight_only"]:
            self.fpn = FPN(in_channels_list=selected_channels)
        
        self.feature_combine = FeatureCombineModule(channels=selected_channels, combine_type=combine_type)
        
        last_ch = selected_channels[-1]  # Use the last channel size
        self.weight_ffn = FFN(in_channels=last_ch*2, out_channels=1)
        self.curv_ffn = FFN(in_channels=last_ch, out_channels=1)

        # 根据特征融合类型初始化不同的模块
        if self.feature_fusion_type == "cross_attention":
            self.cross_attention = CrossAttention(last_ch)
        elif self.feature_fusion_type == "concat":
            # Concatenation + Conv 方案
            self.concat_conv = nn.Conv2d(in_channels=last_ch*2, out_channels=last_ch, kernel_size=1)
        
        self.conv = nn.Conv2d(in_channels=last_ch*2, out_channels=last_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=last_ch, out_channels=last_ch // 2, kernel_size=1)
        self.ffn = FFN(in_channels=last_ch // 2, out_channels=4)

    def forward(self, x):
        # x = self.lraspp(x)

        weight_features = self.weight_module(x)
        curv_features = self.curv_module(x)

        # 根据模型类型选择不同的特征层
        if self.model_name.startswith("vgg"):
            # VGG模型：使用所有4层特征
            weight_selected = weight_features
            curv_selected = curv_features
        else:
            # ResNet模型：选择后4层特征（跳过第一层）
            weight_selected = weight_features[-4:]
            curv_selected = curv_features[-4:]

        # 根据FPN消融类型处理特征
        if self.fpn_ablation_type == "none":
            # 都没有FPN：直接使用原始特征
            weight_fpn_features = weight_selected
            curv_fpn_features = curv_selected
        elif self.fpn_ablation_type == "both":
            # 都有FPN：weight和curv都使用FPN
            weight_fpn_features = self.fpn(weight_selected)
            curv_fpn_features = self.fpn(curv_selected)
        elif self.fpn_ablation_type == "curv_only":
            # Curv有FPN，weight没有：当前方案
            weight_fpn_features = weight_selected
            curv_fpn_features = self.fpn(curv_selected)
        elif self.fpn_ablation_type == "weight_only":
            # Weight有FPN，curv没有：反向方案
            weight_fpn_features = self.fpn(weight_selected)
            curv_fpn_features = curv_selected
        else:
            raise ValueError(f"Unknown fpn_ablation_type: {self.fpn_ablation_type}")
        
        weight_curv_combined_features = self.feature_combine(weight_fpn_features, curv_fpn_features)
        weight_final_feature = F.adaptive_avg_pool2d(weight_curv_combined_features[-1], 1).reshape(x.shape[0], -1)
        weight = self.weight_ffn(weight_final_feature)
         
        curv_final_feature = F.adaptive_avg_pool2d(curv_fpn_features[-1], 1).reshape(x.shape[0], -1)
        curv = self.curv_ffn(curv_final_feature)

        weight_feature = weight_curv_combined_features[-1]
        weight_feature = self.conv(weight_feature)

        curv_feature = curv_fpn_features[-1]

        # 根据特征融合类型选择不同的特征融合方式
        if self.feature_fusion_type == "cross_attention":
            # 原始 CrossAttention 方案
            combined_feature = self.cross_attention(weight_feature, curv_feature)
        elif self.feature_fusion_type == "add":
            # Elementwise Add 方案
            combined_feature = weight_feature + curv_feature
        elif self.feature_fusion_type == "multiply":
            # Elementwise Multiply 方案
            combined_feature = weight_feature * curv_feature
        elif self.feature_fusion_type == "concat":
            # Concatenation + Conv 方案
            combined_feature = torch.cat([weight_feature, curv_feature], dim=1)  # (B, 2C, H, W)
            combined_feature = self.concat_conv(combined_feature)  # (B, C, H, W)
        else:
            raise ValueError(f"Unknown feature_fusion_type: {self.feature_fusion_type}")
        
        combined_feature = self.conv1(combined_feature)
        combined_final_feature = F.adaptive_avg_pool2d(combined_feature, 1).reshape(x.shape[0], -1)
        logits = self.ffn(combined_final_feature)

        return weight, curv, logits

if __name__ == "__main__":
    input_tensor = torch.randn(20, 3, 224, 224)  # Random input data

    print("Testing WeightModule with ResNet50...")
    weight_module = WeightModule(modelpath="model/resnet50.pth")
    features = weight_module(input_tensor)
    for i, feature in enumerate(features):
        print(f"Feature {i+1} Shape: {feature.shape}")
    print("ResNet50 WeightModule Test Completed!\n")

    print("Testing RipeModule with ResNet50...")
    curv_module = CurvModule(modelpath="model/resnet50.pth")
    features = curv_module(input_tensor)
    for i, feature in enumerate(features):
        print(f"Feature {i+1} Shape: {feature.shape}")
    print("ResNet50 RipeModule Test Completed!")