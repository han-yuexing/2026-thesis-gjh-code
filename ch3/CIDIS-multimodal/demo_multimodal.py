"""
多模态CIDIS模型演示脚本
演示如何使用图片+重量+形状特征进行水果/蔬菜分级预测
"""

import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CIDIS_model_multimodal import CIDIS_Model_Multimodal, CIDIS_Model_Multimodal_Simple

def load_model(model_path, num_classes=4, num_shape_features=2, use_simple=False, device='cpu'):
    """
    加载训练好的多模态模型
    
    参数:
        model_path: 模型权重文件路径
        num_classes: 分类类别数
        num_shape_features: 形状特征数量 (黄瓜=2, 香蕉=1)
        use_simple: 是否使用简化版模型
        device: 设备 ('cpu' 或 'cuda')
    """
    if use_simple:
        model = CIDIS_Model_Multimodal_Simple(num_classes=num_classes, 
                                              num_shape_features=num_shape_features)
    else:
        model = CIDIS_Model_Multimodal(num_classes=num_classes, 
                                       num_shape_features=num_shape_features)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功: {model_path}")
    return model

def preprocess_image(image_path):
    """
    预处理图像
    
    参数:
        image_path: 图像文件路径
    
    返回:
        处理后的图像张量 (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor

def predict(model, image_tensor, weight, shape_features, class_names, device='cpu'):
    """
    使用模型进行预测
    
    参数:
        model: 训练好的模型
        image_tensor: 图像张量 (1, 3, 224, 224)
        weight: 重量值
        shape_features: 形状特征 (list或tensor)
        class_names: 类别名称列表
        device: 设备
    
    返回:
        预测的类别和置信度
    """
    # 准备输入
    image_tensor = image_tensor.to(device)
    
    # 将weight转换为tensor
    if not isinstance(weight, torch.Tensor):
        weight_tensor = torch.tensor([[weight]], dtype=torch.float32).to(device)
    else:
        weight_tensor = weight.to(device)
    
    # 将shape_features转换为tensor
    if not isinstance(shape_features, torch.Tensor):
        shape_tensor = torch.tensor([shape_features], dtype=torch.float32).to(device)
    else:
        shape_tensor = shape_features.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor, weight_tensor, shape_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item()
    
    return predicted_class, confidence_value, probabilities[0].cpu().numpy()


def demo_cucumber():
    """黄瓜分级预测示例"""
    print("\n" + "="*80)
    print("🥒 黄瓜分级预测示例")
    print("="*80)
    
    # 定义类别
    class_names = ['super', 'first', 'second', 'third']
    
    # 模拟输入数据 (实际使用时替换为真实数据)
    print("\n📊 输入数据:")
    print("  - 图像: [模拟图像张量]")
    print("  - 重量: 150.5 克")
    print("  - 弯曲度 (arch): 12.3 度")
    
    # 创建模拟输入
    # 实际使用时: image_tensor = preprocess_image("path/to/cucumber.jpg")
    image_tensor = torch.randn(1, 3, 224, 224)
    weight = 150.5
    shape_features = [12.3]  # arch弯曲度
    
    # 创建模型 (实际使用时应加载训练好的权重)
    print("\n🔧 初始化模型...")
    model = CIDIS_Model_Multimodal(num_classes=4, num_shape_features=1)
    model.eval()  # 设置为评估模式
    
    # 进行预测
    print("\n🔮 开始预测...")
    predicted_class, confidence, all_probs = predict(
        model, image_tensor, weight, shape_features, class_names, device='cpu'
    )
    
    # 显示结果
    print("\n" + "="*80)
    print("📋 预测结果:")
    print("="*80)
    print(f"  🏆 预测等级: {predicted_class.upper()}")
    print(f"  📊 置信度: {confidence*100:.2f}%")
    print(f"\n  各等级概率分布:")
    for i, class_name in enumerate(class_names):
        prob = all_probs[i] * 100
        bar = "█" * int(prob / 2)
        print(f"    {class_name:8s}: {prob:5.2f}% {bar}")
    print("="*80)


def demo_banana():
    """香蕉分级预测示例"""
    print("\n" + "="*80)
    print("🍌 香蕉分级预测示例")
    print("="*80)
    
    # 定义类别
    class_names = ['super', 'first', 'second', 'third']
    
    # 模拟输入数据
    print("\n📊 输入数据:")
    print("  - 图像: [模拟图像张量]")
    print("  - 重量: 185.3 克")
    print("  - 成熟度 (ripeness): 0.75")
    
    # 创建模拟输入
    image_tensor = torch.randn(1, 3, 224, 224)
    weight = 185.3
    shape_features = [0.75]  # [ripeness]
    
    # 创建模型 (简化版)
    print("\n🔧 初始化模型 (简化版)...")
    model = CIDIS_Model_Multimodal_Simple(num_classes=4, num_shape_features=1)
    model.eval()  # 设置为评估模式
    
    # 进行预测
    print("\n🔮 开始预测...")
    predicted_class, confidence, all_probs = predict(
        model, image_tensor, weight, shape_features, class_names, device='cpu'
    )
    
    # 显示结果
    print("\n" + "="*80)
    print("📋 预测结果:")
    print("="*80)
    print(f"  🏆 预测等级: {predicted_class.upper()}")
    print(f"  📊 置信度: {confidence*100:.2f}%")
    print(f"\n  各等级概率分布:")
    for i, class_name in enumerate(class_names):
        prob = all_probs[i] * 100
        bar = "█" * int(prob / 2)
        print(f"    {class_name:8s}: {prob:5.2f}% {bar}")
    print("="*80)


def demo_batch_prediction():
    """批量预测示例"""
    print("\n" + "="*80)
    print("📦 批量预测示例 (黄瓜)")
    print("="*80)
    
    class_names = ['super', 'first', 'second', 'third']
    
    # 模拟一批数据
    batch_size = 5
    print(f"\n📊 批量大小: {batch_size}")
    
    images = torch.randn(batch_size, 3, 224, 224)
    weights = torch.tensor([[120.5], [150.3], [180.2], [95.7], [160.8]], dtype=torch.float32)
    shape_features = torch.tensor([
        [10.5],
        [12.3],
        [15.8],
        [8.2],
        [13.1]
    ], dtype=torch.float32)  # arch弯曲度
    
    # 创建模型
    print("\n🔧 初始化模型...")
    model = CIDIS_Model_Multimodal(num_classes=4, num_shape_features=1)
    model.eval()
    
    # 批量预测
    print("\n🔮 开始批量预测...")
    with torch.no_grad():
        outputs = model(images, weights, shape_features)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
    
    # 显示结果
    print("\n" + "="*80)
    print("📋 批量预测结果:")
    print("="*80)
    print(f"{'序号':<6} {'重量(g)':<10} {'弯曲度':<10} {'预测等级':<12} {'置信度':<10}")
    print("-" * 80)
    for i in range(batch_size):
        weight_val = weights[i, 0].item()
        arch_val = shape_features[i, 0].item()
        pred_class = class_names[predicted[i].item()]
        conf_val = confidences[i].item() * 100
        print(f"{i+1:<6} {weight_val:<10.2f} {arch_val:<10.2f} {pred_class.upper():<12} {conf_val:<10.2f}%")
    print("="*80)


def demo_with_real_image():
    """
    使用真实图像进行预测的示例
    (需要提供实际的图像路径和模型权重路径)
    """
    print("\n" + "="*80)
    print("📷 使用真实图像预测示例")
    print("="*80)
    
    # TODO: 替换为实际路径
    image_path = "path/to/your/cucumber_image.jpg"
    model_path = "path/to/your/best_model.pth"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"\n⚠️  图像文件不存在: {image_path}")
        print("💡 请替换为实际的图像路径")
        return
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  模型文件不存在: {model_path}")
        print("💡 请先训练模型或提供训练好的模型路径")
        return
    
    # 加载模型
    model = load_model(model_path, num_classes=4, num_shape_features=1, 
                      use_simple=False, device='cpu')
    
    # 预处理图像
    image_tensor = preprocess_image(image_path)
    
    # 实际的重量和形状特征 (从测量或CSV文件读取)
    weight = 150.5
    shape_features = [12.3]  # arch弯曲度
    
    # 预测
    class_names = ['super', 'first', 'second', 'third']
    predicted_class, confidence, all_probs = predict(
        model, image_tensor, weight, shape_features, class_names, device='cpu'
    )
    
    # 显示结果
    print(f"\n✅ 图像: {image_path}")
    print(f"🏆 预测等级: {predicted_class.upper()}")
    print(f"📊 置信度: {confidence*100:.2f}%")


if __name__ == "__main__":
    print("="*80)
    print("🚀 多模态CIDIS模型演示程序")
    print("="*80)
    print("\n本演示展示如何使用多模态CIDIS模型进行水果/蔬菜分级预测")
    print("输入: 图像 + 重量 + 形状特征")
    print("输出: 分级结果 (super/first/second/third)")
    
    # 运行各种演示
    demo_cucumber()
    demo_banana()
    demo_batch_prediction()
    
    print("\n" + "="*80)
    print("💡 提示:")
    print("="*80)
    print("1. 要使用真实图像预测，请修改 demo_with_real_image() 中的路径")
    print("2. 先运行 CIDIS_train_test_multimodal.py 训练模型")
    print("3. 然后使用训练好的模型权重进行预测")
    print("4. 参考 README_MULTIMODAL.md 获取更多信息")
    print("="*80)

