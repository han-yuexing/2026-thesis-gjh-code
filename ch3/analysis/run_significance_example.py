#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grade分类显著性分析示例脚本

使用方法:
1. 修改下面的 model_results 字典，添加你要比较的模型结果文件路径
2. 运行: python analysis/run_significance_example.py
"""

from analysis.significance import compare_models
import os

def main():
    # ============================================
    # 配置：修改这里添加你要比较的模型
    # ============================================
    
    # 定义模型结果文件路径
    # 格式: {'模型名称': '结果文件路径'}
    model_results = {
        'CIDIS': 'results/20251203-CIDIS-Test-b-1/cidis_results_summary_banana_combined.txt',
        'CIDIS-Multi': 'results/20251203-CIDIS-Multi-Test-b-1/cidis_multimodal_results_summary_banana_combined.txt',
        'FruitVision-非冻结': 'results/20251203-FruitVision-Test-非冻结-b-1/fruit_vision_results_summary_banana_combined_非冻结.txt',
        'FruitVision-冻结': 'results/20251203-FruitVision-Test-冻结-b-1/fruit_vision_results_summary_banana_combined_冻结.txt',
        'FruitVision-Multi-非冻结': 'results/20251203-FruitVision-Multi-Test-非冻结-b-1/fruitvision_multimodal_results_summary_banana_combined_非冻结.txt',
        'FruitVision-Multi-冻结': 'results/20251203-FruitVision-Multi-Test-冻结-b-1/fruitvision_multimodal_results_summary_banana_combined_冻结.txt',
        
        # ============================================
        # 在这里添加你自己的模型结果文件路径
        # ============================================
        # 'YourModel': 'results/your_model_path/your_results.txt',
        # 'YourModel-Multi': 'results/your_model_path/your_multimodal_results.txt',
    }
    
    # 检查文件是否存在
    print("检查结果文件...")
    missing_files = []
    for model_name, file_path in model_results.items():
        if not os.path.exists(file_path):
            print(f"警告: {model_name} 的结果文件不存在: {file_path}")
            missing_files.append(model_name)
    
    # 移除不存在的文件
    for model_name in missing_files:
        model_results.pop(model_name)
    
    if not model_results:
        print("错误: 没有找到任何有效的结果文件！")
        print("\n请确保:")
        print("1. 结果文件路径正确")
        print("2. 文件格式符合要求（见 significance_README.md）")
        return
    
    print(f"\n将比较以下 {len(model_results)} 个模型:")
    for model_name in model_results.keys():
        print(f"  - {model_name}")
    
    # ============================================
    # 执行比较分析
    # ============================================
    output_dir = 'significance_analysis'
    alpha = 0.05  # 显著性水平
    
    print(f"\n开始显著性分析...")
    print(f"输出目录: {output_dir}")
    print(f"显著性水平: α = {alpha}")
    print("="*80)
    
    try:
        comparisons = compare_models(
            model_results_dict=model_results,
            output_dir=output_dir,
            alpha=alpha
        )
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print(f"\n结果已保存到: {os.path.abspath(output_dir)}")
        print("\n生成的文件:")
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                print(f"  - {file}")
        
    except Exception as e:
        print(f"\n错误: 分析过程中出现异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

