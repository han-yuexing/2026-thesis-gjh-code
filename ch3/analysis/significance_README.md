# Grade分类显著性分析使用说明

## 功能概述

`significance.py` 脚本用于比较不同模型在grade（等级分类）任务上的性能，并进行统计显著性检验。

## 支持的指标

脚本会分析以下grade分类指标：
- **Accuracy** (准确率)
- **Macro Precision** (宏平均精确率)
- **Macro Recall** (宏平均召回率)
- **Macro F1-Score** (宏平均F1分数)
- **Weighted Precision** (加权平均精确率)
- **Weighted Recall** (加权平均召回率)
- **Weighted F1-Score** (加权平均F1分数)

## 统计检验方法

脚本会根据数据情况自动选择检验方法：

1. **配对检验**（当两组数据长度相同时）：
   - 配对t检验 (Paired t-test)
   - Wilcoxon符号秩检验 (Wilcoxon signed-rank test)

2. **独立样本检验**（当两组数据长度不同时）：
   - 独立样本t检验 (Independent t-test)
   - Mann-Whitney U检验 (Mann-Whitney U test)

3. **效应量**：
   - Cohen's d（用于评估差异的实际意义）

## 使用方法

### 基本用法

```python
from analysis.significance import compare_models

# 定义模型结果文件路径
model_results = {
    'CIDIS': 'results/20251203-CIDIS-Test-b-1/cidis_results_summary_banana_combined.txt',
    'FruitVision-非冻结': 'results/20251203-FruitVision-Test-非冻结-b-1/fruit_vision_results_summary_banana_combined_非冻结.txt',
    'FruitVision-冻结': 'results/20251203-FruitVision-Test-冻结-b-1/fruit_vision_results_summary_banana_combined_冻结.txt',
    'YourModel': 'results/your_model_results.txt',  # 添加你的模型结果
}

# 执行比较分析
comparisons = compare_models(
    model_results_dict=model_results,
    output_dir='significance_analysis',  # 输出目录
    alpha=0.05  # 显著性水平
)
```

### 直接运行脚本

修改 `significance.py` 文件末尾的 `model_results` 字典，然后运行：

```bash
cd /root/code/FruVeg
python analysis/significance.py
```

## 结果文件格式要求

结果文件应该是文本格式，包含以下信息：

```
运行 1 (种子: 24):
  测试准确率: 0.3900
  宏平均Precision: 0.2164
  宏平均Recall: 0.3773
  宏平均F1-Score: 0.2635
  加权平均Precision: 0.2182
  加权平均Recall: 0.3900
  加权平均F1-Score: 0.2681

运行 2 (种子: 42):
  ...
```

或者包含汇总行：

```
各次运行: 0.3900, 0.5000, 0.4700, ...
```

## 输出结果

脚本会在指定的输出目录中生成以下文件：

1. **significance_test_results.csv**: 详细的统计检验结果表格
2. **significance_analysis_report.txt**: 文本格式的分析报告
3. **可视化图表**:
   - `accuracy_comparison.png`: 准确率比较图
   - `macro_precision_comparison.png`: 宏平均精确率比较图
   - `macro_recall_comparison.png`: 宏平均召回率比较图
   - `macro_f1_comparison.png`: 宏平均F1分数比较图
   - `weighted_precision_comparison.png`: 加权平均精确率比较图
   - `weighted_recall_comparison.png`: 加权平均召回率比较图
   - `weighted_f1_comparison.png`: 加权平均F1分数比较图
   - `all_metrics_comparison.png`: 所有指标的综合比较图

## 结果解读

### 显著性判断

- **p < 0.05**: 差异显著，拒绝零假设（两组性能有显著差异）
- **p >= 0.05**: 差异不显著，不能拒绝零假设（两组性能差异不显著）

### 效应量 (Cohen's d)

- **|d| < 0.2**: 效应量小
- **0.2 ≤ |d| < 0.5**: 效应量中等
- **0.5 ≤ |d| < 0.8**: 效应量大
- **|d| ≥ 0.8**: 效应量非常大

## 示例：比较你的模型与基线模型

假设你的模型结果文件在 `results/YourModel/your_results.txt`，可以这样比较：

```python
from analysis.significance import compare_models

model_results = {
    'YourModel': 'results/YourModel/your_results.txt',
    'CIDIS': 'results/20251203-CIDIS-Test-b-1/cidis_results_summary_banana_combined.txt',
    'FruitVision': 'results/20251203-FruitVision-Test-非冻结-b-1/fruit_vision_results_summary_banana_combined_非冻结.txt',
}

comparisons = compare_models(
    model_results_dict=model_results,
    output_dir='my_model_significance_analysis',
    alpha=0.05
)
```

## 注意事项

1. **数据配对**: 如果多次运行使用了相同的随机种子，建议使用配对检验（脚本会自动检测）
2. **样本量**: 建议至少10次运行以获得可靠的统计结果
3. **显著性水平**: 默认使用α=0.05，可以根据需要调整
4. **文件路径**: 确保结果文件路径正确，且文件格式符合要求

## 常见问题

**Q: 如果我的结果文件格式不同怎么办？**

A: 可以修改 `parse_results_file()` 函数中的正则表达式来适配你的文件格式，或者直接提供结果列表：

```python
# 直接提供结果列表
model_results = {
    'YourModel': [
        {'accuracy': 0.85, 'macro_precision': 0.82, ...},
        {'accuracy': 0.87, 'macro_precision': 0.84, ...},
        # ... 更多运行结果
    ]
}
```

**Q: 如何只比较特定指标？**

A: 修改脚本中的 `metrics` 列表：

```python
# 只比较准确率和F1分数
metrics = ['accuracy', 'macro_f1', 'weighted_f1']
```

**Q: 如何调整显著性水平？**

A: 在调用 `compare_models()` 时设置 `alpha` 参数：

```python
comparisons = compare_models(
    model_results_dict=model_results,
    output_dir='output',
    alpha=0.01  # 使用更严格的显著性水平
)
```

