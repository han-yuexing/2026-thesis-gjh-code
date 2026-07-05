# CIDIS模型训练和测试

## 快速开始

### 安装依赖
```bash
cd /root/code/FruVeg
pip install -r requirements.txt
```

### 运行训练
```bash
# 使用默认配置（黄瓜数据集，10次运行，100轮）
python CIDIS/CIDIS_train_test.py

# 训练香蕉数据集
python CIDIS/CIDIS_train_test.py --dataset banana

# 快速测试（3次运行，50轮）
python CIDIS/CIDIS_train_test.py --n_runs 3 --epochs 50
```

### 查看结果
结果保存在 `results/日期-CIDIS-数据集-序号/` 目录下，包含模型、日志、图表和摘要。

---

## 主要特性

- **YAML配置管理**: 使用`config.yaml`管理超参数
- **命令行参数支持**: 可覆盖YAML配置
- **多次运行**: 默认运行10次（用于显著性检验）
- **结构化结果保存**: 自动创建带日期和序号的结果文件夹

## 文件结构

```
CIDIS/
├── CIDIS_train_test.py    # 训练和测试主程序
├── CIDIS_model.py          # 模型定义
├── config.yaml             # 配置文件
├── README.md               # 本文件
└── CHANGELOG.md            # 版本变更记录
```

## 配置说明

### 配置文件 (config.yaml)

```yaml
dataset:
  type: 'cucumber'  # 'banana' 或 'cucumber'

training:
  epochs: 100       # 训练轮数
  batch_size: 20    # 批次大小
  lr: 0.00005       # 学习率
  n_runs: 10        # 运行次数

seeds: [24, 42, 2024, 2025, 3407, 1234, 5678, 9999, 7777, 8888]

save:
  results_dir: 'results'
  create_dated_folder: true
  model_name: 'CIDIS'
```

### 命令行参数

```bash
# 覆盖数据集
python CIDIS_train_test.py --dataset banana

# 覆盖训练参数
python CIDIS_train_test.py --epochs 150 --batch_size 32 --lr 0.0001

# 指定运行次数和随机种子
python CIDIS_train_test.py --n_runs 5 --seeds 42 2024 3407

# 使用自定义配置文件
python CIDIS_train_test.py --config /path/to/config.yaml

# 查看所有参数
python CIDIS_train_test.py --help
```

## 结果保存

### 文件夹命名

格式：`日期-模型-数据集标识-序号`

示例：
- `20251027-CIDIS-c-1` (黄瓜数据集，第1次运行)
- `20251027-CIDIS-b-1` (香蕉数据集，第1次运行)
- `20251027-CIDIS-c-2` (黄瓜数据集，第2次运行)

### 文件夹结构

```
20251027-CIDIS-c-1/
├── config.yaml                          # 本次运行的配置
├── cidis_results_summary_cucumber.txt   # 结果摘要
├── models/                              # 模型文件
│   ├── best_cidis_cucumber_seed24_run1.pth
│   ├── best_cidis_cucumber_seed42_run2.pth
│   └── ...
├── logs/                                # 训练日志
│   └── cidis_training.log
└── figures/                             # 可视化图表
    ├── cidis_run1_seed24_curves_cucumber.png
    ├── confusion_matrix_cucumber_*.png
    ├── cidis_multi_run_results_cucumber.png
    └── ...
```

### 输出内容

1. **训练过程**: 每个epoch的损失、准确率
2. **验证集结果**: 平均准确率 ± 标准差
3. **测试集结果**: 
   - 准确率 ± 标准差
   - 宏平均/加权平均指标
   - 每个类别的详细指标
   - 混淆矩阵
4. **可视化图表**:
   - 训练曲线（每次运行）
   - 多次运行结果对比
   - 混淆矩阵（每次运行）

## 显著性检验

程序提供显著性检验功能，用于比较不同模型：

```python
from CIDIS_train_test import perform_significance_test

results1 = [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.85, 0.84, 0.86, 0.85]
results2 = [0.82, 0.83, 0.81, 0.84, 0.82, 0.83, 0.82, 0.81, 0.83, 0.82]

perform_significance_test(results1, results2, metric_name="Accuracy")
```

## 常见问题

### GPU内存不足
```bash
python CIDIS_train_test.py --batch_size 8
```

### 训练时间过长
```bash
python CIDIS_train_test.py --n_runs 3 --epochs 50
```

### 查看详细日志
```bash
cat results/日期-CIDIS-数据集-序号/logs/cidis_training.log
```

## 注意事项

1. 确保随机种子数量 ≥ 运行次数
2. GPU内存不足时减小 `batch_size`
3. 每次运行自动创建新文件夹，不会覆盖之前的结果
4. 建议定期清理旧的结果文件夹以节省空间

## 更多信息

- 版本变更记录: 参见 [CHANGELOG.md](CHANGELOG.md)
- 依赖列表: 参见项目根目录的 `requirements.txt`
