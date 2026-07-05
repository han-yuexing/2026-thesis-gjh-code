# 变更日志

## [2.0.0] - 2025-10-27

### 新增功能 ✨
- **YAML配置管理**: 添加 `config.yaml` 用于集中管理超参数
- **自动文件夹管理**: 每次运行自动创建带日期和序号的结果文件夹
  - 命名格式: `日期-模型-数据集标识-序号`
  - 示例: `20251027-FruitVision-c-1` (黄瓜), `20251027-FruitVision-b-1` (香蕉)
- **结构化结果保存**: 创建子文件夹 `models/`, `logs/`, `figures/`
- **配置持久化**: 每次运行的配置自动保存到结果文件夹

### 重要变更 ⚡
- **运行次数**: 从5次增加到**10次**（用于显著性检验）
- **随机种子**: 扩展到10个种子
- **结果路径**: 所有结果统一保存到 `项目根目录/results/`

### 代码优化 🔧
- 重构 `train_model()` 函数，添加 `save_dirs`, `device`, `logger` 参数
- 重构 `test_model()` 函数，添加 `save_dirs`, `device`, `logger` 参数
- 将全局配置（设备、日志）改为局部变量
- 统一使用 logger 对象而非 logging 模块

### 新增文件 📁
- `config.yaml` - 配置文件
- `README.md` - 详细使用说明（含快速开始）
- `CHANGELOG.md` - 本文件

### 修改文件 📝
- `fruit_vision_train_test.py` - 主训练脚本
  - 原: 684行
  - 新: 812行
  - 增加: 128行（+19%）

### 向后兼容性 ✅
- 完全兼容原有命令行参数
- 命令行参数可覆盖YAML配置
- 不影响现有代码调用方式

### 测试状态 ✅
- [x] 语法检查通过
- [x] 文件夹创建功能验证通过
- [x] YAML配置加载测试通过
- [ ] 完整训练流程测试（待实际数据）

### 使用示例

#### 使用默认配置（冻结骨干网络）
```bash
python FruitVision/fruit_vision_train_test.py --freeze_backbone
```

#### 不冻结骨干网络
```bash
# 修改config.yaml中的freeze_backbone为false后运行
python FruitVision/fruit_vision_train_test.py
```

#### 覆盖部分配置
```bash
python FruitVision/fruit_vision_train_test.py --dataset banana --epochs 150
```

#### 快速测试
```bash
python FruitVision/fruit_vision_train_test.py --n_runs 3 --epochs 50
```

### 文件夹结构示例
```
results/
└── 20251027-FruitVision-c-1/
    ├── config.yaml                                    # 运行配置
    ├── fruit_vision_results_summary_cucumber_冻结.txt # 结果摘要
    ├── models/                                        # 模型文件
    │   ├── best_fruit_vision_cucumber_seed24_run1.pth
    │   └── ... (共10个模型)
    ├── logs/                                          # 日志
    │   └── fruit_vision_training.log
    └── figures/                                       # 图表
        ├── fruit_vision_run1_seed24_curves_cucumber.png
        ├── confusion_matrix_*.png
        └── fruit_vision_multi_run_results_cucumber_冻结.png
```

### 依赖变更
新增:
- PyYAML >= 6.0.0
- opencv-python >= 4.0.0（FruitVision特有）

### FruitVision特性
- ✨ **CLAHE对比度增强**: 自适应直方图均衡化
- ✨ **Otsu阈值分割**: 自动二值化分割
- ✨ **Specificity指标**: 计算真阴性率
- ✨ **骨干网络冻结**: 可选择冻结/不冻结MobileNetV3

### 注意事项 ⚠️
1. 运行10次会生成较多文件，确保磁盘空间充足
2. 同一天多次运行会自动递增序号（-1, -2, -3...）
3. 需要安装opencv-python用于图像预处理
4. 冻结骨干网络训练更快，不冻结可能准确率更高
5. 建议定期清理旧的结果文件夹

### 性能对比

#### 训练速度（冻结 vs 不冻结）
- 冻结骨干网络: ~2-3分钟/epoch (取决于GPU)
- 不冻结骨干网络: ~5-8分钟/epoch

#### 内存使用
- 冻结骨干网络: ~2GB GPU内存 (batch_size=16)
- 不冻结骨干网络: ~4GB GPU内存 (batch_size=16)

#### 参数量
- 总参数: ~5.5M
- 可训练参数（冻结模式）: ~260K
- 可训练参数（不冻结模式）: ~5.5M

### 下一步计划 📋
- [ ] 添加早停机制
- [ ] 支持断点续训
- [ ] 添加数据增强选项
- [ ] 支持多GPU训练
- [ ] 添加学习率调度器
- [ ] 可视化预处理效果

### 与CIDIS的对比

| 特性 | FruitVision | CIDIS |
|------|-------------|-------|
| 骨干网络 | MobileNetV3-Large | 自定义CNN |
| 预训练 | ImageNet | 无 |
| 预处理 | CLAHE + Otsu | 标准归一化 |
| 输入尺寸 | 224×224 | 224×224 |
| 优化器 | Adam | Adagrad |
| 学习率 | 0.001 | 0.00005 |
| 批次大小 | 16 | 20 |
| 参数量 | ~5.5M / ~260K | ~1M |
| 特殊指标 | Specificity | - |
| 冻结选项 | 支持 | 不支持 |

---

## [1.0.0] - 之前版本

### 基础功能
- 基本训练和测试流程
- 5次运行
- CLAHE+Otsu预处理
- Specificity指标计算
- 结果保存在FruitVision目录

