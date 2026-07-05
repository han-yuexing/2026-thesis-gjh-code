# FruVeg 水果蔬菜表型分级项目

本项目是一个基于 PyTorch 的水果/蔬菜图像分级与表型预测实验代码库，主要围绕香蕉和黄瓜数据集开展如下任务：

- 根据图像预测等级：`super`、`first`、`second`、`third`
- 根据图像回归表型指标，例如重量、香蕉成熟度、黄瓜弯曲/拱度
- 对比多种模型结构：自定义多任务模型、共享骨干模型、CIDIS、FruitVision、单任务基线、多模态模型
- 生成训练日志、模型权重、混淆矩阵、ROC/AUC、R2 散点图、显著性检验和 K 折汇总报告

## 项目结构

```text
.
├── data/                         # 数据集、CSV标注文件、Dataset封装
├── model/                        # 主要模型定义和预训练权重
├── model-SharedBackbone/         # 共享骨干版本的模型定义和权重
├── train_test_banana/            # 香蕉多任务模型训练、测试、K折配置
├── train_test_banana_shared/     # 香蕉共享骨干版本训练和测试
├── train_test_cucumber_shared/   # 黄瓜共享骨干版本训练和测试
├── train_test_separate/          # 水果/蔬菜二分类模型训练和测试
├── FruitVision/                  # FruitVision单模态基线
├── FruitVision-multimodal/       # FruitVision多模态版本
├── CIDIS/                        # CIDIS单模态基线
├── CIDIS-multimodal/             # CIDIS多模态版本
├── test_banana/                  # 香蕉扩展测试集相关评估脚本
├── analysis/                     # 结果分析、显著性检验、K折汇总
├── distribution/                 # 香蕉新增数据分布分析
├── relations/                    # 表型相关性分析和绘图
├── banana_train_single.py        # 香蕉单任务基线训练脚本
├── cucumber_train_single.py      # 黄瓜单任务基线训练脚本
├── plot_detailed_comparison_trait.py
└── requirements.txt              # 基础依赖列表
```

## 环境安装

建议使用 Python 3.9 或 3.10，并在虚拟环境中运行。

```bash
cd /Users/harryge/Desktop/备份/fruveg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前代码中还直接使用了 `pandas`、`timm`、`seaborn`、`torchinfo`、`opencv-python` 等包。若运行时报缺少模块，建议补充安装：

```bash
pip install pandas timm seaborn torchinfo opencv-python
```

如果使用 GPU，请根据本机 CUDA 版本安装匹配的 PyTorch；如果是 macOS 或 CPU 环境，代码会自动退回 CPU，但训练速度会明显变慢。

## 数据说明

数据位于 `data/` 目录，主要由图片和 CSV 标注文件组成。

### 香蕉数据

```text
data/banana/
├── train/
│   ├── *.jpg
│   └── train.csv
├── test/
│   ├── *.jpg
│   └── test.csv
└── banana_level.csv
```

`data/banana/train/train.csv` 和 `data/banana/test/test.csv` 的字段示例：

```csv
image,weight,pix,black,ripeness,level
0.jpg,152.1,1487345,164902,0.1109,second
```

字段含义：

- `image`：图片文件名
- `weight`：重量，回归目标之一
- `pix`、`black`：图像统计特征
- `ripeness`：成熟度，香蕉表型回归目标
- `level`：等级分类标签

`data/banana_add_70/` 是额外香蕉数据，包含 `banana_add_70.csv`，常用于与原始香蕉测试集合并评估。

### 黄瓜数据

```text
data/cucumber/
├── train/
│   ├── *.jpg
│   └── train.csv
└── test/
    ├── *.jpg
    └── test.csv
```

CSV 字段示例：

```csv
filename,weight,ij_curvature,curvature,ij_arch,arch,level
1_0.jpg,131,3.79,3.774,6.518,6.079,second
```

字段含义：

- `filename`：图片文件名
- `weight`：重量，回归目标之一
- `curvature`：弯曲度
- `arch`：拱度/形状特征，部分脚本将其作为 shape trait
- `level`：等级分类标签

### 混合数据

`data/mixed/` 合并了香蕉和黄瓜样本，用于水果/蔬菜二分类任务。CSV 主要字段为：

```csv
filename,weight,appearance,category,level
```

其中 `category` 为 `cucumber` 或 `banana`。

### Dataset 封装

`data/dataset.py` 提供了四个 PyTorch Dataset：

- `CucumberDataset`：读取黄瓜数据，返回 `image, weight, curvature, arch, level`
- `BananaDataset`：读取香蕉数据，返回 `image, weight, ripeness, level`
- `BananaCombinedDataset`：合并原始香蕉测试集和 `banana_add_70` 扩展数据
- `MixedDataset`：读取混合数据，返回 `image, weight, appearance, category, level`

标签映射统一为：

```python
{'super': 0, 'first': 1, 'second': 2, 'third': 3}
```

## 模型与代码说明

### `model/`

`model/modules.py`

- `LR_ASPP`：轻量级 ASPP 模块，用于提取多尺度上下文
- `FPN`：特征金字塔网络，用于融合多层特征
- `CrossAttention`：跨特征注意力模块
- `LKA_Attention`：Large Kernel Attention 模块，用于空间注意力增强

`model/banana.py`

- 定义 `BananaModel`
- 使用两个特征分支分别面向重量和成熟度
- 支持 `resnet`、`vgg` 等 `timm` 骨干网络
- 输出三个结果：`weight, ripeness, logits`
- 支持不同融合策略：
  - `cross_attention`
  - `add`
  - `multiply`
  - `concat`
- 支持 FPN 消融：
  - `none`
  - `both`
  - `ripe_only`
  - `weight_only`

`model/cucumber.py`

- 定义 `CucumberModel`
- 结构与 `BananaModel` 类似
- 输出三个结果：`weight, curv, logits`
- 将黄瓜形状/弯曲相关特征作为第二个表型分支
- 支持 FPN 消融：
  - `none`
  - `both`
  - `curv_only`
  - `weight_only`

`model/simple_model.py`

- 定义 `SimpleModel`
- 单任务基线模型
- `switch=1`：预测重量
- `switch=2`：预测形状特征，香蕉为 `ripeness`，黄瓜脚本中为 `arch`
- `switch=3`：预测等级分类

`model/separate.py`

- 定义 `SeparateModule`
- 基于 ResNet18 的二分类模型，用于区分 `banana` 和 `cucumber`

`model/pretrained/`

- 保存本地预训练权重，例如 `resnet18.pth`、`resnet50.pth`、`vgg16.pth`、`vit_b_16.pth`
- 多数训练脚本默认从该目录加载权重

### `model-SharedBackbone/`

该目录包含共享骨干版本的 `banana.py`、`cucumber.py`、`modules.py` 和预训练权重。用途与 `model/` 类似，但训练脚本位于：

- `train_test_banana_shared/`
- `train_test_cucumber_shared/`

### `train_test_banana/`

香蕉多任务模型训练和测试。

主要文件：

- `config.yaml`：多次运行配置，默认 10 次运行
- `config_kfold.yaml`：K 折交叉验证配置，默认 5 折
- `train_banana.py`：K 折训练入口
- `train_banana_once.py`：多次独立训练入口
- `test_banana.py`：加载指定权重进行测试并输出图表

常用命令：

```bash
# K折训练
python train_test_banana/train_banana.py --config train_test_banana/config_kfold.yaml

# 快速调试
python train_test_banana/train_banana.py --epochs 2 --n_folds 2

# 多次独立训练
python train_test_banana/train_banana_once.py --epochs 50 --n_runs 3

# 测试指定模型
python train_test_banana/test_banana.py \
  --model_path results/你的结果目录/models/fold1/best_epochxx.pth \
  --fold 1
```

重要参数：

- `--pretrained`：预训练权重路径
- `--img_dir`：数据根目录，默认 `data/banana`
- `--feature_fusion_type`：特征融合方式
- `--fpn_ablation_type`：FPN 消融方式
- `--combine_type`：特征合并模块类型
- `--loss_type`：`multi_task`、`huber` 或 `mse`
- `--gamma`：等级分类损失权重

### `train_test_banana_shared/`

共享骨干版本的香蕉训练和测试脚本。文件和参数与 `train_test_banana/` 基本一致，适用于共享 backbone 实验。

常用命令：

```bash
python train_test_banana_shared/train_banana.py
python train_test_banana_shared/test_banana.py --test_all_folds --result_dir results/你的K折结果目录
```

### `train_test_cucumber_shared/`

黄瓜共享骨干版本训练和测试。

主要文件：

- `config_kfold.yaml`
- `train.py`
- `test.py`

常用命令：

```bash
python train_test_cucumber_shared/train.py
python train_test_cucumber_shared/test.py --test_all_folds --result_dir results/你的K折结果目录
```

### `train_test_separate/`

水果/蔬菜二分类模型训练和测试。

主要文件：

- `config.yaml`
- `train_separate.py`
- `test_separate.py`

常用命令：

```bash
python train_test_separate/train_separate.py
python train_test_separate/test_separate.py --model_path results/你的结果目录/models/best_*.pth
```

### 单任务基线脚本

`banana_train_single.py`

- 使用 `SimpleModel` 依次训练三个单任务：
  - 重量回归
  - 成熟度回归
  - 等级分类
- 默认模型名为 `vgg16`
- 输出保存到 `save/{model_name}/simple_train_*`

运行：

```bash
python banana_train_single.py
```

`cucumber_train_single.py`

- 使用 `SimpleModel` 依次训练：
  - 重量回归
  - 形状/拱度回归
  - 等级分类
- 默认模型名为 `vit_b_16`

运行：

```bash
python cucumber_train_single.py
```

注意：这两个脚本的关键参数目前写在 `if __name__ == "__main__"` 中，如需切换模型、epoch 或学习率，需要修改脚本内变量。

## CIDIS 相关代码

### `CIDIS/`

CIDIS 是一个单模态图像分类基线，输入图片，输出四等级分类。

主要文件：

- `CIDIS_model.py`：复现论文中的 CIDIS CNN 结构
- `CIDIS_train_test.py`：训练、测试、多次运行、结果保存
- `config.yaml`：默认配置
- `README.md`：CIDIS 子模块详细说明
- `CHANGELOG.md`：变更记录

常用命令：

```bash
# 默认黄瓜
python CIDIS/CIDIS_train_test.py

# 香蕉
python CIDIS/CIDIS_train_test.py --dataset banana

# 快速调试
python CIDIS/CIDIS_train_test.py --dataset cucumber --epochs 2 --n_runs 1
```

### `CIDIS-multimodal/`

CIDIS 多模态版本，输入为图像 + 重量 + 形状特征，输出等级分类。

主要文件：

- `CIDIS_model_multimodal.py`：完整版和简化版多模态模型
- `CIDIS_train_test_multimodal.py`：训练测试入口
- `demo_multimodal.py`：模型调用演示
- `config_multimodal.yaml`：配置
- `README_MULTIMODAL.md`：多模态详细说明

常用命令：

```bash
python CIDIS-multimodal/CIDIS_train_test_multimodal.py --dataset cucumber --simple --n_runs 1
python CIDIS-multimodal/CIDIS_train_test_multimodal.py --dataset banana --epochs 50
```

## FruitVision 相关代码

### `FruitVision/`

FruitVision 是 MobileNetV3-Large 单模态图像分类基线。

主要文件：

- `fruit_vision.py`：模型定义
- `fruit_vision_train_test.py`：训练、测试、多次运行、结果保存
- `config.yaml`：默认配置
- `README.md`：FruitVision 子模块详细说明
- `CHANGELOG.md`：变更记录

该模型使用 CLAHE 和 Otsu 预处理，因此需要 `opencv-python`。

常用命令：

```bash
# 默认黄瓜
python FruitVision/fruit_vision_train_test.py

# 香蕉
python FruitVision/fruit_vision_train_test.py --dataset banana

# 快速调试
python FruitVision/fruit_vision_train_test.py --dataset cucumber --epochs 2 --n_runs 1

# 仅测试指定权重
python FruitVision/fruit_vision_train_test.py --test_only path/to/model.pth --dataset cucumber
```

### `FruitVision-multimodal/`

FruitVision 多模态版本，输入为图像 + 重量 + 形状特征。

主要文件：

- `fruit_vision_multimodal.py`：完整版和简化版多模态模型
- `fruit_vision_train_test_multimodal.py`：训练测试入口
- `config_multimodal.yaml`：配置
- `README_MULTIMODAL.md`：多模态详细说明
- `CHANGELOG.md`：变更记录

常用命令：

```bash
python FruitVision-multimodal/fruit_vision_train_test_multimodal.py --dataset cucumber --simple --n_runs 1
python FruitVision-multimodal/fruit_vision_train_test_multimodal.py --dataset banana --no-freeze
```

## 扩展测试与评估

### `test_banana/`

该目录主要用于在原始香蕉测试集与 `banana_add_70` 合并后的扩展测试集上评估模型。

主要文件：

- `test_cidis_combined.py`
- `test_cidis_multimodal_combined.py`
- `test_fruitvision_combined.py`
- `test_fruitvision_multimodal_combined.py`
- `test_fruitvision_combined.py`
- `train_banana_once.py`
- `config.yaml`

示例：

```bash
python test_banana/test_fruitvision_combined.py --result_folder results/你的FruitVision结果目录
python test_banana/test_cidis_combined.py --result_folder results/你的CIDIS结果目录
```

### `analysis/`

结果分析脚本目录。

常用文件：

- `significance.py`：比较不同模型的分类指标并进行统计显著性检验
- `significance_README.md`：显著性检验使用说明
- `run_significance_example.py`：显著性分析示例
- `kfold_summary.py`：汇总 K 折测试日志，输出均值、标准差和表格
- `banana_eval_original.py`：在原始香蕉测试集上评估模型
- `banana_eval_validation.py`：在训练集验证划分上评估模型
- `banana_combined_eval.py`：在香蕉合并测试集上评估模型
- `confusion_matrix.py`：混淆矩阵相关分析
- `auto_analysis.py`：自动化结果分析
- `wilcoxon_banana_analysis.py`、`wilcoxon_cucumber_analysis.py`：Wilcoxon 检验分析

示例：

```bash
# 汇总K折结果
python analysis/kfold_summary.py

# 显著性分析示例
python analysis/run_significance_example.py

# 香蕉合并测试集评估
python analysis/banana_combined_eval.py \
  --model_result_dir results/你的BananaModel结果目录 \
  --banana_dir data/banana \
  --banana_add_dir data/banana_add_70
```

### `distribution/`

用于分析新增香蕉数据与原始数据之间的分布差异。

主要文件：

- `distribution_30_70.py`：比较 30 条原始数据与 70 条新增数据
- `distribution_30_100.py`：比较原始数据与合并后 100 条数据
- `banana_30.csv`、`banana_70.csv`、`banana_100.csv`：分析用 CSV

运行：

```bash
python distribution/distribution_30_70.py
python distribution/distribution_30_100.py
```

### `relations/`

用于分析表型变量之间的相关性，并生成散点图、直方图等。

主要文件：

- `relations.py`
- `banana.csv`
- `cucumber.csv`

运行：

```bash
python relations/relations.py
```

## 配置文件说明

多数训练脚本使用 YAML 管理配置，并允许命令行参数覆盖。

常见配置项：

```yaml
dataset:
  type: banana
  img_dir: data/banana

training:
  epochs: 100
  batch_size: 20
  lr: 0.00005
  n_runs: 10
  n_folds: 5

model:
  pretrained_path: model/pretrained/resnet18.pth
  feature_fusion_type: cross_attention
  fpn_ablation_type: ripe_only
  combine_type: lka

loss:
  type: multi_task
  gamma: 0.1

save:
  results_dir: results
  model_name: BananaModel
```

常见参数含义：

- `epochs`：训练轮数
- `batch_size`：批大小，显存不足时优先减小
- `lr`：学习率
- `n_runs`：重复运行次数，通常用于显著性检验
- `n_folds`：K 折交叉验证折数
- `pretrained_path`：本地预训练权重路径
- `feature_fusion_type`：多分支特征融合策略
- `fpn_ablation_type`：FPN 消融设置
- `combine_type`：特征合并模块类型
- `loss.type`：多任务损失类型
- `gamma`：等级分类损失权重

## 结果输出

训练脚本通常会在 `results/` 下创建带日期和序号的实验目录，例如：

```text
results/
└── 20251125-BananaModel-b-1/
    ├── config.yaml
    ├── models/
    │   └── fold1/
    │       └── best_epochxx.pth
    ├── logs/
    │   └── training.log
    ├── figures/
    │   ├── fold1/
    │   │   ├── loss.png
    │   │   └── test/
    │   │       ├── confusion_matrix_*.png
    │   │       ├── roc_curves_*.png
    │   │       └── test_fold1.log
    │   └── ...
    └── *_summary.txt
```

常见输出包括：

- 最优模型权重：`.pth`
- 训练日志：`.log`
- 损失曲线：`loss.png`
- 混淆矩阵：`confusion_matrix_*.png`
- ROC 曲线和 AUC 图
- R2 散点图
- 多次运行均值和标准差
- 显著性检验报告

## 推荐运行流程

1. 安装依赖并确认能导入 PyTorch、timm、pandas。
2. 先用极小 epoch 验证数据路径和模型能跑通。
3. 确认 `results/` 中输出正常后，再运行完整实验。
4. 完整训练后使用对应测试脚本或 `analysis/` 脚本汇总结果。
5. 需要论文对比时，使用 `n_runs=10` 或 K 折结果做显著性检验。

示例调试流程：

```bash
# 1. 检查数据集读取
python data/dataset.py

# 2. 快速训练CIDIS
python CIDIS/CIDIS_train_test.py --dataset cucumber --epochs 2 --n_runs 1

# 3. 快速训练FruitVision
python FruitVision/fruit_vision_train_test.py --dataset cucumber --epochs 2 --n_runs 1

# 4. 快速训练香蕉多任务K折模型
python train_test_banana/train_banana.py --epochs 2 --n_folds 2
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'timm'`

安装缺失依赖：

```bash
pip install timm
```

### 2. `ModuleNotFoundError: No module named 'cv2'`

FruitVision 的 CLAHE/Otsu 预处理需要 OpenCV：

```bash
pip install opencv-python
```

### 3. 找不到预训练权重

检查配置中的 `model.pretrained_path` 是否存在。例如：

```text
model/pretrained/resnet18.pth
model/pretrained/resnet50.pth
model-SharedBackbone/pretrained/resnet50.pth
```

如果切换骨干网络，需要同步修改权重路径和模型配置。

### 4. GPU 显存不足

优先减小 `batch_size`：

```bash
python train_test_banana/train_banana.py --batch_size 8
```

FruitVision 还可以冻结骨干网络或使用多模态简化版。

### 5. 训练时间太长

先减少运行次数、折数和 epoch：

```bash
python train_test_banana/train_banana.py --epochs 20 --n_folds 3
python CIDIS/CIDIS_train_test.py --epochs 20 --n_runs 3
```

### 6. 随机种子数量不足

部分脚本要求 `seeds` 数量不少于 `n_runs`。如果将 `n_runs` 调大，需要同步增加配置文件中的随机种子。
