# ADNI Multimodal (MRI+PET) 3D DOCO

本目录是在原 ImageNet 2D DOCO 之外新增的 **ADNI 3D 多模态** 实验线，保留 Prompt + 源域 mean/std 补偿 + OOD 路由。

## 数据布局（服务器）

```text
/home/lz/DOCO-main/data/
  ADNI.csv
  MRI/sub-ADNI002S1261.nii.gz
  PET/sub-ADNI002S1261.nii.gz
  ...
```

`ADNI.csv` 关键字段示例：

```text
Image Data ID,Subject,Group,Sex,Age,Visit,Modality,...,PTID,Closest_MMSE,...
I251187,sub-ADNI153S4172,AD,M,76,v02,MRI,...,153_S_4172,23.0,...
```

配对规则：

1. 标签与元信息来自 `ADNI.csv`（`PTID=002_S_1261` 或 `Subject=sub-ADNI002S1261`）
2. 文件名 `sub-ADNI002S1261.nii.gz` 会解析成同一 PTID
3. 模态由目录决定：`MRI/` vs `PET/`
4. 仅保留 **CN/AD** 且 **两模态文件都在** 的被试

文件名示例：

```text
/home/lz/DOCO-main/data/MRI/sub-ADNI002S1261.nii.gz
/home/lz/DOCO-main/data/PET/sub-ADNI002S1261.nii.gz
```

第一版任务：**CN vs AD**（CN=0, AD=1）。

## 依赖

```bash
conda activate doco
pip install nibabel
```

## 1) 生成配对清单与划分

随机划分（同分布）：

```bash
python -m adni.data.prepare_manifest \
  --data_root /home/lz/DOCO-main/data \
  --csv /home/lz/DOCO-main/data/ADNI.csv \
  --split_by random \
  --seed 1
```

按站点留出测试域（推荐给 DOCO）：

```bash
python -m adni.data.prepare_manifest \
  --data_root /home/lz/DOCO-main/data \
  --csv /home/lz/DOCO-main/data/ADNI.csv \
  --split_by site \
  --test_ratio 0.2 \
  --seed 1
```

输出：

- random → `manifest_cn_ad_pairs.csv`
- site → `manifest_cn_ad_site.csv`

## 2) 训练源模型

默认使用训练集逆频率加权 CE，并把 Age/Sex 拼到 CLS 后再分类（年龄用训练集 mean/std 标准化，M=1/F=0，缺失性别=0.5）。选模按 val balanced acc，平局留 val_loss 更低的 checkpoint。旧的纯影像 `best.pt` 不能 load 到新分类头，需要重训。关掉：`--no_class_weight` / `--no_tabular`。

```bash
python adni/train_source.py \
  --data_root /home/lz/DOCO-main/data \
  --csv /home/lz/DOCO-main/data/ADNI.csv \
  --manifest /home/lz/DOCO-main/data/manifest_cn_ad_site.csv \
  --save_dir /home/lz/DOCO-main/output_adni/source_site \
  --model_size small \
  --img_size 96 \
  --batch_size 2 \
  --epochs 50
```

## 3) DOCO 测试时自适应

```bash
python adni/tta_main.py \
  --data_root /home/lz/DOCO-main/data \
  --csv /home/lz/DOCO-main/data/ADNI.csv \
  --manifest /home/lz/DOCO-main/data/manifest_cn_ad_site.csv \
  --checkpoint /home/lz/DOCO-main/output_adni/source_site/best.pt \
  --save_dir /home/lz/DOCO-main/output_adni/doco_tta_site \
  --batch_size 2
```

`--gate_mode none` 关闭门控（对照实验建议关掉）。默认 **闭集**（整 batch 更新 prompt）。

## MAE 预训练 ViT-B（MICCAI 2024）

用 [ViT_recipe_for_AD](https://github.com/qasymjomart/ViT_recipe_for_AD) 的 3D ViT-B MAE 编码器（75% mask，BRATS+IXI+OASIS3，单通道 T1，`128³`）。分类头在你们的站点划分上重训。TTA 仍是原文：冻骨干、CLS 后插 prompt、`L_stat` + `L_reg`。旧 `best.pt` 对不上，CN/AD 和 MCI 都要重训源。

权重路径：

`/home/lz/DOCO-main/pretrained/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth`

源阶段 `--prompt_num 0`（和原版 DOCO 一样，prompt 只在 TTA 出现）。骨干 lr = `1e-4 * 0.1 = 1e-5`，头 `1e-4`。CN/AD 数据在 `data_AD_CN`。

CN vs AD（3 seed；seed1 清单名没有 `_seed1`）：

```bash
PRE=/home/lz/DOCO-main/pretrained/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth
ROOT=/home/lz/DOCO-main/data_AD_CN

python adni/train_source.py \
  --data_root $ROOT \
  --csv $ROOT/ADNI.csv \
  --manifest $ROOT/manifest_cn_ad_site.csv \
  --save_dir /home/lz/DOCO-main/output_adni/source_mae_seed1 \
  --model_size base --img_size 128 --batch_size 2 --epochs 80 \
  --lr 1e-4 --backbone_lr_scale 0.1 --prompt_num 0 --seed 1 \
  --pretrained $PRE

python adni/train_source.py \
  --data_root $ROOT \
  --csv $ROOT/ADNI.csv \
  --manifest $ROOT/manifest_cn_ad_site_seed2.csv \
  --save_dir /home/lz/DOCO-main/output_adni/source_mae_seed2 \
  --model_size base --img_size 128 --batch_size 2 --epochs 80 \
  --lr 1e-4 --backbone_lr_scale 0.1 --prompt_num 0 --seed 2 \
  --pretrained $PRE

python adni/train_source.py \
  --data_root $ROOT \
  --csv $ROOT/ADNI.csv \
  --manifest $ROOT/manifest_cn_ad_site_seed3.csv \
  --save_dir /home/lz/DOCO-main/output_adni/source_mae_seed3 \
  --model_size base --img_size 128 --batch_size 2 --epochs 80 \
  --lr 1e-4 --backbone_lr_scale 0.1 --prompt_num 0 --seed 3 \
  --pretrained $PRE
```

pMCI vs sMCI（3 seed，`--no_zscore`）：

```bash
PRE=/home/lz/DOCO-main/pretrained/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth
for s in 1 2 3; do
  python adni/train_source.py \
    --data_root /home/lz/DOCO-main/data_MCI \
    --manifest /home/lz/DOCO-main/data_MCI/manifest_pmci_smci_site_seed${s}.csv \
    --save_dir /home/lz/DOCO-main/output_mci/source_mae_seed${s} \
    --model_size base --img_size 128 --batch_size 2 --epochs 80 \
    --lr 1e-4 --backbone_lr_scale 0.1 --prompt_num 0 --seed ${s} \
    --no_zscore --pretrained $PRE
done
```

闭集 DOCO（prompt=8，和原文一致）：

```bash
# CN/AD（seed1 清单是 manifest_cn_ad_site.csv）
python adni/tta_main.py \
  --data_root /home/lz/DOCO-main/data_AD_CN \
  --csv /home/lz/DOCO-main/data_AD_CN/ADNI.csv \
  --manifest /home/lz/DOCO-main/data_AD_CN/manifest_cn_ad_site.csv \
  --checkpoint /home/lz/DOCO-main/output_adni/source_mae_seed1/best.pt \
  --save_dir /home/lz/DOCO-main/output_adni/doco_mae_seed1 \
  --split_mode closed --gate_mode none --prompt_num 8 --batch_size 2

# MCI
python adni/tta_main.py \
  --data_root /home/lz/DOCO-main/data_MCI \
  --manifest /home/lz/DOCO-main/data_MCI/manifest_pmci_smci_site_seed1.csv \
  --checkpoint /home/lz/DOCO-main/output_mci/source_mae_seed1/best.pt \
  --save_dir /home/lz/DOCO-main/output_mci/doco_mae_seed1 \
  --split_mode closed --gate_mode none --prompt_num 8 --batch_size 2 --no_zscore
```

`--pretrained` 会强制 MRI-only。PET 不进这套权重。输入会从 checkpoint 读 `img_size=128`，TTA 不用再写 `--img_size`。


`--drop_pet`：仅测试时把 PET 通道置 0，源统计仍用双模态训练集。`--mri_only`：训练/测试都只读 MRI（`in_chans=1`），和缺 PET 不是同一设定。`--dual_stream`：MRI/PET 分路 patch、测时按模态插入 prompt（源训练用 `--prompt_num 0`，不要叠源阶段 VPT）。缺 PET 时只更新 MRI prompt。

额外设定：

```bash
# TENT 基线（只更新 LayerNorm 仿射参数，熵最小化）
python adni/tta_main.py ... --method tent --gate_mode none

# DOCO：按「到源域 CLS 均值的距离」划分 ID/OOD
# ID 子集更新 prompt，OOD 子集只套用当前 prompt（0 梯度）
python adni/tta_main.py ... --split_mode source_stat --gate_mode none
```

开集类别路由消融：`--split_mode open` 或 `--open_set_routing`。

## 设计说明

| 项 | 选择 |
|----|------|
| 融合 | MRI/PET stack → `[2,D,H,W]` |
| 输入 | resize 到 `96³` |
| 骨干 | `VisionTransformer3D` |
| DOCO | `PromptViT3D` + `DOCO3D` |
| 类别 | 先 2 类 CN/AD |

原 `imagenet/` 实验线未改动。
