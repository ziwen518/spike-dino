# main_code -- Arm I（SEW-ResNet18）核心架构代码

本目录是 **Arm I**（`sdino_v7_koleo_balance_nodc_c10_e300_20261112`）那份代码的本地副本，**只含代码，不含任何 checkpoint / 数据 / 日志**。每个文件的字节数与 sha256 见 `MANIFEST.md`（那张表由脚本实测生成，不是手抄的）。

> **想看方法本身、而不是目录清单** → 读 `README_ARM_I.md`。那份文档单独讲 Arm I 的损失构成、三个组件、唯一变量、复现命令、结果与引用边界。本文件只讲目录里放了什么、从哪来。

## 1. 一句话说明这套代码是什么

Arm I = **Arm F + 一个 flag**：`--no-center-rate`。也就是说，v7 的 DINO-on-sphere 方法里去掉"球面 DC 原点减法"，其余完全相同。

三个组件（本项目的全部方法组合）：

| 组件 | 位置 | 作用 |
|---|---|---|
| KoLeo floor（`--koleo-*`） | `spike_dino/sdino_v7.py` | 特征均匀性正则（`koleo_floor`），v7 第一个正收益项 |
| `proto_balance`（`--proto-balance-weight`） | `spike_dino/sdino_v7.py` | 原型边际平衡正则，第一个同时推动机制与精度的 v7 项 |
| 球面 DC 原点减法（`--center-rate` / 本 arm 的 `--no-center-rate`） | `spike_dino/sdino_v7.py:484-498` | 特征级 EMA 原点；**Arm I 把它关掉了** |

关掉之后 `spherical_feature(rate, None, False)` 退化成 `F.normalize(rate)`，`feature_center_norm` 全程记 0。这是本 arm 唯一被操纵的东西。

## 2. 关键文件

| 文件 | 角色 |
|---|---|
| `spike_dino/sdino_v7.py` | **模型本体**：Teacher/Student、球面特征、原型、`proto_balance`、KoLeo |
| `spike_dino/spiking_vit.py` | Spiking ViT（SPS stem、`spikformer_count`、残差策略）；`sdino_v4` 会 import 它 |
| `spike_dino/sdino_v4.py` | Encoder wrapper 与 `spike_dino_v7` 依赖的 helper |
| `spike_dino/sdino_sew.py` | **SEW-ResNet18 trunk** |
| `third_party/eventclr/models/sew_resnet.py` | 上游 SEW-ResNet18 定义（`spike_dino/core.py` 引用） |
| `scripts/train_sdino_v7.py` | **训练器**（Arm I 版本，含 `--no-center-rate` 与全部 v7 flag） |
| `spike_dino/data.py` / `views_v4.py` / `views_v5.py` | CIFAR-10-DVS 读取与 multi-crop / temporal DINO 视图 |
| `spike_dino/probe_v4.py` | e20/e50/…/e300 的 kNN / linear / rank 探针 |
| `scripts/eval_v7_downstream.py` + `scripts/eval_c10_v4.py` | few-shot / 官方 test 的 LP & FT 评测 |

## 3. 复现命令（Arm I 原样）

```bash
python -m scripts.train_sdino_v7 \
  --output runs/sdino_v7_koleo_balance_nodc_c10_e300_20261112 \
  --dataset cifar10dvs --backbone sew_resnet18 \
  --epochs 300 \
  --batch-size 64 --base-lr 2e-3 --warmup-epochs 10 \
  --precision bf16 --teacher-bn-mode train \
  --proto-count 256 --no-center-rate \
  --teacher-temp 0.04 0.07 --teacher-temp-warmup-epochs 30 --student-temp 0.1 \
  --koleo-weight 0.1 --koleo-distance-floor 0.1 \
  --koleo-auto-weight --koleo-probe-target .25 --koleo-probe-cap 1.0 \
  --koleo-grad-probe 20 \
  --proto-balance-weight 0.333125 --balance-temp .1 \
  --save-at-steps 20 40 80 140 \
  --probe-at-start --probe-every 10 --gate-epoch 9999 \
  --workers 12 --log-every 20
```

把 `--proto-count 256 --no-center-rate` 那一行里的 `--no-center-rate` 换成 `--center-rate`（或直接删掉该 flag）就是 **Arm F**，两者是本项目唯一一对干净的单变量对照。

**注意默认值方向**：Arm I 的原件 `train_sdino_v7.py` 里 `center_rate` 默认是 **True**（`--center-rate` 是 `store_true` + `default=True`，`:728`），因为 Arm I 与 Arm F 的差别只在命令行这一个 flag 上。所以**不加 flag 直接跑得到的是 Arm F，不是 Arm I**。为此本目录多放了一个不修改任何源文件的小包装：

```bash
python scripts/train_arm_i.py --output runs/arm_i_repro --epochs 300 ...   # 其余参数同上
```

它只是在你没有显式指定 centering 模式时往 `sys.argv` 追加 `--no-center-rate`，所以四个被 pin 的源文件哈希不受影响。实测：走包装 `center_rate=False`，直接跑 `train_sdino_v7.py` 是 `center_rate=True`。

启动后核对两条：`recorded_config` 里 `rate_centering: none`，`metrics.jsonl` 每一行 `feature_center_norm` 严格为 `0.0`（Arm I 就是这么验的）。

## 4. Arm I 的最终读数（记录用，不在本目录重新计算）

dev probe（concat kNN / linear / rank）：

| epoch | Arm I | Arm F | I − F（linear） |
|---|---|---|---|
| e150 | 54.50 / 60.03 / 46.40 | 53.70 / 59.27 / 35.59 | +0.76 |
| e300（PRIMARY） | 56.50 / 61.47 / 56.43 | 55.10 / 59.10 / 44.74 | +2.37 |

e140–e300 之间 17 个同 epoch 探针，Arm I 全部为正（均值 +2.01，从未为负），centred rank 每个探针都高 +11.0 到 +11.7。

- 官方 1000 条 test 的全标签 LP：**58.80**，与 Arm F 的 58.80 **完全相等**（look TEN）→ 按本项目判例记作 **NEGATIVE**：DC 减法既不承重也无害。
- 10-seed few-shot（1 / 9 / 45 shot）：**22.39 / 35.52 / 46.10**。
- 全标签 concat linear 对 v6 对比学习对照组（61.10）的差距 −1.50，**落在 3.60 重跑带之内**。

结论写在 `AGENTS.md` §4：Arm I 已 CLOSED，方法就是 **两个组件（KoLeo + `proto_balance`）加一个中性默认值**。

## 5. 一个重要警告：这套代码不是服务器当前的代码

服务器上现在的 `spike_dino/sdino_v7.py` 已经是 **51,663 B `DCC5CA811A88CE0B`**（local-pred 分支），`scripts/train_sdino_v7.py` 是 **58,435 B `0963E6B0D13A5289`**，`spiking_vit.py` 42,380 B，`sdino_v4.py` 29,219 B —— 都**不是** Arm I 跑的那一份。

所以本目录里决定模型行为的四个文件不是从服务器现在的副本取的，而是从本地归档里取回的 Arm I 原件，并逐个对上了 `AGENTS.md` 记录的 sha256：

| 文件 | 字节 | sha256[0:16] | 取自 |
|---|---|---|---|
| `spike_dino/sdino_v7.py` | 47,646 | `EDD4549DA2EF1E22` | `tmp/_deployed/sdino_v7.py` |
| `scripts/train_sdino_v7.py` | 51,372 | `D1FBD0ABD3D9707D` | `tmp/_deployed/train_sdino_v7.py` |
| `spike_dino/spiking_vit.py` | 38,919 | `CEC65516B853FE7F` | `tmp/_srv_spiking_vit.py` |
| `spike_dino/sdino_v4.py` | 27,250 | `2ADC1F67A66219C5` | `tmp/_srv_sdino_v4.py` |

其余文件（`data.py`、`views_*.py`、`probe_v4.py`、`pretrain_v3.py`、`sdino_sew.py`、`third_party/`、eval 脚本等）是服务器**当前**副本，它们没有 Arm I 时期的哈希记录，`MANIFEST.md` 里单独列了一张表标明这一点。**只有上面四行可以被引用为"与 Arm I 同源"。**

## 6. 未包含的东西

- `runs/`、`logs/`、任何 `*.pt`。Arm I 的 checkpoint 是 `runs/sdino_v7_koleo_balance_nodc_c10_e300_20261112/epoch_300.pt`（293,446,507 B `4BC6DB9F519E504C`），留在服务器上。
- CIFAR-10-DVS 数据与 manifest、`third_party` 预训练权重。
- 训练用的 `.cmd` 启动器（那些在 `tmp/`）。

## 7. 自检

在本目录下（`sys.path` 指到本目录）已实测通过：

- `python -m py_compile` 全部文件通过；
- 以下模块全部 import 成功：`spike_dino.{core, benchmark_manifest, data, gradient_fusion, optimizer_precision, pretrain_v3, views_v4, views_v5, probe_v4, downstream, sdino_sew, spiking_vit, sdino_v4, sdino_v7}` 与 `scripts.{train_generalization_reference, train_sdino_v4, eval_c10_v4, eval_v7_downstream, train_sdino_v7}`；
- `from spike_dino.sdino_v7 import SpikeDinoV7` 成功。

依赖：`torch`、`spikingjelly`、`numpy`、`paramiko`（仅上传用）。本地用 `D:/APP/Anaconda3/python.exe`，服务器用 `.venv\Scripts\python.exe`。
