# Arm I —— 方法说明

> 这是一份**方法文档**，只讲 Arm I 是什么、为什么这样做、怎么复现、以及它能/不能拿来支持什么结论。
> 目录结构与逐文件哈希见同目录的 `README.md` 与 `MANIFEST.md`。

---

## 0. 一句话

**Arm I = Arm F 的完整 Spike-DINO 方法，把「球面 DC 原点减法」这一个组件关掉。**

Arm F 是本项目当前最好的 SEW-ResNet18 配置（KoLeo + `proto_balance`）。Arm I 是它唯一一次干净的单变量消融：**只把 `--center-rate` 换成 `--no-center-rate`**，其余 40 多个 flag 逐字相同，两份源文件同源（`sdino_v7.py` 47,646 B `EDD4549DA2EF1E22`，`train_sdino_v7.py` 51,372 B `D1FBD0ABD3D9707D`）。

---

## 1. 方法全貌

Spike-DINO 是 **DINO 自蒸馏目标**（没有 InfoNCE / NT-Xent / 负样本对），加两个辅助正则项，在 **特征球面** 上运作。

### 1.1 主干与 DINO 目标

- 主干 `sew_resnet18`（`spike_dino/sdino_sew.py` + `third_party/eventclr/models/sew_resnet.py`），输入是 CIFAR-10-DVS 事件体（形状由 `data.py` / `views_v*.py` 决定，本 arm 没有覆盖任何尺寸或时间步参数）。
- Teacher = Student 的 EMA；Teacher 的 BN 用 `--teacher-bn-mode train`。
- 每步 4 个视图：`[global0, global1, temporal0, temporal1]`（`spike_dino/views_v4.py`、`views_v5.py`）。
- 两个 global 视图过 Teacher 得到原型分布，温度 `0.04 → 0.07`（30 epoch warmup）；Student 温度 `0.1`。
- 交叉视图蒸馏：`q0 → student(view1)`、`q1 → student(view0)`；temporal 视图的项按 **Teacher 两视图自洽度** `agreement` 加权，下限 `reliability_floor = 0.25`。

```
dino_global   = mean[ -q · log_softmax(z_s / 0.1) ]        跨 global 视图
dino_temporal = 按 agreement 加权的同类项
loss_DINO     = dino_global + 1.0 · dino_temporal
```

`lambda_temporal` 默认 1.0。所有几何计算在 autocast 之外用 fp32 做（余弦 logits 除以 0.04，bf16 会把目标量化到约 0.1 个 logit 步长）。

### 1.2 组件一：KoLeo hinge（`koleo_floor`）

`--koleo-weight 0.1 --koleo-distance-floor 0.1 --koleo-auto-weight --koleo-probe-target .25 --koleo-probe-cap 1.0`

在 Student 两个 global 视图的球面特征上，对**最近邻距离**做排斥：

```
L = -log(d + floor)          d = 最近邻 L2 距离，floor = 0.1
```

- `floor = 0` 时逐位退化为 DINOv2 的 `koleo_loss`（代码里**直接调用**它，见 `sdino_v7.py:211`，所以等价性是构造出来的而不是抄出来的）。
- 加 floor 的理由是实测的：无 floor 的 KoLeo 在 weight 0.1 下 pre-clip `grad_norm` 冲到 **167.58**，而 DINO 项只有 2.27–3.23，那个 arm 在 e4.71 就被预算停掉了。爆炸完全来自 `d → 0` 尾巴的 `1/d` 梯度（典型最近邻距离 `exp(-1.1417) = 0.319`）。`student_feature_cos_max` 在各 arm 都是 0.95–1.00，尾巴是真实且被占满的；floor 0.1 落在尾巴内部（cos 0.995），把它压约 10 倍而不动典型项。
- 与 `clamp_min` 的区别很重要：hinge 在最接近的那些对上**保留有限排斥**，而不是把它们静音 —— 而最接近的对恰恰是多样性项存在的理由。
- `--koleo-auto-weight` 在运行时用梯度探针**标定权重**：测得 DINO 梯度范数 `mean_d` 与 KoLeo 梯度范数 `mean_k` 后取

  ```
  w_k = 0.25 · mean_d / mean_k ,  且  w_k · mean_k ≤ 1.0
  ```

  即把 KoLeo 的梯度**校准到 DINO 梯度的约 25%**，并且报 `--koleo-weight` 是未加权 hinge 的范数，所以一次测量就定权重，不会产生加权目标那种震荡反馈。

### 1.3 组件二：`proto_balance`

`--proto-balance-weight 0.333125 --balance-temp .1`

K = 256 个可学习原型的**边际使用**要均匀。取 Student 两个 global 视图的 logits 拼接、除以 `balance_temp = 0.1`、softmax、对样本取均值，得到边际 `m_k`：

```
balance = Σ_k m_k · log m_k        （= 负边际熵）
loss   += 0.333125 · balance
```

**方向必须从损失项推，不能从名字推**：这个项被**最小化**，而 `Σ m log m` 在边际均匀时取最小值（`-log K`）。所以它把边际**推向均匀**，读到的是一个**使用计数**型的量 —— PASS 在高的一侧，坍塌统计在低的一侧。

它做的是：把「原型摆在球面上互相分开」和「原型真的被样本选中」这两件事区分开。`proto_frame` 只保证前者；一个从不被选中的原型照样占着球面。实测上 `proto_balance` 是 v7 第一个**同时**推动机制与精度的项（Arm C，e50 53.83 首次通过）。

### 1.4 组件三：球面 DC 原点减法 —— **Arm I 关掉的就是它**

Spike rate 非负，所以球面特征被困在一个「帽」里：实测 DC 占比 `||mean|| / mean||x|| = 0.752`。DINO 的反坍塌机制是为零均值特征设计的，于是这里在 **feature 层**减掉一个 EMA 原点：

```python
# sdino_v7.py:105-118
def spherical_feature(rate, center=None, arcsin_sqrt=False, eps=1e-6):
    x = rate.float()
    if arcsin_sqrt:  x = torch.asin(x.clamp(0.,1.).sqrt())   # variance stabilisation
    if center is not None:  x = x - center                    # <-- 原点减法
    return F.normalize(x, dim=-1, eps=eps)

# sdino_v7.py:484-486
def _sphere(self, rate):
    origin = self.feature_center if self.center_rate else None
    return spherical_feature(rate, origin, self.arcsin_sqrt)
```

- 原点 `feature_center` 是 Student global 视图的 **EMA 均值**（`sdino_v7.py:489-498`，momentum 0.99，label-free，只从 Student 取）。
- `center_rate = False` 时：`origin = None` → 退化为 `F.normalize(rate)`，即 v5 的几何；`update_feature_center` 直接 early-return，缓冲从不更新，日志里 `feature_center_norm` 恒为 0。
- `arcsin_sqrt` 在两侧都是 **OFF**，所以 Arm I 与 Arm F 之间唯一变化的就是这个 EMA 原点。

### 1.5 总损失

```
loss = DINO(global) + 1.0 · DINO(temporal, reliability-weighted)
     + w_koleo   · KoLeo_hinge(floor=0.1)          w_koleo 由梯度探针标定 (~0.25 比例)
     + 0.333125  · proto_balance(balance_temp=0.1)
```

Arm I 里 `feature_center` **不进入**这条式子。没有 InfoNCE、没有负样本对、没有第三个目标。

---

## 2. 唯一的变量：Arm F → Arm I

| | Arm F | **Arm I** |
|---|---|---|
| `--center-rate` | True（默认） | **False**（`--no-center-rate`） |
| 球面特征 | `normalize(rate − EMA_origin)` | **`normalize(rate)`** |
| `feature_center_norm` | 2.0958 @ step1，4.977 @ e20 | **0.0 全程** |
| `rate_centering` 记录 | `ema_origin` | **`none`** |
| 其余全部 flag | — 完全相同 — | — 完全相同 — |

---

## 3. 代码地图

| 位置 | 内容 |
|---|---|
| `spike_dino/sdino_v7.py:105-118` | `spherical_feature`（arcsin_sqrt → 减原点 → L2 normalize） |
| `spike_dino/sdino_v7.py:178-215` | `koleo_hinge`（floor 版 KoLeo；floor≤0 直接转调 `sdino_v4.koleo_loss`） |
| `spike_dino/sdino_v7.py:353-421` | `SpikeDinoV7.__init__`，全部组件权重与开关，含 `center_rate` 默认值 |
| `spike_dino/sdino_v7.py:484-498` | `_sphere` / `update_feature_center` —— **DC 开关的作用点** |
| `spike_dino/sdino_v7.py:511-625` | `forward`：目标构造、DINO 项、KoLeo、CSSP、`proto_balance`、总损失 |
| `spike_dino/sdino_v7.py:768` | `feature_center_norm` 的日志（关闭时记 `loss.new_tensor(0.)`） |
| `scripts/train_sdino_v7.py:589-597` | `--koleo-auto-weight` 的梯度标定 |
| `scripts/train_sdino_v7.py:728-731` | `--center-rate` / `--no-center-rate`（**dest 相同，默认 True**） |
| `spike_dino/sdino_sew.py` | SEW-ResNet18 主干封装 |
| `spike_dino/probe_v4.py` | e20/e50/…/e300 探针（kNN / linear / effective rank） |

---

## 4. 完整训练配置（Arm I 原样）

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

> **默认值方向**：这个文件里 `center_rate` 默认是 **True**。不加 flag 直接跑，得到的是 **Arm F**。
> 图省事可以用本目录的 `scripts/train_arm_i.py`，它在未经显式指定时追加 `--no-center-rate`（不改任何源文件）：
>
> ```bash
> python scripts/train_arm_i.py --output runs/arm_i_repro --epochs 300 ...
> ```

`--gate-epoch 9999` 是刻意的：训练器自带的 gate 机制被停用，判定只在人工登记的 epoch 上读一次。

---

## 5. 结果

**dev probe（concat kNN / linear / rank）**，Arm I 与 Arm F 同 epoch 对照：

| epoch | Arm I | Arm F | I − F（linear） |
|---|---|---|---|
| e150 | 54.50 / 60.03 / 46.40 | 53.70 / 59.27 / 35.59 | **+0.76** |
| e300（PRIMARY） | 56.50 / 61.47 / 56.43 | 55.10 / 59.10 / 44.74 | **+2.37** |

- e140–e300 之间 **17 个同 epoch 探针，Arm I 全部为正**（均值 +2.01，从未为负）；centred rank 每个探针高 +11.0 到 +11.7。
- 但 **+2.37 仍落在 3.60 的重跑带之内**，且 e150 与 e300 方向不一致（e150 FLAT、e300 超 +2.0）—— 按登记规则这是一次 **epoch 之间的矛盾**，不是结论。

**官方 1000 条 test，全标签 linear probe = 58.80**（val 60.90，sd 0.00，单次确定性拟合）。
Arm F 同协议 = **58.80** → 差 **0.00**，在登记的 55.20–62.40 带内 → **NOT SEPARABLE，按本项目的判例记作 NEGATIVE**。

**10-seed few-shot（1 / 9 / 45 shot）**：**22.39 / 35.52 / 46.10**（配对 vs Arm F：+0.11 / +0.64 / +1.18，全部 NULL）。

**对 v6 对比学习对照组**（全标签 concat linear 61.00 / 62.97）：**−1.50**，**落在 3.60 带内** —— Arm I 是第一个全标签差距进入带内的 v7 arm。

### 结论

球面 DC 原点减法在这个配置下 **既不是承重结构，也不是有害结构**：dev probe 上有一致但带内的小正号，官方传输上精确为零。方法因此仍写作 **两个组件（KoLeo + `proto_balance`）加一个中性默认值**。

---

## 6. 怎么确认一条 run 真的是 Arm I（I-CONFIG）

三者**缺一不可**，任何一条不符 → 这条 run 不是 Arm I，后面所有数字都不能读：

1. **配置**：`recorded_config` 里 `rate_centering: none`；输出目录名与登记一致。
2. **机制**：`metrics.jsonl` 每一行 `feature_center_norm` **严格等于 0.0**（关闭时记 `loss.new_tensor(0.)`）。
   对照组：Arm F 在 step 1 记 2.0958、e20 记 4.977 —— 只要出现任何一个非零值，说明 flag 没生效。
3. **数值**：0 条 non-finite `loss` / `grad_norm`；Arm I 实测 pre-clip `grad_norm` 最大 **2.0997**。

---

## 7. 这个 arm 允许 / 不允许说的话

**允许**

- 「去掉球面 DC 原点后，机制统计有一致但落在重跑带内的小幅正向（17/17 个探针同号，均值 +2.01），而官方全标签传输差为 0.00。」
- 「DC 减法在本配置下是中性默认值：既不承重也无害。」
- 「Arm I 是 v7 里第一个全标签差距进入 3.60 带内的 arm（对 v6 对照 −1.50）。」

**不允许**

- 把 DC removal 写成核心贡献或 headline 机制 —— 官方读数精确持平，按判例是 NEGATIVE。
- 只引 dev 的 **+2.37** 而不并列官方 **58.80**。
- 用 `feature_dc_share`（Arm I e20 = 0.487，F = 0.865）论证「表示被 DC 毒害」。这个统计量算在 `_transform(teacher_rate)` 上，**本身就是 `center_rate` 的痕迹**，跨不同 `center_rate` 的 arm 之间不可比。
- 拿 Arm I 与 A/C/F/G 之外的 arm 配对 —— 只有共享同一组源哈希的 arm 才能构成一对。Arm I 与 ViT 线、local-pred 线**不配对**。

---

## 8. 复现注意事项

- **源哈希**：`sdino_v7.py` 47,646 B `EDD4549DA2EF1E22`、`train_sdino_v7.py` 51,372 B `D1FBD0ABD3D9707D`、`spiking_vit.py` 38,919 B `CEC65516B853FE7F`、`sdino_v4.py` 27,250 B `2ADC1F67A66219C5`。跑之前核对。
- **重跑带 3.60 点**（concat linear），机制统计 0.004。任何精度比较都必须带这条带；带内的差异写「与水平一致，不构成可分差异」，不写「接近」。
- **训练中不要读 `status.json`**：只读 `metrics.jsonl`（逐步追加）与 `probes/*.json`（每个探针写一次）。
- **唯一停止规则**：出现 non-finite，或 pre-clip `grad_norm` > 20。没有精度停止规则；被停掉的 arm 永不重启；超参不在中途修改。
- checkpoint：`runs/sdino_v7_koleo_balance_nodc_c10_e300_20261112/epoch_300.pt`（293,446,507 B `4BC6DB9F519E504C`），留在服务器，不在本目录。
