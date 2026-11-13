# MANIFEST -- Arm I (SEW-ResNet) source tree

Generated from the files on disk. Every size and hash below is measured.

## Pinned: Arm I's exact source set (hash-verified)

| file | bytes | sha256[0:16] | status | recovered from |
|---|---|---|---|---|
| `spike_dino/sdino_v7.py` | 47646 | `EDD4549DA2EF1E22` | MATCHES recorded | `tmp/_deployed/sdino_v7.py` |
| `scripts/train_sdino_v7.py` | 51372 | `D1FBD0ABD3D9707D` | MATCHES recorded | `tmp/_deployed/train_sdino_v7.py` |
| `spike_dino/spiking_vit.py` | 38919 | `CEC65516B853FE7F` | MATCHES recorded | `tmp/_srv_spiking_vit.py` |
| `spike_dino/sdino_v4.py` | 27250 | `2ADC1F67A66219C5` | MATCHES recorded | `tmp/_srv_sdino_v4.py` |

## Supporting modules (server's current copy; not hash-pinned for Arm I)

These are the import closure. AGENTS.md records no Arm-I-time hash for
them, so they are the server's present files and are labelled as such.

| file | bytes | sha256[0:16] |
|---|---|---|
| `scripts/eval_c10_v4.py` | 26640 | `6c665e253abb0edf` |
| `scripts/eval_v7_downstream.py` | 13424 | `6edd33aa7d18eb17` |
| `scripts/run_dvs_standard_ann_triage.py` | 7256 | `965ed0550402ca07` |
| `scripts/train_arm_i.py` | 1121 | `a098c5566bae6a98` |
| `scripts/train_generalization_reference.py` | 13744 | `11d0c0cb2a8bd028` |
| `scripts/train_sdino_v4.py` | 29051 | `fef22d1f61a509a8` |
| `spike_dino/__init__.py` | 0 | `e3b0c44298fc1c14` |
| `spike_dino/benchmark_manifest.py` | 7579 | `72ef73f9e5a0ebd9` |
| `spike_dino/core.py` | 14048 | `1b5c94b861f97994` |
| `spike_dino/data.py` | 34662 | `aa1bd4ad129cf998` |
| `spike_dino/dinov3.py` | 47353 | `3bd90504526ae727` |
| `spike_dino/downstream.py` | 10985 | `1681be782d29c4c0` |
| `spike_dino/fused_lif.py` | 4717 | `b31506e2f5e630ed` |
| `spike_dino/gradient_fusion.py` | 7125 | `f0b5aebf93470011` |
| `spike_dino/optimizer_precision.py` | 2943 | `0b491c23bd29727a` |
| `spike_dino/pretrain_v3.py` | 64195 | `8fada150eb64223c` |
| `spike_dino/probe_v4.py` | 20424 | `03d6708b136a5725` |
| `spike_dino/sdino_sew.py` | 6724 | `a318c4bb87e4478a` |
| `spike_dino/views_v4.py` | 7927 | `62efc8467d4cd83e` |
| `spike_dino/views_v5.py` | 9304 | `f7078b83076a7908` |
| `third_party/__init__.py` | 0 | `e3b0c44298fc1c14` |
| `third_party/eventclr/__init__.py` | 0 | `e3b0c44298fc1c14` |
| `third_party/eventclr/models/__init__.py` | 0 | `e3b0c44298fc1c14` |
| `third_party/eventclr/models/pretraining.py` | 979 | `4d7f58d8b03dc5f4` |
| `third_party/eventclr/models/sew_resnet.py` | 7738 | `13732e420fabcad7` |
| `third_party/eventclr/ntxent.py` | 3605 | `f125f1bbe3d144fd` |
| `third_party/eventclr/utils/__init__.py` | 0 | `e3b0c44298fc1c14` |
| `third_party/eventclr/utils/spiking_neuron.py` | 723 | `6922c57debfcc30b` |

Excluded on purpose: `runs/`, `logs/`, every `*.pt` checkpoint, the
CIFAR-10-DVS manifest and data cache, `third_party` model weights.
Nothing here is model weights; the tree is code only.
