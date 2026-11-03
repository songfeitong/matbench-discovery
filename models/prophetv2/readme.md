# ProphetV2 本地接入冒烟测试

这个目录用于验证 ProphetV2 与官方共享代码的调用链，尚不是榜单模型提交。
ProphetV2 源码在相邻的 `../prophetv2` 仓库中维护，不复制进本仓库。

`smoke.py` 临时注册 `prophetv2_local` Calculator，使用官方
`load_calculator()` 加载导出文件，再调用官方 `relax_atoms()`，不改写评测算法。
注册仅在测试进程内生效；不提供虚构的模型 YAML、公开权重链接或榜单指标。

## 环境与运行

在本仓库根目录执行：

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e . \
  'torch==2.12.1' \
  'nvidia-cutlass-dsl==4.8.1' \
  'nvalchemi-toolkit-ops==0.5.0' \
  'apache-tvm-ffi==0.1.15.post2' \
  'accelerate==1.15.0' \
  'ase-db-backends==0.11.0' 'lmdb==3.0.0'

PYTHONPATH="$(realpath ../prophetv2/src)" CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python models/prophetv2/smoke.py \
  --output-dir ../runs/matbench-discovery-smoke-20261103

uv pip freeze --python .venv/bin/python --exclude-editable \
  > ../runs/matbench-discovery-smoke-20261103/environment.txt
```

源码路径通过 `PYTHONPATH` 显式提供，仅用于 Python 3.14 兼容性探测。
当前 ProphetV2 包的 `requires-python` 仍是 `>=3.12,<3.13`，不能据此声称
已经完成 Python 3.14 的常规包安装支持。正式接入前应更新并验证其构建元数据、
依赖锁定和相关检查。本测试不修改 ProphetV2 仓库。

首次导入官方 `discovery` 模块会按官方机制下载约 13 MB 的 WBM 汇总表；
测试本身只用合成硅晶体，不下载 WBM 结构全集。

## 验证范围

- 固定随机种子，分别构造未训练的小型 Dense 和 edge-MoE，导出并重新加载。
- 使用扰动后的双原子 Si 周期晶胞，核对能量、力、应力有限，
  并与直接调用原始 MLIP 的结果比较，容差为 `rtol=1e-4, atol=1e-5`。
- 采用官方 `FIRE + FrechetCellFilter` 实际执行两步变胞优化。
  为避免随机小模型在初始点提前收敛，测试专用阈值设为 `fmax=1e-8`；
  这不是正式评测协议，未收敛是两步截断的预期结果。
- 以 FP32、无 autocast、`matmul_precision=highest` 运行；
  输出随机模型文件、权重 SHA-256 和 `report.json`。

2026-11-03 在 RTX PRO 6000 Blackwell 上验证通过：

| 模型 | 能量/力/应力与直接调用一致 | 变胞优化步数 |
| --- | --- | --- |
| Dense | 是 | 2 |
| edge-MoE | 是 | 2 |

验证版本：Python 3.14.6、PyTorch 2.12.1+cu130，
Matbench Discovery 基于 `c70947f7fd176b8914b72c4ec9a974adfa3b4d50`，
ProphetV2 基于 `34a72ebe22ca377a422ad681aafca2d59240d5c7`。

这证明源码调用和共享弛豫路径可用，不代表训练权重精度、WBM 全流程、
形成能修正、声子、MD 或正式 runner CLI 已完成验证。
后续用固定训练权重注册正式模型，再运行官方 Discovery dry-run。
