# ProphetV2 本地模型评测

`prophetv2_local` 是一次注册、反复使用的本地 Calculator 入口，接受任意兼容的 FP32 Dense/MoE 导出文件。权重、代码和协议的具体身份由每次运行的外部配置保存，不为私有权重创建虚构的榜单 YAML。

## Discovery runner

在已安装固定 ProphetV2 wheel 的 Python 3.14 环境中运行官方共享入口：

```bash
export MBD_CACHE_DIR=/absolute/path/to/matbench-discovery-data
.venv/bin/python models/run_discovery.py \
  --model prophetv2_local --checkpoint /absolute/path/to/best.pt \
  --dtype float32 --device cuda --matmul-precision highest \
  --max-force 0.05 --max-steps 500 \
  --ase-optimizer FIRE --cell-filter FrechetCellFilter \
  --material-ids /absolute/path/to/pilot-material-ids.txt \
  --out-dir /absolute/path/to/run/pilot --shard-dir /absolute/path/to/run/pilot/shards \
  --n-shards 2 --shard-index 0
```

`--material-ids` 是每行一个 WBM ID 的固定子集，ZIP 读取在成员解析前按 ID 选择。完整评测省略这个参数；另一个分片用相同配置和 `--shard-index 1`。`--dry-run` 使用至多四个结构、每个至多两步，只用于检查调用。

运行同样的命令并把 `--shard-index 0` 替换为 `--merge-shards`，即可严格检查覆盖范围、合并和进行官方 MP2020 修正及形成能评分。每个分片都保存 checkpoint SHA-256 和 FP32 matmul policy，换权重或精度不能复用原有分片，即使该分片已完成。

合并在 `--out-dir/discovery-metrics.json` 保存指标和范围：`n_expected`、MP2020 后形成能有效数 `n_present`、缺失数 `n_missing`、官方异常值过滤后总缺失数 `n_masked_after_cleaning`、收敛数 `n_converged`、弛豫异常数 `n_relaxation_failed`，以及官方指标和运行成本。固定子集标记 `scope=wbm-selected`，`metrics.selected_set` 只评价所选 ID；`metrics.unique_prototypes` 的 DAF 分母使用该子集未舍入参考的稳定比例。全集标记 `scope=wbm-full`，保留官方全量及 unique-prototype 口径。子集不能通过 `--write-yaml` 冒充榜单结果。

这里直接使用准备好的环境执行，环境安装及每轮实验的不可变配置由工作区 `experiments/matbench-discovery/` 管理。通用 wrapper 负责启动、记录和汇总，弛豫、分片、恢复、MP2020 处理与评分仍归官方共享实现。

## 历史源码冒烟测试

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
2026-11-03 测试时 ProphetV2 包声明仅支持 Python 3.12，因此这份历史记录不能证明常规包安装支持。后续正式评测应安装已验证的 Python 3.14 wheel，并保存其版本与哈希。

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
