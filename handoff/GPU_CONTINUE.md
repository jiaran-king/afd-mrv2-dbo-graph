# GPU MRV2 + DBO + FULL CUDA Graph 接续说明

## 范围与状态

用户已将当前范围调整为 **只支持 GPU**。交接仓库
`jiaran-king/afd-mrv2-dbo-graph` 已由用户改为公开；2026-09-29 用户授权在
独立 `codex/` 分支提交并推送续做工作。不创建上游 PR，不改仓库可见性。

这是正在实现的代码，**不是已通过实机验收的支持版本**。开发分支已定向开放 GPU validation
（两个 microbatch，Attention DP > 1）；NPU 仍拒绝 DBO。配置入口可用不代表
实机通过，必须完成下述核验。
本次增量不包含 NPU 专属修改；源码基线中已有 NPU 功能保持原样。

## 固定版本

- AFD 基线：vllm-project/afd-plugin，update/v0.30.0，
  `4f74787e80da29d9f06adcbd14c03aa002dcc3f8`。
- vLLM 目标：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`（0.30）。
- GPU 环境、CUDA、PyTorch、NCCL、模型权重版本尚需在目标机器实测固定。
- 交接仓库用两个普通提交保存上游源码快照与 GPU 增量，未修改公开上游。
  基线快照不包含完整上游 Git 历史；首个提交的 tree 对应上述 AFD SHA。

## 已实现

1. `attention_model_runner_v2.py`：实例级临时包装原生 UBatchRunner.prepare。
   原生 MRV2 继续负责输入、microbatch 切分、attention metadata、工作线程、
   stream 交接和输出合并；AFD 从返回的 UBatchState 安装 sidecar 并发送控制。
2. `attention_metadata.py`：为每个 stage 克隆独立 metadata；保留全批 token /
   request offset，分别记录 padded 和真实 token 长度，包括全 padding 尾阶段。
3. 多阶段 capture 通过同一 event tracker 配对 warmup/capture；descriptor 校验
   包含阶段数。双阶段 FULL replay 复用 prepare 时发送的完整控制，避免误发
   一个单阶段控制或重复发送。scope 退出恢复实例方法和 runner 状态。
4. `test_model_runner_v2.py`：阶段隔离、padding、控制去重、capture 模式、相同
   token 数不同阶段布局、异常恢复等局部回归。

生产增量只涉及上述两个 runner/metadata 文件；FFN 使用既有 daemon 和 graph
cache，不重写 FFN 执行器。公共 mixin 的新方法仅由 MRV2 新入口调用。

## 已知未完成项

- GPU 上 B/E/G 均未运行，不能将 CPU/mock 测试视为 CUDA/NCCL 或模型精度证据。
- 开发分支已定向开放 GPU validation，并补 MRV2 + DBO E2E 场景；尚待实机运行。
  旧 `afd-graph-dbo-*` 场景属于 MRV1，不能拿它们作为本目标通过。
- FFN graph cache 已以 stage ID 与各 DP token 数构成 key，代码层面可区分
  单/双阶段；仍需验证真实 A/F 两端连续 replay、输入刷新与通信匹配。
- 需确认模型 proxy 的 yield 与原生 UBatchContext 的实际计算/通信交错，不能
  通过同步或串行两个 batch 来替代 DBO。
- 检查 graph memory profiling → 正式 capture 的再次捕图、状态恢复与 FFN
  图缓存生命周期；检查 dummy/profile/idle DP、阈值边界和合法 fallback。
- 检查 GPU profile 标记透传是否需要补齐；当前此次增量没有改变既有行为。

## 目标机器先做的检查

使用已有 GPU 容器、依赖及权重。核对当前 vLLM SHA 是否与目标一致，勿静默
升级依赖或修改已安装上游。记录 `nvidia-smi`、torch/CUDA/NCCL 版本与占用。
代码安装应沿用环境既定流程，避免 `uv sync` 自动替换锁定运行时。

在仓库根目录运行局部测试（需要可导入的目标 vLLM 和 torch）：

```bash
OMP_NUM_THREADS=1 python -m pytest \
  tests/unit/v1/worker/test_model_runner_v2.py \
  tests/unit/v1/worker/test_dbo.py \
  tests/unit/v1/worker/test_ffn_model_runner.py \
  tests/unit/v1/worker/test_cuda_graph.py -o addopts='' -q
```

已有 no-DBO MRV2 eager 基线入口（需四张容量足够的 GPU）：

```bash
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_GPU_E2E_MODEL=/absolute/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dp2]'
```

上面只是已存在的启动/基线入口，不是最终精度矩阵。最终 B/E/G 固定同一模型、
拓扑、题集、解码、并发与缓存条件，仅改变 DBO/graph 设置。采用现有题集与
判定标准，事先固定样本数；不要直接比较默认 7 题 baseline 和 24 题 DBO。

## 余下执行顺序及验收

1. 完成 GPU 必经接口检查，基线不正确先定位；复用上游 #50945 / #51700
   在固定版本中的最终实现，不搬 MRV1 的完整循环。
2. 通过已开放的正常配置入口核验同步 P2pNcclAFDConnector 的 eager DBO，
   修复实机暴露的必要缺口。至少 Attention DP > 1，确认原生 ubatch runner 已创建，
   真实步 num_ubatches == 2；不得只看 enable_dbo 参数。
3. 补 MRV2 DBO eager / FULL_DECODE_ONLY graph 场景，验证多次动态输入的 FULL
   replay。保持 A/F stage、shape、dtype、DP 控制与通信顺序一致。
4. 同配置 B（无 DBO eager）/ E（DBO eager）/ G（DBO graph）精度对照，保存
   题目级输出、accuracy、异常、超时，定位明显下降。补一段原生 profiler
   trace，确认计算/通信交错；性能不设固定提升百分比门槛。
5. 审查完整增量、更新既有文档，交付可复现命令及实际模式证据。

只在 GPU eager DBO、真实双阶段 FULL CUDA Graph replay、精度与交错证据全部
成立时称当前 GPU 目标完成。合法 fallback 可保留，但不能把所有 live DBO
改成 eager，也不能静默切回 MRV1。共享专家、量化和其他已有路径不得退化。

NPU 已移出当前目标。若未来恢复 NPU，固定 Ascend 原生模型状态明确拒绝
ubatch_idx > 0；还存在 stage builder/seq_lens_np、graph 参数更新及设备接口
缺口，需要另行评估，不能据本次 GPU 代码宣称 NPU 已支持。

## 本次迁移前复核

最终 GPU-only 版本运行上面的四个单测文件：**141 passed，0 skipped**，日志见
[cpu-unit-tests.txt](cpu-unit-tests.txt)。运行环境是可导入固定 vLLM/torch 的
Ascend 容器，执行内容仅为 CPU 逻辑/mock；没有分配 GPU/NPU、加载模型或实际
捕图。因此此结果证明的是上述局部控制逻辑及既有 FFN/graph policy 回归，
不能证明 GPU eager、DBO overlap、CUDA Graph 或精度已经通过。


## 2026-09-29 H200 接手进展

- `handoff_sha`: `1b89f19a6f0653d03543c5a5d35e7f98f4099115`。
- 开发分支：`codex/gpu-mrv2-dbo`；未改原工作目录，未创建上游 PR。
- H200 默认 Python 的 vLLM 已为 0.28.0，不用于本目标。可复用已安装的
  `/data/zhouziheng/pr407-gpu-regression-20260929/runtime`；其 vLLM 版本文件
  为 `0.30.0` / `gced6857af`。实际 import 路径仍由分配内的运行时预检确认。
- 本任务远程目录：`/data/zhouziheng/afd-mrv2-dbo-20260929`，与 #407 隔离；
  仅复用其运行依赖，不修改 #407 的代码、安装包或实验现场。
- 本地 Python 3.12：`tests/unit/test_e2e_runner.py` **171 passed**；
  `tests/unit/config/test_validation.py` + `tests/unit/v1/worker/test_cuda_graph.py`
  **45 passed**。新增配置测试只隔离模型注册模块的运行时 import，执行真实
  GPU/NPU validator 和 graph policy；不构成运行时集成证据。
- 新场景及默认受控评测条件见 `tests/e2e/README.md`。本次还未得到 GPU
  B/E/G 结果，不能宣称 DBO eager 或双阶段 FULL replay 已通过。

### 待运行的 B/E/G 命令

以下命令必须在完成服务器健康检查、取得四卡 `gpu run` 配额后执行；不要在
未分配的 SSH 会话中直接加载模型。它们是待执行命令，不是通过记录。先运行 B，
再分别运行 E、G 并检查结果。MRV2 已独立检查评测窗口内的真实双阶段 eager/FULL
及匹配的 FFN 模式和 DP 布局，G 还要求连续 replay；仍需检查题目级差异和设备
trace，不能只凭 pytest 成功就宣布最终验收。

```bash
RUNTIME=/data/zhouziheng/pr407-gpu-regression-20260929/runtime
export PATH="$RUNTIME/bin:$PATH"
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES="$CUDA_VISIBLE_DEVICES"
export AFD_GPU_E2E_VLLM_BIN="$RUNTIME/bin/vllm"
export AFD_GPU_E2E_MODEL=/data/models/hub/models--deepseek-ai--Deepseek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0
export HF_HOME=/data/zhouziheng/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export AFD_GSM8K_LIMIT=128
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# 在本任务代码目录运行；pytest 的独立输出目录应位于本任务 evidence 下。
python -m pytest -q -s 'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dp2]'
python -m pytest -q -s 'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dbo-dp2]'
python -m pytest -q -s 'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-dbo-dp2]'
```

本轮 H200 调度请求 `408d77ed`（1 GPU，等待上限 10 分钟，运行上限 15 分钟）
返回 `wait timeout exceeded`，未获得配额，预检脚本和四个运行时单测均未启动。
控制日志与调度记录位于远程 `evidence/unit-controller.log`、
`evidence/scheduler-queued.txt`、`evidence/scheduler-after.txt`。
资源可用后需先运行交接单测，再进行 B/E/G 实机验证；不跳过这些步骤。


用户要求继续实机验收后，已于 2026-09-29 重新提交唯一四卡任务 `d99401f0`，
控制进程 `2690692`，排队上限 6 小时、执行上限 4 小时。任务先做运行时单测，
再按 B/E/G 顺序执行，G 复用已有 profiler 采集 8 个 active steps。日志入口为
`evidence/acceptance-controller.log`，分配后 `evidence/current-run.txt` 指向
本次目录。失败先核验清理并读日志，最多允许两次有实际修复的定向重试。
本对话已启用自动跟进 `gpu-dbo`，持续检查进展和处理结果，资源等待不视为完成。

新增执行证据只在原有 runner 中累计计数并稀疏记录（初始两次、之后每128次），
不加 device synchronize、barrier、线程锁或新执行器。Attention 的 FULL 标志
来自成功调用实际 `run_fullgraph`，live/dummy/profile 分开；FFN 记录实际选择的
模式和 graph key。命令/证据解析单测本地 **174 passed**；运行时新增日志测试
尚待本次 H200 配额执行。这些更改不构成 GPU 实测通过声明。

## H20 迁移（用户最新指令）

用户随后要求切换到 H20。H200 的 `d99401f0` 已由 `gpu cancel` 精确取消，
PID 2690692/2690694 均已退出；H200 没有启动模型，不再提交本任务的 H200 作业。
自动跟进 `gpu-dbo` 已改为 H20 SLURM 路线。

- 登录：`ssh vllm-h20-head`，账号 `david_cwq`，不启用 Desktop 持久SSH连接。
- 远程目录：`/home/david_cwq/zhouziheng/afd-mrv2-dbo-20260929`。
- 当前已提交作业：**4115**，`h20` 分区，4 GPU、32 CPU、256G 内存、4小时；
  不限制具体节点，由SLURM分配。此前4113仅运行轻量预检，因可选目录扫描权限
  错误退出，未启动模型；问题已修复，4113已结束。
- 运行脚本：`afd-mrv2-h20.slurm` → `start-acceptance.sh` →
  `afd-mrv2-hardware-controller.py`。环境准备、下载、单测及模型运行均在分配内。
- 隔离环境：本任务 `runtime/`，安装目标 `vllm==0.30.0` 后断言
  `_version.__commit_id__ == 'gced6857af'`，核对实际 import 路径和 SM90。
  不使用已有其他版本来替代验收，不修改其他项目环境。
- 模型仍为 base DeepSeek-V2-Lite，固定 revision
  `604d5664dddd88a0433dbae533b7fe9472482de0`；不拿现场 Chat 权重替代。
- 最新日志从 `evidence/slurm-4115.log`、`evidence/job-4115/install.log`、
  `runtime-check.json`、`assets.log` 开始；实测启动后 `evidence/current-run.txt`
  指向本轮 B/E/G 目录。SLURM 分配的 CUDA_VISIBLE_DEVICES 保留，AFD 角色仅在
  分配设备集合中拆分。共享账号其他交互和 issue398 作业均保持不动。

迁移及提交SLURM作业不等于GPU验收通过。持续跟进环境准备和真实运行结果，
失败按日志定位并作有界修复，最后仍需完整的准确率、双阶段FULL及交错证据。
