# GPU MRV2 + DBO + FULL CUDA Graph 接续说明

## 当前状态（2026-09-30）

代码分支 `codex/gpu-mrv2-dbo`，最新已推送 `98fa974`。
最终代码在 H20 的 340 项局部单测及 B/E/G 均通过。样本 64 的固定输入对照
已解释首次分叉：近似并列的候选 logits 差 0.125 即改变 argmax，FULL 再次
重放完全复现。未发现该步输入或 metadata 刷新错误；不宣称跨模式数值等价。

| 档位 | 场景 | GSM8K 前 128 题 |
| --- | --- | --- |
| B | afd-v2-eager-dp2 | 41/128 |
| E | afd-v2-eager-dbo-dp2 | 40/128 |
| G | afd-v2-graph-dbo-dp2 | 43/128 |

三档输入与生成参数一致。E/G 已通过所有预期 Attention/FFN rank 的执行断言；
G 要求连续真实双阶段 FULL，两个阶段均有真实 token，FFN 有匹配 replay 布局。
这是评测窗口内的布局覆盖匹配，不宣称 A/F 日志逐事务精确配对。
原始 profiler 已证明设备计算交错，未测同口径加速比。
完整结果与限制见 [h20-results/README.md](h20-results/README.md)。

## 范围与必要实现

原始要求是在 `update/v0.30.0` 上支持 GPU ModelRunnerV2 + DBO + graph。
本次只做 GPU；不新增 NPU DBO，不破坏已有 NPU、MRV1、非 DBO 和共享功能。
不创建上游 PR，不修改已安装上游源码、共享环境或模型缓存。

相对快照 `73ced14`，生产增量为 **+130/-6，净增 124 行**，仅涉及：

- `attention_metadata.py`：为原生阶段安装独立 AFD metadata，保留全批 offset、
  padded 传输长度与真实 token 长度，避免阶段间串用状态。
- `attention_model_runner_v2.py`：在原生 prepare 后发布 A/F 控制，复用原生
  microbatch 输入、线程和 stream；capture 配对阶段数，FULL 不重复发送控制。
- `validation.py`：开放已支持的 GPU 双 microbatch，限制 Attention DP > 1。

新增的生产验收日志及计数状态已全部移除，FFN 生产文件相对快照无增量。
`tests/e2e/mrv2_evidence.py` 仅由 B/E/G 显式使用原生 worker-extension 入口加载，
记录完成后的原生调用，不增加 worker 类、device synchronize、barrier 或执行框架。
连续性与全 rank 检查留在测试侧。不能再用单个 rank 的任意一次匹配代表全体通过。

## 固定版本和运行条件

- 交接起点：`1b89f19a6f0653d03543c5a5d35e7f98f4099115`。
- AFD 来源：`vllm-project/afd-plugin:update/v0.30.0`，
  `4f74787e80da29d9f06adcbd14c03aa002dcc3f8`；任务仓库是源码快照历史，不强求 merge-base。
- vLLM：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`，0.30.0 / `gced6857af`。
- Torch 2.13.0+cu130、CUDA 13.0、NCCL 2.29.7、H20 SM90。
- Base DeepSeek-V2-Lite revision `604d5664dddd88a0433dbae533b7fe9472482de0`。
- 2A2F、DP2、TP1、同步 P2pNccl，`compute_gate_on_attention=false`。
- GSM8K 前 128 题、8-shot、并发 12、seq8、MBT4096、model_len4096、max_tokens512、
  temperature0；G 使用 FULL_DECODE_ONLY。准确率阈值仍为 0.27。

## 运行与证据入口

本地代码 `/private/tmp/afd-mrv2-dbo-handoff-20260929`。
H20 根目录 `/home/david_cwq/zhouziheng/afd-mrv2-dbo-20260929`，下文路径相对此目录：

| 作业 | 目录 | 结果 |
| --- | --- | --- |
| 4124 | evidence/acceptance-20260929T201853 | 初轮 B39/E41/G40，profiler；COMPLETED，清理通过 |
| 4131 | evidence/replay-20260929T205310 | 6 题各 4 次重放；COMPLETED，清理通过 |
| 4324 | evidence/minimal-20260930T152957 | 最终 340 单测、B41/E40/G43 通过；随后的短请求诊断未形成目标布局，作业 FAILED；清理通过 |
| 4325 | evidence/forks-20260930T155121 | 保存 28 个固定输入 Graph/eager/Graph 对照；请求超时，FAILED；清理通过 |
| 4326 | evidence/forks-20260930T160735 | 首次分叉固定输入诊断通过，12 个完整响应、34 个对照快照；COMPLETED，清理通过 |

4326 在两 rank 的 position1438、token2116（“ left”）处均观察到候选
“ to”/“ in”交换；4/12 个对应样本行的 argmax 在 Graph/eager 间不同。
12 个响应的完整共同前缀 tokens 已核对，所有保存的真实 token/position 均与
请求历史一致，34 个 Graph/eager/Graph 快照中两次 Graph 的最大差值均为 0。
临时诊断源码及复核脚本固定保存在该 run 的 `diagnostic/`，不属于生产实现。

本地初轮样本和 profiler：`/private/tmp/afd-mrv2-acceptance-evidence`；
最终 B/E/G 样本：`/private/tmp/afd-mrv2-final-evidence`。
`h20-results/summarize.py` 从原始 samples/trace 生成汇总；最终对照在
`h20-results/final-comparison.json`。不把巨大 trace 放进 Git。

服务器操作遵守 `~/.codex/references/server-health.md` 和 `h20-slurm.md`：
只用 `ssh vllm-h20-head`、david_cwq、SLURM h20；计算节点仅在有效分配内访问。
Desktop 持久 SSH 保持关闭，保留 SLURM 的 CUDA_VISIBLE_DEVICES。
每次使用独立输出目录、有界四卡任务和同一个 controller.lock；失败先读日志与
核验精确清理，再按具体修复续跑。sacct 不可用，终态用 scontrol 和保存日志确认。
不按用户名清理，不动共享账号其他任务。H200 d99401f0 已取消、2690692/2690694
退出且未启动模型，不恢复 H200。

## 变更与失败历史

- 原交接 141 项 CPU/mock 单测只证明局部控制逻辑，不能代替 GPU 验收；原日志保留。
- 4113/4115/4116 分别修复预检权限、文件软限制、控制器 Python pidfd 兼容问题。
  环境和固定模型/GSM8K 在隔离 runtime 与指定缓存中准备完成。
- 4124 E-1 因日志被过滤而断言失败，b1479d5 修复测试日志配置；E-2 因复用 NFS
  basetemp 删除失败而未运行模型，保存旧目录后 E-3 成功。后续始终使用新目录。
- 28353bf 增加全 rank 验收；1d66d60 删除生产探针，将必要证据留在测试侧。
- 4322 是前一观测候选，通过并清理；4323 在模型启动前因 worker 类别名被拒绝，
  98fa974 改用原生 worker-extension 入口，未放宽生产校验。
- 4324/4325 的失败仅来自临时数值诊断，保留其原始日志与独立阶段结果。
  4325 退出后四卡 0MiB、无模型/GPU 应用、任务端口及 IPC 残留。

更早的完整迁移记录保留在 Git 的 `95dc727:handoff/GPU_CONTINUE.md`。
本阶段约定验收已闭环；结论限于上述 H20/模型/拓扑和首次分叉诊断。不要扩大
为所有 GPU、模型、NPU 或逐题数值等价。受影响检查已通过，不继续扩展实验矩阵。
