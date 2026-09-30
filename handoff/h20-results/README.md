# H20 MRV2 + DBO + FULL 实测证据

状态：最终精简代码的 H20 B/E/G、全 rank 执行断言及样本 64 首次分叉诊断
完成。首次分叉由同输入下近似并列的候选 logits 改变解释；不宣称跨模式数值等价。

最终代码：`98fa974`。生产验收日志已删除，执行证据由测试扩展采集。
目标：vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (0.30.0)，
Torch 2.13.0+cu130、CUDA13.0、NCCL2.29.7、H20 SM90。
Base DeepSeek-V2-Lite revision `604d5664dddd88a0433dbae533b7fe9472482de0`。

## 固定比较条件及结果

2A2F DP2 TP1、GSM8K前128题、8-shot、并发12、max_num_seqs8、MBT4096、
max_model_len4096，temperature0。全部128题prompt及generation参数完全相同。
这是一组约定的128题诊断验收，不能声称完成GSM8K全量或GPU全部功能矩阵认证。

| 档位 | 准确率 | 执行证据 |
| --- | --- | --- |
| B eager DP2 | 41/128 = 32.03125% | E2E通过 |
| E eager DBO DP2 | 40/128 = 31.25% | 所有预期 A/F rank，live eager 双microbatch与匹配FFN布局 |
| G FULL DBO DP2 | 43/128 = 33.59375% | 所有预期 A/F rank，连续live FULL双阶段与匹配FFN replay |

最终六个局部测试文件在目标环境 **340 passed**；本地执行器测试 **174 passed**。
作业 4324，目录 `evidence/minimal-20260930T152957`。B/E/G 均超过原 0.27
阈值，未降低阈值。`final-comparison.json` 的 BE/BG/EG 正确性变化分别为
3/4/5 题，文本变化为 49/44/48 题；128 题 prompt 及生成参数逐项一致。
样本 64 此轮 B/G 答 300，E 未匹配答案；仅靠这次结果不能解释其变化。

## Profiler

复用 4124 原有 GPU profiler（生产计算逻辑未改变），四 worker 各采集ProfilerStep#301至#308。
`profile-summary.json`来自实际kernel时间区间合并和交集，不采用CPU调度区间代替。
四trace的baseTimeNanoseconds相同；Attention/FFN各自的device0/1是角色内编号。
跨角色统计配对相同DP rank，对应不同物理GPU。

| DP rank | Attention图内kernels | FFN图内kernels | A/F计算重叠 | FFN NCCL/计算重叠 |
| --- | --- | --- | --- | --- |
| 0 | 6512 | 8000 | 14.4326 ms | 1.3302 ms |
| 1 | 6512 | 8000 | 14.5150 ms | 1.3586 ms |

计算类排除名称含nccl的kernel，统计区间并集以免多stream重复累计。
这证明实际设备交错；没有同口径性能计时，不能据此声称任何加速比。

## 初轮差异及重放（2026-09-29）

初轮 4124 为 B39/E41/G40，各 128 题；当时 341 项局部单测通过。
源码为 e3cb4d2，日志配置修复 b1479d5；其日志探针已被最终测试扩展替代。

`comparison.json`列出全部答案差异及正确性差异。正确性变化集中于0-based
题号18、63、64、90、99、104。BE有6题变化，BG有3题，EG有3题。
文本变化分别46、47、44题。使用原prompt、stop、temperature0、seed1234及
max_tokens512，在同样拓扑及并发12下每题重复4次，保留请求和完整响应。
SLURM4131已完成RB/RE/RG；这是有选择的差异诊断集，不是新增128题准确率结果。

重复重放结果（每格为该题四次请求的正确次数）：

| 0-based题号 | 目标答案 | B | E | G |
| --- | --- | --- | --- | --- |
| 18 | 7 | 3/4 | 1/4 | 2/4 |
| 63 | 1596 | 4/4 | 4/4 | 4/4 |
| 64 | 300 | 2/4 | 3/4 | 0/4 |
| 90 | 225 | 0/4 | 1/4 | 0/4 |
| 99 | 58 | 2/4 | 2/4 | 4/4 |
| 104 | 500 | 4/4 | 4/4 | 4/4 |

完整抽取答案在`replay-summary.json`。`[invalid]`表示未匹配到GSM8K的
`#### 数值`格式，不能解释为引擎崩溃。72次请求均返回完整响应，RE/RG同时
再次通过真实双microbatch及连续FULL/匹配FFN执行断言。

B自身在18、64、99题重复答案变化，证明跨模式差异不能全部归因于DBO/FULL。
63与104的原始正确性差异在重放中消失。64此次G为0/4而B/E为2/4和3/4，
仍是模式相关数值差异，不能用基线也有波动来排除；初轮G在该题回答正确，
故这也不是每次必现的固定错误。现有证据不足以定位到具体算子或证明数值等价。
没有修改阈值、替换权重或绕过错误以消除这一限制。

MRV2 原生双 microbatch、连续真实 FULL 与 FFN replay、设备交错及 128 题
准确率门槛均有实机证据。样本 64 已进一步完成下面的同输入诊断。

## 样本 64 首次分叉的固定输入对照

作业 **4326**，run `evidence/forks-20260930T160735`。沿用原始 8-shot prompt，
12 个并发请求通过 regex 约束生成相同的 23-token 共同前缀，结束于“ left”。
临时诊断设 min/max_tokens=55、ignore_eos，保证测量窗口中请求仍存活；这是
固定输入诊断，不计入 GSM8K 准确率。正常 B/E/G 参数不变。

在同一个已准备的 batch 上依次运行 **FULL → eager DBO → FULL**，复用同一
input、position、KV 请求状态、stage metadata 与 [4,4] 布局（真实 token [4,2]）。
eager 调用原生 UBatchRunner，并让 A/F 均采用 eager 模式；随后恢复 FULL。
前后 token/position 不变，返回原 FULL 输出继续采样。诊断有额外执行与读回，
不作为性能证据，不修改已安装上游或生产源码。

12 个完整响应与 tokenizer 核对了共同前缀；34 个快照的所有真实 token/position
均能对应请求历史，采样的下一 token 也与记录的 FULL argmax 一致。
首次分叉输入位置 **1438**、token **2116（“ left”）**，两 DP rank 的第 22 个
双阶段执行都观察到以下差异（各 rank 的 row2 相同）：

| 候选 | 首次 FULL | eager DBO | 再次 FULL |
| --- | --- | --- | --- |
| token276，“ to” | 28.125 | 28.125 | 28.125 |
| token279，“ in” | 28.125 | 28.250 | 28.125 |
| argmax | “ to” | “ in” | “ to” |

两个 rank 的 row3 同样发生 argmax 交换；合计 4/12 个首次分叉样本行改变首选。
34 个快照的两次 FULL 全词表 logits 最大差值均为 **0**；随着真实 token 和
position 推进，输出也对应更新后的请求历史，未观察到陈旧输入/输出缓冲复用。

这给出了该首次分叉的直接解释：相同输入与布局下，近似并列候选的 0.125
logit 差异改变了选择，后续生成因此走向不同上下文。无需添加 AFD 防御分支。
本诊断未归因到单个底层算子，也不证明所有步骤逐位相同或全量精度无退化；
没有用自由生成后的不同上下文互比 logits 来作此结论。

原始 `D1/requests.json`、`D1/rank*-step*.json`、`D1/tokens.json` 与
`diagnostic/sample64_worker.py`、`sample64_client.py`、`verify.py` 保存在该 run。
本地副本：`/private/tmp/afd-sample64-diagnostic/evidence-4326`。
复核（安装 tokenizers 的本地 Python，MODEL 为固定 revision 的模型目录）：

```bash
python diagnostic/verify.py D1 MODEL/tokenizer.json diagnostic/sample64-forks.json
```

4325 曾在测量窗口末尾只剩 4 个请求时停滞并超时；该诊断把 graph 的 padded
布局强制用于额外 eager 对照，不能据此认定正常执行有同样故障。4326 延长所有
请求的最小生成长度，使整个测量窗口保持两个非空阶段后完成；未改变生产路径。

## 原始证据与清理

H20根目录 `/home/david_cwq/zhouziheng/afd-mrv2-dbo-20260929`。
初轮：`evidence/acceptance-20260929T201853`；重放：`evidence/replay-20260929T205310`。
本地原始样本与trace：`/private/tmp/afd-mrv2-acceptance-evidence`。

4124已COMPLETED，ExitCode0:0。B-1/E-3/G-1-result.json无残留/清理失败/新增共享内存；
controller-final-cleanup.json为空；health-controller-exit确认无模型进程/GPU应用、
24180/24181/1239端口清空及ipcs空。4131同样COMPLETED、ExitCode0:0，21:01:26结束；RB/RE/RG结果清理项均空，
controller-final-cleanup.json为空，退出快照四卡0MiB、无GPU应用/模型进程，
端口及IPC清空。共享模型缓存、其他项目环境均保留。


最终 B/E/G 原始样本在本地 `/private/tmp/afd-mrv2-final-evidence`。
作业 4324 因随后 D1 短请求未形成诊断所需布局而 FAILED，不影响此前三档独立
结果；所有阶段及退出清理通过。4325 的配对诊断保存了 28 个快照后请求超时，
FAILED；退出快照四卡 0MiB，无模型/GPU 进程、任务端口或 IPC 残留。

复算命令（本地执行，输入为上述原始证据目录）：

```bash
python handoff/h20-results/summarize.py /path/to/extracted-run
```

该脚本从 B/E/G samples 生成 `comparison.json`；目录中存在 profiler 时，
同时生成 `profile-summary.json`。已复算并确认与保存的初轮/最终汇总一致。

4326 已 **COMPLETED，ExitCode0:0**，2026-09-30 16:10:56 结束。
D1 与 controller 清理结果均空，退出时无模型进程/GPU 应用、任务端口或 IPC
残留；GPU 仅各 4MiB 驱动记账。`slurm-terminal.txt` 保存最终调度状态。
