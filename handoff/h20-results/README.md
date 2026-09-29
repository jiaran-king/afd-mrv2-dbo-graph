# H20 MRV2 + DBO + FULL 实测证据

状态：约定的H20双microbatch/FULL功能与128题准确率阈值验收通过。
差异重放完成；逐题数值等价不成立，不应据此声明无模式相关数值差异。

源码：GPU执行逻辑 `e3cb4d2`，E2E日志配置修复 `b1479d5`。
目标：vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (0.30.0)，
Torch 2.13.0+cu130、CUDA13.0、NCCL2.29.7、H20 SM90。
Base DeepSeek-V2-Lite revision `604d5664dddd88a0433dbae533b7fe9472482de0`。

## 固定比较条件及结果

2A2F DP2 TP1、GSM8K前128题、8-shot、并发12、max_num_seqs8、MBT4096、
max_model_len4096，temperature0。全部128题prompt及generation参数完全相同。
这是一组约定的128题诊断验收，不能声称完成GSM8K全量或GPU全部功能矩阵认证。

| 档位 | 准确率 | 执行证据 |
| --- | --- | --- |
| B eager DP2 | 39/128 = 30.46875% | E2E通过 |
| E eager DBO DP2 | 41/128 = 32.03125% | live eager双microbatch，real_tokens=[3,3]，双FFN匹配布局 |
| G FULL DBO DP2 | 40/128 = 31.25% | live FULL双阶段，real_tokens=[4,2]，连续21次及双FFN匹配replay |

六个局部测试文件在目标环境341 passed；日志修复后的E2E执行器测试175 passed。
B/E/G均超过原0.27阈值，未降低阈值。模式间输出不逐字一致。

## Profiler

复用原有GPU profiler，四worker各采集ProfilerStep#301至#308。
`profile-summary.json`来自实际kernel时间区间合并和交集，不采用CPU调度区间代替。
四trace的baseTimeNanoseconds相同；Attention/FFN各自的device0/1是角色内编号。
跨角色统计配对相同DP rank，对应不同物理GPU。

| DP rank | Attention图内kernels | FFN图内kernels | A/F计算重叠 | FFN NCCL/计算重叠 |
| --- | --- | --- | --- | --- |
| 0 | 6512 | 8000 | 14.4326 ms | 1.3302 ms |
| 1 | 6512 | 8000 | 14.5150 ms | 1.3586 ms |

计算类排除名称含nccl的kernel，统计区间并集以免多stream重复累计。
这证明实际设备交错；没有同口径性能计时，不能据此声称任何加速比。

## 逐题差异及重放

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

验收结论限于约定范围：MRV2原生双microbatch、连续真实FULL与FFN replay、
设备交错及128题准确率>=0.27均有实机证据。若发布标准要求逐题相同或排除
任何模式相关精度退化，数值等价仍未通过，需要另行做固定batch/逐token logits
对照；本报告不将该更强结论标为通过。

## 原始证据与清理

H20根目录 `/home/david_cwq/zhouziheng/afd-mrv2-dbo-20260929`。
初轮：`evidence/acceptance-20260929T201853`；重放：`evidence/replay-20260929T205310`。
本地原始样本与trace：`/private/tmp/afd-mrv2-acceptance-evidence`。

4124已COMPLETED，ExitCode0:0。B-1/E-3/G-1-result.json无残留/清理失败/新增共享内存；
controller-final-cleanup.json为空；health-controller-exit确认无模型进程/GPU应用、
24180/24181/1239端口清空及ipcs空。4131同样COMPLETED、ExitCode0:0，21:01:26结束；RB/RE/RG结果清理项均空，
controller-final-cleanup.json为空，退出快照四卡0MiB、无GPU应用/模型进程，
端口及IPC清空。共享模型缓存、其他项目环境均保留。
