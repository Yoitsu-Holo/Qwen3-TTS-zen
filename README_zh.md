# Qwen-TTS-zen

Qwen3-TTS 推理的即插即用加速层。它不重训练、不量化、也不做近似——它消除的是 CPU 侧的
下发（launch）瓶颈，正是这个瓶颈让原版实现无法真正用上它所运行的 GPU。

zen 是自包含的：它在 `zen/_vendor` 下自带了一份上游 `qwen_tts` 库，因此不需要额外
checkout 该包，也不需要安装它的 wheel。加载一律通过 `zen.load_model(...)`。

> 内置副本是**未经改动的上游代码**。`Qwen3-TTS-opt` 这个 fork 唯一的功能性改动，是在
> `talker.forward()` 路径上手写了一个 `_subtalker_decode`——而 zen 完全绕开这条路径，
> 因为 zen 直接替换掉 `talker.generate` 本身。因此那个优化在这里纯属赘余，况且它的替代
> 实现（`zen/code_predictor.py`）本身更快（15 个子步 15.2 ms vs 84.1 ms）。

## 原版路径为什么慢

在 RTX A3000（sms_86）上对真实请求做 profiling 显示，单步耗时几乎完全受限于主机侧
下发，而不是计算：

| 阶段 | 占实际耗时比例 |
|---|---|
| 代码预测器（每帧 15 个自回归子步 × 5 层） | **65%** |
| talker（28 层，每帧一个 token） | 27% |
| codec 解码 + prompt 处理 | <8% |

单个 talker 层在 1 个 token 的序列上做 `2048x6144` 的 GEMM：算术量约 1.3 us。实测约
1.1 ms。约 800 倍的差距，也就是说约 99.9% 都是下发开销。每帧要发起约 1000 次 CUDA
下发；一段 4 秒音频约 65 000 次。

## zen 做了什么

两处改动，都完整保留权重：

1. **重写 talker 解码步**，成为一个静态形状的模块（`zen/talker.py`），自带 RoPE 表、
   由 `index_copy_` 写入的扁平 KV cache、以及预分配的加性 mask。之后在
   `torch.compile(mode="reduce-overhead", fullgraph=True)` 下编译，把整个 28 层步骤
   捕获成一张 CUDA graph——一次 replay 替代数百次下发。

2. **代码预测器的 15 个子步以同样方式捕获进一张图**（`zen/code_predictor.py`）。

prompt 构造、codec、分词和 batching 都没有改动：zen 只接管 `talker.generate`，其余
全部留给原版库（`zen/__init__.py`）。

### 为什么捕获能成立

原版模块在四个方面破坏了 CUDA graph 捕获；zen 逐条规避：

| 原版行为 | 为何破坏捕获 | zen 的做法 |
|---|---|---|
| `cache[:, :, pos:pos+1] = k` | 依赖数据的切片，dynamo 无法对 `pos` 建立 guard | 用位置张量做 `index_copy_` |
| 每一步 `create_causal_mask(...)` | 在图内构造新的 mask | 单个预分配 mask 缓冲，形状在外面定好 |
| `@dynamic_rope_update` + 每步 `cos`/`sin` | Python 分支、主机侧标量计算 | 预计算表 + `index_select` |
| `StaticCache` 在图内惰性初始化 | cache 被登记为图的*输出*，其后续 reset 会以 "output has been overwritten" 失败 | 在编译前就立即初始化 cache |

还有三条要求容易漏掉：

1. **每次图调用都包在 `torch.no_grad()` 里。** 若开启梯度，输出会为一次永不发生的
   反向传播而保持存活，cudagraph_trees 也会拒绝走它的快路径。
2. **每个图输入都必须是原地更新的预分配缓冲。** 每步喂一个新张量会改变其地址；
   talker 步实测 41.4 ms（稳定缓冲）vs 96.8 ms（新张量）。
3. **输入 *stride* 同样要 guard。** 真实调用方传入的是非连续视图
   （`last_hidden_state[:, -1:, :]`）；直接把它交给已编译的代码预测器，会在首个请求时
   触发重新编译。`CodePredictorDecode.load_inputs` 在编译调用*之外*先拷贝进固定缓冲。

### 窗口大小

`ZenTTS(max_frames=N)` 决定 RoPE 表、扁平 KV cache 与 mask 缓冲的大小，覆盖
**prompt + 生成帧**。解码循环从位置 `S`（prompt 长度）开始，所以 `_zen_run` 最多产出
`N - S` 帧。这里搞错是致命的而不是变慢：`pos = S + frame` 越过表尾会触发
`indexSelectSmallIndex`，以 device-side assert 杀掉 worker。现在 `TalkerStep._mask`
改为在主机侧抛 `IndexError`，而当 prompt 本身就填满窗口时，`_zen_run` 会提前抛
`ValueError`。

### 仍然是 eager 的部分

talker 的 `GenerationMixin.generate` 脚手架与 codec 以 eager 运行；只有内部模块调用
被捕获。这是本设计的下限——再往下就必须整体替换 `generate`，而 zen 有意避免这么做，
以便 prompt 语义仍由该库负责。

## 用法

```python
import zen

# zen 自己解析内置的 qwen_tts；无需导入该包。
model = zen.load_model(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)

zen.install(model, compile=True, max_frames=3072)   # 编译并捕获两张图

wavs, sr = model.generate_voice_design(text="...", instruct="...", language="Chinese")
```

`max_frames` 是 RoPE/KV/mask 的总窗口，覆盖 **prompt + 生成帧**——取值应当明显超过
你打算发送的最长 prompt。`zen.install` 返回 `ZenTTS` 句柄，并在两张图编译期间阻塞约
70 秒，此后**每一个**请求——包括第一个——都以稳态速度运行。`zen.uninstall(model)`
恢复原版路径。

### 约束：单线程

捕获到的图只能从捕获它的那个线程 replay（cudagraph_trees 把它的树管理器放在线程局部
存储里）。因此服务端必须拥有一个专用推理线程：install、warmup 以及所有请求都在同一
线程上。`server.py` 用一个单 worker 的 `ThreadPoolExecutor` 做到这一点。

## 实测

voice-design，1.7B bf16，RTX A3000 Laptop，相同 prompt 与 seed：

| | 实际耗时 | 音频时长 | RTF |
|---|---|---|---|
| 原版 | 5.38 s | 3.76 s | 1.43 |
| zen | 2.06 s | 4.16 s | **0.50** |

首个请求即等于稳态。收益来自压缩下发与同步开销，而不是让算术更快：解码步是带宽受限的，
每帧要读 talker 的 2.82 GB 权重，而实测内存带宽约 207 GB/s——每步约 13.6 ms 的
流量不可避免，实测约 15.6 ms。

各组件的数字：

| | eager | 捕获后 |
|---|---|---|
| talker 步（28 层） | 25.7 ms | 16.6 ms |
| 代码预测器（15 个子步） | 84.1 ms | 15.2 ms |

### 可复现性

两条路径在固定 seed 下各自是确定性的，但彼此**不**一致，而且差异**并非**只是采样噪声。

贪婪解码（不采样、单一固定 seed、长 prompt），分别在独立进程中运行：

| 路径 | 帧数 | 码序列 sha1 |
|---|---|---|
| eager | 1151 | `1fda53fc1448b80a` |
| compiled | 1151 | `e32af0942e570c44` |

长度相同，内容不同。贪婪解码消除了采样阈值这个放大器，所以如果差距只是采样噪声，两者
就应当一致；而它们并不一致。剩下的差异来自 `torch.compile` 的融合改变了前向中 bf16 的
舍入点。它按路径是确定性的，听感上也微弱到人耳判定输出可接受，但这意味着**编译后的输出
并非 eager 的逐位复现**——应当把它们视为两个密切相关的变体，而不是同一次计算。

本文件早前的版本声称该分歧是"采样噪声"，并引用了 0.46 ms 的 talker 步。两者都错了：
采样之说只比较了帧数而没有比较内容；0.46 ms 的探针从未推进 `pos`，测的是一条退化路径。
真实的步耗时约 15 ms。

## 目录结构

```
zen/rope.py           预计算 RoPE 表（交错式 mrope 在文本场景退化为 1-D）
zen/talker.py         静态形状的 28 层解码步，图编译
zen/code_predictor.py 15 子步的代码预测器解码，图编译
zen/pipeline.py       ZenTTS：缓存、mask、warmup、prefill 吸收、采样器
zen/__init__.py       load_model / install / uninstall + talker.generate 的替换实现
zen/_vendor/qwen_tts/ 上游 qwen_tts，未改动（模型类、codec、推理封装）
tests/                对齐性、组件级与端到端检查
```

## 测试

```
python tests/test_talker.py         # TalkerStep 稳定性 + 捕获带来的加速
python tests/test_code_predictor.py # 每个子步的 logits 与 _subtalker_decode 对齐
python tests/test_window.py         # prompt+解码始终留在 RoPE/KV 窗口内
python tests/test_e2e.py            # zen vs 原版端到端：码匹配、实际耗时、RTF
```
