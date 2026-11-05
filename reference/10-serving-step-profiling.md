# BW1100 全模型推理 profiling 流程

这份指南用于分析真实服务中 MoE、DSA、KDA 等模块的设备工作。先证明工具能记录目标，再用短请求估计 profiler 的扰动，最后才决定是否采集完整负载。

单篇副本只包含本文与新增辅助文件；它是所属 skill 的摘录，不是完整安装包。权威版本是本仓库中的完整 skill。
本指南是这项工作流的权威入口。项目运行脚本仍负责资源申请和服务生命周期。这里的探针不能替代 GPU 租约、模型正确性验证或无 profiler 的性能基线。

## 先看当前结论

截至 2026-10-30，已观察到以下结果：

- 固定镜像和 node2 上，一个包含 16 节点和 128 节点图的旧版探针记录了 1589 行。已知模型程序应产生 792 次乘法和 792 次加法；另外 5 行是初始化活动。此结果只证明那次玩具图的 kernel 计数。
- 本指南的 graph 探针在同一计算主体上增加了两个 CPU ROCTX 标记。这一修改尚未在 GPU 上验证。event 探针也尚未在 GPU 上验证。
- 某些 DTK torch profiler 和 hipprof trace 会漏掉 HIP Graph 节点。真实 decode trace 中，MoE kernel 可以因此完全缺失；不能解释为 MoE 耗时为零。
- 当前 bw1100 固定 serving 镜像可从 host 只读挂载 <code>/opt/dtk</code> 并使用 <code>xsys --kernel-trace</code>。2026-10-30 的单卡、8 worker Graph 和完整 TP1/DP8/EP8 服务均记录到 Graph Task 节点。完整服务短窗口有 8 个可解析 worker CSV、839,560 个 Task，且目标 DeepEP、专家 GEMM、DSA 和 KDA 名称均存在。它证明工具覆盖，不证明端到端模块占比或采集开销。
- rocprofv2 的固定输出名 <code>-o k</code> 使多个进程使用同一文件名。多进程采集省略该参数，并核对每个文件名中的 PID 与每行 PID。
- 连续全模型 kernel trace 曾在 60 分钟预算内未完成 8 条长请求。结束时留下 8 个 worker 的约 1.7 GiB 原始 kernel CSV。8 条请求都在达到 1024 个输出 token 前终止；流式解析时 6 个 CSV 出现字符串未闭合错误，另有 1 行无效；其中一个文件还包含 PID、GPU、队列、dispatch ID 与起止时间均相同的重复事件。该次没有通过工作负载或 trace 完整性验收，也没有可作为原生推理占比的结果。退出码为 137；达到预算后仍需显式停止本次服务并核验 GPU 资源，不能把超时当作清理成功。
- 本次成功接收了停止和恢复 profiler 的接口请求，但没有证明该接口能消除底层 kernel tracing 的扰动。没有匹配的真实文本无 profiler 对照，不能计算其端到端开销倍数。
- 已核验的编译器子进程环境过滤只适用于固定镜像中三个已检查的 ELF 路径。CPU 测试确认该边界保留其他 preload，并生成字节相同的代表性 C 动态库和 gfx938 object。8 个真实模型 worker 的 profiler 加载收据也已通过。它不是模型数值或全模型 profiling 的验收。

本轮原始记录位于 bw1100 的 <code>/data/testuser01-b1/glm53-profiling-20261030-1114</code>。其中 <code>profile-clean/realtext/summary.json</code>、<code>profile-clean/rocprof/</code>、<code>profile-clean/analysis-run.log</code>、<code>profile-clean/final-resource-check.json</code> 和 <code>profile-clean/final-kfd-pids.txt</code> 分别保存请求状态、原始 trace、解析失败以及资源检查。不要覆盖这些记录。14:40 的本轮退出检查通过；此后已有其他 GPU 工作，下一次必须重新检查。

## 1. 冻结问题和负载

在新运行目录中记录这些字段，再申请资源：

1. 模型配置和权重身份、镜像 ID、源码与本地修改、驱动、HCU 与 PCI 映射。
2. TP、DP、EP、实际 rank、专家通信后端、graph 模式、KV dtype 和 MTP 配置。
3. 每条请求的输入长度、已缓存长度、输出上限、输入 token ID 哈希、采样设置及请求数。
4. 需要回答的问题、时间边界、warmup、采样时段、停止条件、最大运行时长和磁盘预算。
5. 输出长度、HTTP 状态、输出 token ID、重复输出、needle 或任务专属正确性检查。记录所有失败和未完成请求。

把 prefill、纯 decode 和混合批次分开。全局并发不是本地 batch。MTP 的 draft 也会执行 DSA/MLA 和 MoE；它不等同于主模型的一个 decoder layer。

固定镜像中主模型配置当前列出 45 层：34 个 KDA attention、11 个 DSA attention，MLP 为 42 个 sparse 和 3 个 dense。旧笔记中的 33/12 与该配置不一致。跨版本重新读取配置，不要按旧层数外推。

以下是检查配置的 CPU 代码。它读取当前节点上的配置，不证明历史运行使用了同一权重字节。

~~~bash
MODEL=/data2/GLM-5.3-Flash-Channel-FP8-w8a8
python3 - "$MODEL/config.json" <<'PY'
import collections, json, sys
d = json.load(open(sys.argv[1]))
d = d.get("text_config", d)
print("attention:", dict(collections.Counter(d["layer_types"])))
print("MLP:", dict(collections.Counter(d["mlp_layer_types"])))
print("vocab_size:", d["vocab_size"])
PY
~~~

## 2. 先完成无 profiler 对照

用同一模型、输入、资源和 serving 参数跑真实文本。保留逐请求记录、成功及失败数、输入与输出长度、实际输出 token ID、TTFT、TPOT 和 MTP acceptance。检查输出质量后再解释耗时。测试不同版本时交替运行并保留原始样本；不要把不同负载或不同机器的数直接比较。

<code>prefix_repetition</code> 可以用于固定的合成控制，不能自动代表正常文本生成。本轮 88 条合成请求 HTTP 和长度均成功，但对生成文本重新分词后，77 条触发重复文本诊断。该诊断与服务直接返回的 token ID 验证不同；它足以阻止把这组结果当作正常文本的性能基线，不能用于宣称模型数值错误。

如果对照缺失、输出被截断或正确性未知，保留原始数据并标明未知。不要启动大规模 profiler 后才补这个判断。

## 3. CPU 预检编译器环境

先确认本次镜像 ID 与运行记录一致，并检查镜像是否已有 <code>sitecustomize</code>。若镜像已有该模块，不要用新路径覆盖它；先核验正确的组合方式。以下示例使用本轮可查到的历史记录和缓存。新任务应换成本次冻结的运行记录与同镜像缓存。

~~~bash
R=/data/testuser01-b1/glm53-b1-decode-20261027
RUN_JSON="$R/results/data2-prof-20261028-0959/run.json"
CACHE_ROOT=/data/testuser01-b1/glm53-profiling-20261030-1114/native-jit-cache
CACHE="$CACHE_ROOT/torchinductor"
HELPERS="$R/.agents/skills/dcu-rocprof-report-skill/helpers"
CPU_OUT="$R/results/profile-cpu-$(date +%Y%m%d-%H%M%S)"

(
set -euo pipefail
test -s "$RUN_JSON"
test -s "$CACHE_ROOT/provenance.json"
test -d "$CACHE"
python3 - "$RUN_JSON" "$CACHE_ROOT/provenance.json" <<'PY'
import json, sys
run, cache = (json.load(open(p)) for p in sys.argv[1:])
if run["image_id"] != cache["image_id"]:
    raise SystemExit("cache and image differ")
print("image identity matches cache provenance")
PY
IMAGE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image"])' "$RUN_JSON")
IMAGE_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image_id"])' "$RUN_JSON")
test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$IMAGE_ID"
SITE_STATE=$(docker run --rm --network none -e VLLM_PLUGINS=none --entrypoint python3 "$IMAGE" \
  -c 'import importlib.util; s=importlib.util.find_spec("sitecustomize"); print(s.origin if s else "none")')
test "$SITE_STATE" = none
mkdir "$CPU_OUT"
docker run --rm --network none -e VLLM_PLUGINS=none \
  -e LD_PRELOAD=/opt/dtk/rocprofiler/lib/rocprofiler/librocprofiler_tool.so \
  -e ROCPROFILER_KERNEL_TRACE=1 -e ROCPROFILER_ROCTX_TRACE=1 \
  -e OUTPUT_PATH=/out/profiler/rocprof -e PYTHONPATH=/tree/compiler_env \
  -v "$HELPERS:/tree:ro" -v "$CACHE:/cache:ro" -v "$CPU_OUT:/out" \
  --entrypoint bash "$IMAGE" -c \
  'python3 /tree/compiler_env/check_compiler_env.py /out/test-artifacts /cache &&
   python3 /tree/compiler_env/check_compiler_env_failure.py /out/failure-case /cache' \
  > "$CPU_OUT/container.log" 2>&1
cat "$CPU_OUT/test-artifacts/acceptance.json"
cat "$CPU_OUT/failure-case/failure-acceptance.json"
REFERENCE_TRACE=/data/testuser01-b1/glm53-profiling-20261030-1114/tool-probe/graph-kernel-roctx/results_59.csv
test -s "$REFERENCE_TRACE"
python3 "$HELPERS/test_probe_validators.py" "$REFERENCE_TRACE" > "$CPU_OUT/probe-validator-acceptance.json"
cat "$CPU_OUT/probe-validator-acceptance.json"
)
~~~

命令会要求镜像的 <code>sitecustomize</code> 查询输出 <code>none</code>。三份 acceptance JSON 必须都显示 <code>accepted: true</code>，且相关命令退出码为 0。失败或文件缺失时停止；旧目录中的成功文件不能替代本次结果。测试使用全新子目录，失败路径还检查被阻塞的编译子进程已回收。<code>REFERENCE_TRACE</code> 是本轮保留的历史探针；在其他节点使用该节点已验收的对应 trace。

测试脚本在 [compiler_env/check_compiler_env.py](../helpers/compiler_env/check_compiler_env.py) 和 [compiler_env/check_compiler_env_failure.py](../helpers/compiler_env/check_compiler_env_failure.py)。安装器在 [compiler_env/sitecustomize.py](../helpers/compiler_env/sitecustomize.py)。它只处理已核实的直接 ELF 编译器调用；shell、ninja、ccache 和其他未核实路径不在覆盖范围。不要把它设为系统全局默认。

在服务中使用该安装器时，保留镜像原有 Python 搜索路径。请求开始前核对 8 个实际 model worker 的 <code>python_hook_loaded</code> 和 <code>profiler_mapped</code> 收据。没有收据时标记未验证，不因服务可访问而推断成功。

## 4. 用短探针检验目标栈

这些 GPU 命令只能在已授权的资源 owner 持有适用租约、且该租约中的设备尚未开始模型服务或其他测量时运行。node2 的 <code>--runtime dtk --privileged</code> 会暴露全部 8 张卡；即使探针只用逻辑设备 0，也必须确认资源 owner 已独占 8 卡。检查项目的锁、<code>scripts/hcu_check.sh</code>、<code>hy-smi --showpids</code> 和其他适用的租约记录。锁或空闲检查不自动授予权限。如果没有能确认独占权的资源 owner，停止，不要仅凭空闲状态启动。

node4 的 <code>scripts/hcu_run.sh</code> 是该节点的单卡入口。它的驱动曾与 node2 不同；node4 的成功不能证明 node2 的工具覆盖。不要在没有验证设备隔离和镜像兼容性的情况下把该脚本搬到 node2。

持有目标资源后，选择全新的输出目录并运行下面的 graph 探针。<code>R</code>、<code>RUN_JSON</code> 和 <code>HELPERS</code> 沿用上一步。命令重新读取并核对镜像 ID。以下命令本身不申请或释放父级租约；父级 owner 必须在子命令失败或中断后继续持有租约，直到核验释放状态。

~~~bash
OUT="$R/results/profile-graph-$(date +%Y%m%d-%H%M%S)"
NAME="glm53-graph-probe-$(date +%Y%m%d-%H%M%S)-$$"
(
set -euo pipefail
IMAGE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image"])' "$RUN_JSON")
IMAGE_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image_id"])' "$RUN_JSON")
test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$IMAGE_ID"
mkdir "$OUT"
docker run --rm --name "$NAME" --network none --runtime dtk --privileged \
  --device /dev/kfd --device /dev/dri -e VLLM_PLUGINS=none \
  -v /opt/hyhal:/opt/hyhal:ro -v "$R:/tree:ro" -v "$OUT:/out" \
  --entrypoint bash "$IMAGE" -c \
  '/opt/dtk/rocprofiler/bin/rocprofv2 --kernel-trace --roctx-trace -d /out \
   python3 /tree/.agents/skills/dcu-rocprof-report-skill/helpers/serving_graph_probe.py' \
  > "$OUT/container.log" 2>&1
)
~~~

graph 命令若非零退出，不要运行后续分析。容器结束后，先由资源 owner 确认该容器已经退出、相关 KFD PID 已消失，并重新执行 HCU 和租约检查。如果退出或释放状态未知，保留租约并停止；不要杀掉不属于本次任务的进程。再执行 CPU 验证：

~~~bash
(
set -euo pipefail
COUNT=$(find "$OUT" -maxdepth 1 -name 'results_[0-9]*.csv' -type f | wc -l)
test "$COUNT" -eq 1
find "$OUT" -maxdepth 1 -name 'results_*.csv' -print
TRACE=$(find "$OUT" -maxdepth 1 -name 'results_[0-9]*.csv' -print -quit)
test -n "$TRACE"
python3 "$HELPERS/verify_graph_trace.py" "$TRACE" | tee "$OUT/graph-acceptance.json"
PID=$(basename "$TRACE" .csv | cut -d_ -f2)
python3 "$HELPERS/verify_roctx_trace.py" "$OUT/roctx_trace_$PID.csv" --pid "$PID" \
  | tee "$OUT/roctx-acceptance.json"
)
~~~

在 bash 中启用 <code>set -o pipefail</code> 后运行带 <code>tee</code> 的验证命令。graph 结果要求唯一 PID、一个 GPU、PID 文件名匹配、有效时间戳、无重复 dispatch ID，且乘法与加法各 792 次。ROCTX 结果单独检查两个 CPU 标记。测试代码包括 [serving_graph_probe.py](../helpers/serving_graph_probe.py)、[verify_graph_trace.py](../helpers/verify_graph_trace.py)、[verify_roctx_trace.py](../helpers/verify_roctx_trace.py) 和 [test_probe_validators.py](../helpers/test_probe_validators.py)。

graph 通过只证明该玩具图的计数。ROCTX 通过只证明 CPU 标记送达；它不证明标记与异步 GPU kernel 的对应关系。本轮完整模型采集中没有找到 ROCTX CSV，因此这项能力仍是未知。

## 4A. 当前首选：xsys 短窗口完整服务 trace

当目标是当前 bw1100 固定镜像上的 graph replay、MoE EP、DSA 或 KDA 节点，而 rocprofv2 连续 trace 已经证明会超时、截断或写出冲突文件时，优先使用 host DTK 的 <code>xsys</code>。它不是无 profiler 性能基线，也不是硬件计数器替代品；它提供 HCC Task 时间线和 per-process CSV，用于确认节点覆盖、rank 对齐和限定时间窗口内的设备活动。

只在资源 owner 已独占 8 卡、并已冻结无 profiler 对照后执行。保留既有服务生命周期、镜像、模型和 client 命令，只改服务容器内实际启动的可执行文件：把原来的 <code>vllm serve ...</code> 包装为下面形式。宿主机工具目录必须只读挂载到容器的 <code>/hostdtk</code>；不要覆盖镜像内的 DTK。

~~~bash
XWIN=720000:180000:3600000
mkdir -p "$OUT/xsys"

/hostdtk/bin/xsys --kernel-trace -tp "$XWIN" -fi 1000 -d "$OUT/xsys" \
  vllm serve "$MODEL" ...原先冻结的 serving 参数...
~~~

<code>-tp</code> 三段依次是延迟、活动和循环间隔，单位为毫秒。第三段必须是非零正数。本机探针中 <code>15000:10000:0</code> 没有限制采集范围，不能使用。<code>720000:180000:3600000</code> 在 2026-10-30 的固定镜像服务中成功产生一个有限窗口，但它不是可移植的启动延迟：xsys 父进程、model worker 和 graph capture 的启动时刻不同。用服务日志中的 <code>[XSYS] Start Profiling</code>、<code>[XSYS] Stop Profiling</code>，以及 client 开始和结束收据确认请求实际落在窗口内。若没有重叠证据，不要分析该 trace。

先用 skill 自带的 8 worker Graph 探针证明这次工具与镜像组合仍能覆盖 child process。它必须在实际服务前、同一资源 owner 的 8 卡容器内单独运行。输出的每个 rank 都应有 <code>value: 4.0</code> 和 <code>expected_task_nodes: 50</code>，并且 xsys 目录中应有 8 个 <code>hcc_ops_trace_&lt;pid&gt;.csv</code>。缺少任一 worker、PID 或 Task 文件时停止，不启动模型服务。

下面的容器命令只说明探针的内部启动形式。项目已有 GPU gateway 时，必须把同一内部命令放进 owner 的生命周期和锁中，不能绕过 admission 直接运行它。

~~~bash
PROBE_OUT="$R/results/xsys-graph-probe-$(date +%Y%m%d-%H%M%S)"
mkdir "$PROBE_OUT"
docker run --rm --runtime dtk --privileged --network none --ipc host --shm-size 2g \
  --device /dev/kfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v /opt/dtk:/hostdtk:ro \
  -v "$R:/tree:ro" -v "$PROBE_OUT:/out" \
  --entrypoint bash "$IMAGE" -lc \
  '/hostdtk/bin/xsys --kernel-trace -fi 1000 -d /out/xsys \
   python3 /tree/.agents/skills/dcu-rocprof-report-skill/helpers/xsys_graph_probe.py' \
  | tee "$PROBE_OUT/container.log"
find "$PROBE_OUT/xsys" -name 'hcc_ops_trace_*.csv' -type f | wc -l
~~~

完成服务请求、正常停止本次容器并确认 CSV 不再变化后，运行以下只读验收。示例中的 <code>--require</code> 只适用于同一 GLM-5.3-Flash TP1/DP8/EP8、prefix-repetition C8 负载；换负载时只保留该负载应实际执行的名称。

~~~bash
XSYS_OUT=/data/testuser01-b1/glm53-profiling-20261030-1114/xsys-full-c8-20261030-1926
CHECK_DIR="$R/results/xsys-inventory-$(date +%Y%m%d-%H%M%S)"
mkdir "$CHECK_DIR"

ionice -c3 nice -n 19 python3 "$HELPERS/verify_xsys_model_trace.py" \
  "$XSYS_OUT/server.log" "$XSYS_OUT/xsys" --workers 8 \
  --require 'deep_ep::internode_ll::dispatch' \
  --require 'deep_ep::internode_ll::combine' \
  --require 'm_grouped_fp8_gemm' \
  --require 'sparse_fp8::flash_fwd' \
  --require 'topKPerRowDecode' \
  --require 'recurrent_gated_delta' \
  > "$CHECK_DIR/inventory.json"
VERIFY_RC=$?
cat "$CHECK_DIR/inventory.json"
echo "verifier exit: $VERIFY_RC"
~~~

只有 exit 为 0 且 <code>accepted: true</code> 才表示所有 worker PID 均有单设备、可解析且非空的 Task trace，并且指定的实际 kernel 名称在 Task 节点中出现。该验收不检查输出质量、客户端阶段、trace 时间窗和性能。另行确认每条请求完成、输出长度和 token/任务专属质量门；若首次请求触发 JIT，也不要把带采集 run 与已预热的无 profiler run 直接比较。

2026-10-30 的完整 C8 成功记录保存于 <code>/data/testuser01-b1/glm53-profiling-20261030-1114/xsys-full-c8-20261030-1926</code>：16/16 请求成功且每条输出 1024 token；8 个 HCC CSV 共约 660 MB，均可解析。相同 run 中 DeepEP dispatch 加 combine 只占 8 卡 Task 时长和的 0.674%，没有重现旧截断 rocprof trace 的 99% EP 结果。两个 workload、工具和时间边界不同，不能把此观察解释为已有端到端性能比较；它只否定了“旧 99% 已足以证明原生 MoE 通信主导”的说法。

<code>rocsys</code> 的当前 DTK 会话启动曾创建 session 但未成功产生 trace，因此它仍是未验证替代方案。不要以 rocsys 接口返回或 xsys 的单一 trace 替代上面的实际验收。

如果准备评估更轻的 event 方案，在同样的独占租约下，用另一个全新目录运行：

~~~bash
EVENT_OUT="$R/results/profile-event-$(date +%Y%m%d-%H%M%S)"
EVENT_NAME="glm53-event-probe-$(date +%Y%m%d-%H%M%S)-$$"
(
set -euo pipefail
IMAGE=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image"])' "$RUN_JSON")
IMAGE_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["image_id"])' "$RUN_JSON")
test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$IMAGE_ID"
mkdir "$EVENT_OUT"
docker run --rm --name "$EVENT_NAME" --network none --runtime dtk --privileged \
  --device /dev/kfd --device /dev/dri -e VLLM_PLUGINS=none \
  -v /opt/hyhal:/opt/hyhal:ro -v "$R:/tree:ro" -v "$EVENT_OUT:/out" \
  --entrypoint python3 "$IMAGE" \
  /tree/.agents/skills/dcu-rocprof-report-skill/helpers/serving_event_probe.py \
  /out/event-probe.json
cat "$EVENT_OUT/event-probe.json"
)
~~~

event 命令结束后也要执行相同的容器、KFD 和租约检查。若命令非零退出，保留失败 JSON，停止下一阶段。[serving_event_probe.py](../helpers/serving_event_probe.py) 检查两个 graph 交替 replay 时输出是否逐位相同、event 时间是否为正、不同工作量能否被区分。它尚未在 GPU 上运行。失败表示这个具体方案未通过；不能据此断言硬件不支持所有 event 方案。成功也只证明玩具图，不证明真实模型中的 hook 被捕获、graph 分割不变或计时扰动可接受。

## 5. 小样本成功后再扩大

先用能快速完成的真实请求检查完整模型路径。冻结输入和输出长度，并验证请求是否完成。短上下文 smoke 可能不会触发 DSA indexer；只有原始记录证明路径执行，才能声称覆盖 DSA indexer。然后选包含目标 KV 长度、实际 local batch 和 MTP 状态的代表性负载，记录预算和停止规则。

在相同负载下分别检查无 profiler 与 profiler 运行。记录 profiler 下请求是否完成、各 rank 是否有记录、每个 CSV 是否解析完整、graph 内专家 GEMM 和 EP 相关 kernel 是否出现、实际 runtime graph 模式是否改变。若只看到启动、编译、prefill、dummy batch 或空文件，不要将其报告为纯 decode。

针对 TP1 且 DP 与 EP 各为 8 的 trace，可以运行下面的只读完整性检查。先确认采集已终止、缓冲已落盘且文件不再变化；该检查不是停止服务或释放资源的操作。示例指向本轮失败记录，预期返回非零并写出错误；不要将该示例当成成功结果。此检查会读取完整 CSV，应安排在不干扰其他任务的时段。它只核对文件解析、PID 和 GPU 身份，不验证阶段、语义或性能。

~~~bash
PROFILE_OUT=/data/testuser01-b1/glm53-profiling-20261030-1114/profile-clean
CHECK_DIR="$R/results/profile-inventory-$(date +%Y%m%d-%H%M%S)"
if mkdir "$CHECK_DIR"; then
  VERIFY_RC=0
  ( set -o noclobber
    ionice -c3 nice -n 19 python3 "$HELPERS/verify_model_trace_inventory.py" \
      "$PROFILE_OUT/server.log" "$PROFILE_OUT/rocprof" --workers 8 \
      > "$CHECK_DIR/inventory.json"
  ) || VERIFY_RC=$?
  cat "$CHECK_DIR/inventory.json"
  echo "verifier exit: $VERIFY_RC"
else
  echo "无法创建全新检查目录；没有覆盖旧结果。" >&2
fi
~~~

使用新 trace 时，只有显示的 verifier exit 为 0 且 JSON 显示 <code>accepted: true</code>，才表示文件与身份检查通过。程序在 [verify_model_trace_inventory.py](../helpers/verify_model_trace_inventory.py)；这不是全模型 profiler 或输出正确性的验收。 文件如果恰好在完整行边界被截断，或只包含启动阶段的数据，CSV 语法仍可能通过；必须另查采集结束收据、实际阶段和预期记录。

当前没有经过验收的一键命令可以给出该模型 MoE、DSA、KDA 的原生端到端占比。连续 kernel tracing 的完整模型运行本轮超出预算，不能直接照搬。2026-10-30 检查的 <code>scripts/run_data2_baseline.sh</code> 的 <code>D2_ROCPROF</code> 分支仍追加固定 <code>-o k</code>；不要按该版本原样启动多进程采集。该脚本已有其他未提交改动，本次没有修改它。可先验证上述 event 探针；若通过，再独立实现并验证模型模块边界、event replay、输出语义、各 rank 和插桩开销。不要把这一研究方案标为已经完成。

## 6. 分析时保留分母和未归属部分

如果原始 kernel trace 与阶段范围已经通过各自检查，可以计算该范围内的累计 kernel 设备时间：

~~~text
某家族份额 = 该家族已记录 kernel 的设备时间之和
           / 同一范围内全部已记录 kernel 的设备时间之和
~~~

先合计时间，再求比例。保留通用 GEMM、quant、elementwise 和无法可靠归属的 kernel；不要从分母删除它们。报告各 rank 的覆盖、样本数和边界。多个 rank 或 stream 的 kernel 时间可能重叠，累加值不是请求 wall time，也不是关键路径。EP kernel 的时间可能包含等待其他 rank。

已确认的名称需按具体语义分类：<code>paged_mqa_logits_fp8</code> 和 <code>topKPerRowDecode</code> 是 DSA indexer；<code>combine_sampled_and_draft_tokens</code> 是采样；<code>deep_ep::internode_ll::dispatch/combine</code> 属于 EP；<code>m_grouped_fp8_gemm_nt_masked</code> 是 routed expert GEMM；<code>_router_gemv_kernel</code> 是当前镜像的 FP32 MoE router；<code>act_and_mul_fp8_quant_ep_masked</code> 是当前专家路径的激活与量化 kernel。仅靠 <code>deep_gemm</code>、<code>masked</code> 或 <code>combine</code> 字符串不能认定完整 MoE。

旧 <code>scripts/kernel_table.py</code>、<code>step_kernels.py</code> 和 <code>step_timeline.py</code> 将同名 CPU 与 GPU execute 标记一起计入，且旧 family 正则存在误分类。本次没有修改这些脚本；不要使用旧 <code>decode-table.json</code> 的 step 数或 MoE 百分比。HIP Graph 内部节点遗漏也使 torch-profiler 家族比例不可靠。

<code>scripts/rocprof_step_table.py</code> 的相邻 rejection 切分只能作为候选分析：无本地请求的 EP rank 可能没有 rejection，多 logit chunk 可能产生多个 rejection，且不同 stream 的 kernel 可能跨越边界。只有先核验 chunk 数、实际 shape、采样间断、跨界 kernel 和各 rank 的对应关系，才能谈被验收的设备周期。MTP draft、采样尾部与下一次 target forward 可能处于同一个 rejection-to-rejection 区间；它不是单次 target forward。

## 7. 保存结果并确认资源状态

每次保留输入清单、实际命令、镜像和源码身份、driver/设备信息、资源收据、阶段边界、请求与有效性记录、所有原始 trace、分析代码及其退出状态。达到预算时提前开始有界退出，并验证 profiler 文件完整落盘；强制终止可能截断 CSV。记录 profiler 对运行的影响和未覆盖的 rank。不要覆盖失败、未完成或已冻结记录。

如果预算即将到期，先通过本次服务支持的接口有界停止请求，等待客户端保存终态，再按 owner 的生命周期流程停止 profiler 和服务。本次固定镜像的开发模式提供以下接口；仅在确认 8203 是本任务独占服务且不承载其他人的请求后使用：

~~~bash
curl -fsS -X POST http://127.0.0.1:8203/abort_requests \
  -H 'Content-Type: application/json' -d '{}'
~~~

响应的成功和中止数量不能证明所有客户端、GPU kernel 或 profiler 缓冲已结束。逐个核对请求终态和文件，保留未完成记录；若正常退出无法确认，不要将已有 trace 标记为完整。

先结束本任务拥有的服务和容器，再检查它们的进程、KFD PID、显存和租约。遇到其他用户的进程时，不要停止它们。检查不可用或结果冲突时，资源状态是未知；进程消失或队列为空本身不能证明资源已经释放。
