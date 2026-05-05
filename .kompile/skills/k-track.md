---
name: k-track
display_name: Kompile Tracking & Analytics
description: Track test milestones, agent performance, and tool usage analytics using kompile's test_milestone, performance_harness, and tool_call_catalog tools.
category: custom
tools: *
---
You are a kompile tracking and analytics manager for deeplearning4j. The user wants: {{args}}

## THREE TRACKING TOOLS

---

### 1. `mcp__kompile__test_milestone` — Test Pass/Fail Tracking

Records which commits have working tests. Always find the last known-good commit.

**Initialize project config:**
```
action: "init"
project: "deeplearning4j"
```

**Add a module:**
```
action: "add_module"
module: "dsp-validation"
path: "platform-tests"
build_command: "/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests"
test_command: "cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestDspValidation 2>&1 | tee /tmp/validation.log"
```

**Record a passing milestone:**
```
action: "record"
module: "dsp-validation"
passed: 8
total_tests: 8
notes: "All 8 DSP matrix configs pass on CUDA"
tags: "cuda,dsp,validation"
# commit and branch auto-detected from git
```

**Record a failure:**
```
action: "fail"
module: "dsp-validation"
passed: 6
failed: 2
total_tests: 8
notes: "TRITON_compileAll and CUDA_GRAPHS_frozen failing"
tags: "cuda,regression"
```

**Set quality targets:**
```
action: "set_target"
module: "dsp-validation"
min_pass_rate: 1.0                 # 100% pass rate required
max_failures: 0                    # Zero failures allowed
```

**Track a regression:**
```
action: "add_regression"
test_name: "TestDspConfigurationMatrix#testConfiguration[TRITON_compileAll]"
module: "dsp-validation"
notes: "TRITON_compileAll produces wrong tokens since frozen constant change"
since_commit: "abc1234"            # When regression first appeared
tags: "triton,regression"
```

**Query milestones:**
```
action: "list"
module: "dsp-validation"
limit: 10

action: "latest"                   # Most recent milestone
module: "dsp-validation"

action: "check"                    # Check if current commit has a milestone
module: "dsp-validation"

action: "compare"                  # Compare two milestones
from_id: "ms-001"
to_id: "ms-005"

action: "summary"                  # Overall project health

action: "list_regressions"         # Active regressions
module: "dsp-validation"
```

**Remove resolved regression:**
```
action: "remove_regression"
id: "reg-001"
```

**Project status:**
```
action: "status"                   # Config + modules + targets
```

---

### 2. `mcp__kompile__performance_harness` — Agent Quality Metrics

Track how well different agents perform on tasks. Escape detection, quality scoring, model recommendations.

**View performance leaderboard:**
```
action: "report"
days: 30                           # Time window
task_type: "code-review"           # Optional: filter by task type
```

**Get model recommendation:**
```
action: "recommend"
task_type: "exploration"           # code-review, planning, research, exploration, general
provider: "anthropic"              # Optional: filter by provider
```

**Record a performance observation:**
```
action: "record"
model: "qwen-coder"
agent_name: "qwen"
agent_output: "Full output text..."    # For automatic escape detection + scoring
quality_score: 4.0                     # Or provide direct score (1-5)
correctness: 4                         # Optional: 1-5
completeness: 5                        # Optional: 1-5
design_quality: 3                      # Optional: 1-5
tool_calls: 15                         # Optional
tool_errors: 1                         # Optional
latency_ms: 45000                      # Optional
hit_max_steps: false                   # Optional
subagents_spawned: 2                   # Optional
reasoning: "Fixed the bug correctly but missed a related issue"
```

**Record an escape/failure:**
```
action: "record"
model: "codex"
agent_name: "codex"
escape_type: "EXPLICIT_REFUSAL"        # EXPLICIT_REFUSAL, EMPTY_OUTPUT, TOOL_LOOP
quality_score: 1.0
reasoning: "Agent refused to modify C++ code"
```

**Configure the harness:**
```
action: "config"
judge_enabled: true                    # Use LLM judge for automatic scoring
judge_provider: "anthropic"
judge_model: "claude-sonnet-4-20250514"
auto_swap: true                        # Auto-swap underperforming models
threshold: 2.5                         # Quality threshold for swap
```

**Session stats:**
```
action: "stats"                        # Current session metrics
```

**Reset data:**
```
action: "reset"
model: "codex"                         # Optional: reset only one model
```

---

### 3. `mcp__kompile__tool_call_catalog` — Tool Usage Analytics

Search, list, and analyze tool calls across all agent sessions.

**Search tool calls:**
```
action: "search"
query: "DynamicShapePlan"             # Matches tool name, input, category, etc.
agent: "claude-code"                   # Optional: filter by agent
project: "deeplearning4j"             # Optional: filter by project
category: "filesystem"                 # Optional: filesystem, shell, search, rag, agent, model, web
limit: 50
```

**List tool calls with filters:**
```
action: "list"
tool: "Edit"                           # Filter by tool name
agent: "claude-code"
project: "deeplearning4j"
sort_by: "timestamp"                   # timestamp, tool, category, agent, project
sort_dir: "desc"
group_by: "category"                   # category, project, agent, tool
limit: 50
```

**Aggregate statistics:**
```
action: "stats"
project: "deeplearning4j"
agent: "claude-code"
```

**Index new sessions:**
```
action: "index"
source: "all"                         # all, claude-code, codex, qwen, opencode, gemini
reindex: false                         # true to re-index already indexed sessions
```

**Available filter options:**
```
action: "filters"
```

---

## DL4J TRACKING PATTERNS

### After a successful benchmark run:
```
test_milestone → record:
  module: "vlm-benchmark"
  passed: 1, total_tests: 1
  notes: "92 tok/s lateSteady, 250 tokens, OPTIMAL config"
  tags: "benchmark,cuda,performance"
```

### After fixing a regression:
```
test_milestone → remove_regression: id: "reg-xxx"
test_milestone → record: module, passed, total, notes
```

### Evaluating agent quality after task dispatch:
```
performance_harness → record:
  model, agent_name, agent_output, quality metrics
```

### Understanding tool usage patterns:
```
tool_call_catalog → stats: project: "deeplearning4j"
# → Shows which tools are used most, error rates, etc.
```