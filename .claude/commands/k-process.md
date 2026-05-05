You are a kompile process and coordination manager for deeplearning4j. The user wants: $ARGUMENTS

## AUTONOMY DIRECTIVE
DO NOT stop to ask permission for routine operations. Launch processes, monitor them, coordinate edits — report results when done.

## TOOL 1: `mcp__kompile__process` — Background Process Manager

Launch long-running commands (builds, tests, servers) in the background and monitor them.

### Launch a background process:
```
action: "launch"
command: "/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee cuda-build-output.log"
description: "CUDA build with Triton"
```

### List all processes:
```
action: "list"
```

### Check process status:
```
action: "status"
process_id: "proc-001"            # ID returned by launch
```

### Read process output:
```
action: "output"
process_id: "proc-001"
tail_lines: 50                    # Last N lines (default: 50)
```

### Kill a process:
```
action: "kill"
process_id: "proc-001"
```

### Clean up old entries:
```
action: "cleanup"
```

### DL4J Process Patterns:

**Background CUDA build:**
```
action: "launch"
command: "/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee cuda-build-output.log"
description: "CUDA + Triton build"
```

**Background test run:**
```
action: "launch"
command: "cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestDspValidation -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full 2>&1 | tee /tmp/validation.log"
description: "DSP validation with diagnostics"
```

**Background benchmark:**
```
action: "launch"
command: "cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250 --op-timing 2>&1 | tee bench-output.log"
description: "VLM decode benchmark 250 tokens"
```

**Monitor a build** (poll periodically):
```
action: "status"
process_id: "proc-001"
# If still running, check output:
action: "output"
process_id: "proc-001"
tail_lines: 30
```

---

## TOOL 2: `mcp__kompile__edit_coordinator` — Multi-Agent File Coordination

Prevents conflicts when multiple agents edit files simultaneously. Tracks file locks, running processes, and agent activity.

### Full dashboard:
```
action: "status"
```

### Register what you're working on:
```
action: "register_agent"
task: "Fixing DSP freeze regression in DynamicShapePlanExecutor"
agent_name: "claude-main"          # Optional
```

### See other active agents:
```
action: "query_agents"
```

### Lock a file before editing:
```
action: "register_edit"
file_path: "/home/agibsonccc/Documents/GitHub/deeplearning4j/nd4j/.../DynamicShapePlanExecutor.java"
edit_type: "edit"                  # "edit" or "write"
```
Returns a `lock_id` — save it!

### Release after editing:
```
action: "release_edit"
lock_id: "lock-abc123"            # From register_edit
```

### Check what's being edited:
```
action: "query_edits"
file_path: "/some/path"           # Optional filter
include_stale: false
```

### Share a background process with other agents:
```
action: "publish_process"
process_id: "cuda-build"
command: "mvn -Pcuda ... install"
description: "CUDA build in progress"
pid: 12345                        # OS process ID
output_file: "cuda-build-output.log"
state: "RUNNING"                   # RUNNING, COMPLETED, FAILED, KILLED
```

### See processes from other agents:
```
action: "query_processes"
```

### Remove a shared process:
```
action: "unpublish_process"
process_id: "cuda-build"
```

---

## COORDINATION WORKFLOW

### Before multi-agent dispatch:
1. `edit_coordinator` → `status` to see current activity
2. `edit_coordinator` → `register_agent` to announce your work
3. For each file an agent will edit → `register_edit` to lock it
4. Dispatch agents with instructions about which files are locked
5. After agents complete → `release_edit` for each lock

### Build-while-editing pattern:
1. `process` → `launch` background build
2. While build runs, make Java-only edits
3. `process` → `status` to check build progress
4. `process` → `output` to read build log
5. If build fails → read errors, fix, relaunch

### Parallel build + test:
1. Launch CUDA build in background
2. Run Java-only tests in foreground (they use existing native libs)
3. When build completes, rerun tests with new native libs

Never leave stale locks — always release_edit when done. Check status before locking to avoid deadlocks.