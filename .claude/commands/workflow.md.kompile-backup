# DL4J Development Workflow

You are working on the deeplearning4j codebase. This workflow integrates memory, code search, milestone tracking, and test recording into every step. $ARGUMENTS

---

## PHASE 0: ORIENT (always run first)

### 0a. Recall memory
Before doing anything, check what you already know:
```
mcp__kompile__memory action=recall query="<topic from the task>" scope=project
mcp__kompile__memory action=recall query="<topic from the task>" scope=global
```
Read `MEMORY.md` if the recall is thin:
```
mcp__kompile__memory action=read file=MEMORY.md scope=project
```

### 0b. Check code index
Ensure the codebase is indexed. If `stats` returns nothing or stale data, re-index:
```
mcp__kompile__local_code_index action=stats project_id=dl4j
```
If missing or stale:
```
mcp__kompile__local_code_index action=index directory=/home/agibsonccc/Documents/GitHub/deeplearning4j project_id=dl4j include_patterns=*.java,*.cpp,*.cu,*.h exclude_patterns=target/*,build/*,.git/*
```

### 0c. Check milestone status
See where tests stand right now:
```
mcp__kompile__test_milestone action=status
mcp__kompile__test_milestone action=latest
```

### 0d. Set up task tracking
Create a todo list for your work:
```
mcp__kompile__todowrite action=set todos=[{"content":"Orient: recall memory + check index + milestones","status":"completed","priority":"high"},{"content":"Investigate: search code + trace root cause","status":"pending","priority":"high"},{"content":"Implement fix/feature","status":"pending","priority":"high"},{"content":"Build","status":"pending","priority":"high"},{"content":"Test + record milestone","status":"pending","priority":"high"},{"content":"Save results to memory","status":"pending","priority":"medium"}]
```

---

## PHASE 1: INVESTIGATE

### 1a. Search code with kompile tools
Use the right search tool for the task:

**Find a symbol/class/method:**
```
mcp__kompile__local_code_index action=search query="ClassName" entity_type=CLASS project_id=dl4j
mcp__kompile__local_code_index action=spath query="org.nd4j.linalg.api.ops.impl.*.ClassName"
```

**Find usages:**
```
mcp__kompile__local_code_index action=usages symbol_name="methodName" directory=/home/agibsonccc/Documents/GitHub/deeplearning4j
```

**Trace dependencies (who calls what):**
```
mcp__kompile__code_graph action=symbol fqn="org.nd4j.SomeClass" depth=2 project_id=dl4j
```

**Semantic search across docs:**
```
mcp__kompile__rag_search query="how does DSP graph replay work" search_type=hybrid
```

**Search past conversations for prior work:**
```
mcp__kompile__transcript_search action=search query="the bug or feature topic"
```

### 1b. Save investigation findings to memory
After you understand the problem, save what you learned:
```
mcp__kompile__memory action=save name="<descriptive-name>" memoryType=project description="<one-line summary>" scope=project content="<what you found>\n\n**Why:** <root cause or motivation>\n**How to apply:** <how this shapes the fix>"
```

Update the todo:
```
mcp__kompile__todowrite action=update task_id=2 status=completed
```

---

## PHASE 2: IMPLEMENT

### 2a. Make changes
Use `mcp__kompile__edit` for targeted edits, `mcp__kompile__write` for new files.

Before editing, register the edit for multi-agent coordination:
```
mcp__kompile__edit_coordinator action=register_edit file_path=/path/to/file edit_type=edit
```

### 2b. Update todo
```
mcp__kompile__todowrite action=update task_id=3 status=completed
```

---

## PHASE 3: BUILD

### 3a. Build commands
**CUDA build:**
```
mcp__kompile__bash command="/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee cuda-build.log" description="CUDA build" timeout=600
```

**CPU build:**
```
mcp__kompile__bash command="/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcpu -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native clean install -DskipTests 2>&1 | tee cpu-build.log" description="CPU build" timeout=600
```

**Java-only (no native):**
```
mcp__kompile__bash command="/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl <module>" description="Java module install" timeout=120
```

Build rules:
- ALWAYS use `-Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12`
- ALWAYS pipe through `tee` to a log file
- NEVER use `make` directly — always full mvn with bindings
- NEVER include `platform-tests` in `-pl`
- NEVER clear ccache or change compute capability

### 3b. On build failure
Read the FIRST error from the tee log, fix it, rebuild. Repeat until clean. If you fix something non-trivial, save it:
```
mcp__kompile__memory action=save name="fix-<short-name>" memoryType=project description="Fixed <what> in <file>" scope=project content="<what was wrong and how it was fixed>\n\n**Why:** <root cause>\n**How to apply:** <when this pattern recurs>"
```

### 3c. Update todo
```
mcp__kompile__todowrite action=update task_id=4 status=completed
```

---

## PHASE 4: TEST + RECORD MILESTONES

### 4a. Run tests
ALL tests run from `platform-tests/`. ALL output piped through tee:
```
mcp__kompile__bash command="cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dtest=TestClass#method 2>&1 | tee /tmp/test-output.log" description="Run test" timeout=600
```

Read the tee log for results — NEVER surefire reports.

With DSP diagnostics:
```
-Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full
```

### 4b. Record milestone — MANDATORY after every test run
On success:
```
mcp__kompile__test_milestone action=record passed=<N> total_tests=<N> notes="<what was tested and why>" tags=<relevant-tags> module=<module-name>
```

On failure:
```
mcp__kompile__test_milestone action=fail passed=<N> failed=<M> total_tests=<N+M> notes="<what failed and why>" module=<module-name>
```

Register known regressions:
```
mcp__kompile__test_milestone action=add_regression test_name="TestClass#method" module=<module> notes="<description of the regression>"
```

### 4c. Save test results to memory
After every test run, save the outcome:
```
mcp__kompile__memory action=save name="test-<date>-<short-desc>" memoryType=project description="Test results: <N> passed, <M> failed for <what>" scope=project content="**Test:** <TestClass#method>\n**Result:** <PASS/FAIL>\n**Details:** <key observations>\n**Milestone:** recorded\n\n**Why:** <what was being verified>\n**How to apply:** <implications for future work>"
```

### 4d. On test failure — fix and retest
Read the tee log, diagnose the failure, fix the code, and retest. After fixing:
- If Java-only fix: rebuild Java module, retest
- If C++ fix: full native rebuild, then retest

Record the fix in memory:
```
mcp__kompile__memory action=save name="fix-<test-name>" memoryType=project description="Fixed <test> failure: <root cause>" scope=project content="..."
```

### 4e. Update todo
```
mcp__kompile__todowrite action=update task_id=5 status=completed
```

---

## PHASE 5: BENCHMARK (when performance-related)

### 5a. Run benchmarks
```
mcp__kompile__bash command="cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250 2>&1 | tee /tmp/bench.log" description="Performance benchmark" timeout=600
```

LLM benchmarks:
```
./run-llm-benchmarks.sh --test baseline --models qwen --tokens 250
```

### 5b. Record benchmark results
Save perf numbers to memory with the date:
```
mcp__kompile__memory action=save name="perf-<date>-<config>" memoryType=project description="Benchmark: <N> tok/s on <config>" scope=project content="**Config:** <SLOT_BY_SLOT/TRITON/CUDA_GRAPHS>\n**Result:** <N> tok/s\n**Comparison:** <vs previous>\n**Details:** <breakdown>"
```

Record as milestone:
```
mcp__kompile__test_milestone action=record passed=1 total_tests=1 notes="Benchmark: <N> tok/s on <config>" tags=benchmark,perf module=platform-tests
```

---

## PHASE 6: WRAP UP (always run last)

### 6a. Final memory update
Save a summary of what was accomplished:
```
mcp__kompile__memory action=save name="session-<date>-<topic>" memoryType=project description="<one-line summary of work done>" scope=project content="**Task:** <what was requested>\n**Changes:** <files modified>\n**Result:** <outcome>\n**Tests:** <pass/fail status>\n**Milestone:** <recorded/not applicable>\n\n**Why:** <context>\n**How to apply:** <follow-up work or things to watch>"
```

### 6b. Update MEMORY.md index
Add a pointer to any new memory files:
```
mcp__kompile__memory action=read file=MEMORY.md scope=project
```
Then append the new entry (keep under 200 lines):
```
mcp__kompile__memory action=write file=MEMORY.md content="<updated index with new entries>" scope=project
```

### 6c. Update todo — mark complete
```
mcp__kompile__todowrite action=update task_id=6 status=completed
```

### 6d. Re-index if significant code changes
If you modified or added many files, refresh the code index:
```
mcp__kompile__local_code_index action=index directory=/home/agibsonccc/Documents/GitHub/deeplearning4j project_id=dl4j include_patterns=*.java,*.cpp,*.cu,*.h exclude_patterns=target/*,build/*,.git/*
```

---

## RULES — ALWAYS FOLLOW

### NO COMPROMISES — ARCHITECTURAL CORRECTNESS IS MANDATORY

**NEVER, under ANY circumstance, work around an issue.** NEVER pick the quick answer. ALWAYS pick the architecturally correct solution. There are NO time pressures that justify a shortcut. Take ALL the time you need to do the RIGHT thing.

This means:
- **If a fix feels hacky, it IS hacky.** Step back and find the real solution. A guard clause in the caller is a workaround. A flag to skip broken code is a workaround. Reordering operations to avoid a crash is a workaround. A cast to silence a type error is a workaround. ALL of these are BANNED.
- **If you don't understand the code, STOP and investigate.** Do NOT guess. Do NOT pattern-match from similar-looking code. Read the actual implementation. Trace values to their origins. Understand WHY the code is structured the way it is before changing it.
- **If you are confused about the right approach, spawn subagents.** Use `mcp__kompile__task` or `mcp__kompile__multi_task` to investigate competing hypotheses IN PARALLEL. Have one subagent research approach A while another researches approach B. Compare their findings. Make a decision based on evidence, not intuition.
- **If two approaches seem equivalent, investigate BOTH.** Dispatch parallel subagents to prototype each approach. The one that fits the existing architecture wins. If neither fits, the architecture needs to be understood better — dispatch another subagent to study it.
- **If you encounter a bug while working on something else, FIX IT.** Dispatch a parallel subagent to fix it while you continue your main task. Do NOT leave it for later. Do NOT work around it.
- **If an existing pattern in the codebase is wrong, fix the pattern.** Do not propagate bad patterns just because they exist. If 10 files do it wrong, that means 10 files need fixing — not that the wrong way is now "the convention."
- **NEVER say "this is good enough."** Either it's correct or it's not. Ship correct code.

When in doubt: **dispatch subagents, gather evidence, make the right call.** The cost of getting it wrong is rebuilding. The cost of getting it right is time. Time is always cheaper.

### Memory rules
- **ALWAYS recall before starting** — check what's known about the topic
- **ALWAYS save after fixing** — future sessions need to know what changed
- **ALWAYS save test results** — milestones AND memory, every time
- **ALWAYS save benchmark numbers** — with date, config, and comparison
- **Use typed memories:** `project` for task outcomes, `feedback` for workflow lessons, `reference` for external resources
- **Keep MEMORY.md under 200 lines** — prune stale entries

### Code search rules
- **Use `local_code_index` for symbol/class/method lookup** — it's offline and fast
- **Use `code_graph` for dependency tracing** — inheritance, imports, call chains
- **Use `rag_search` for semantic questions** — "how does X work"
- **Use `transcript_search` for prior conversations** — "did we fix this before"
- **Re-index after significant code changes** — keeps search results fresh

### Milestone rules
- **ALWAYS record after test runs** — `action=record` on pass, `action=fail` on failure
- **ALWAYS register regressions** — `action=add_regression` when a test starts failing
- **Check milestones before fixing** — `action=latest` to see baseline state
- **Compare after fixing** — `action=compare` to verify improvement

### Build rules
- NEVER use `make` directly — always full mvn with bindings
- ALWAYS use `-Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12`
- ALWAYS pipe through `tee`
- NEVER include `platform-tests` in build `-pl`
- NEVER clear ccache or change compute capability

### Test rules
- ALL tests from `platform-tests/`
- ALL output through `tee` — NEVER surefire reports
- Environment vars via `-D` Maven properties, NOT shell exports
- NEVER use `LD_PRELOAD=libjemalloc.so`

### Code rules
- NEVER use `ews()` / `elementWiseStride` — use `strideDescendingCAscendingF()`
- NEVER use `unique_ptr` / `shared_ptr` — raw pointers with manual delete
- NEVER use workarounds — fix root causes
- NEVER dismiss errors — if an issue is a blocker, FIX it no matter what
- Do NOT write one-off `syncToDevice()` calls — assume basic CUDA device syncing works
- If you suspect an infra issue, focus on simpler causes first (wrong shapes, types, data flow)
- For debugging, use: `Nd4j.getEnvironment().setDebug(true); Nd4j.getEnvironment().setVerbose(true);`
- Use platform macros: `SD_HOST`, `SD_DEVICE`, `PRAGMA_OMP_*`, `BUILD_SINGLE_TEMPLATE`

### Autonomy
- NEVER stop to ask the user if you should continue — the answer is always YES
- Build fails? Fix it and rebuild
- Test fails? Fix it and retest
- New error? Fix it
- Repeat until done or genuinely stuck after 5+ different approaches