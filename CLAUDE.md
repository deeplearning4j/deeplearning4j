@AGENTS.md

# Claude Code-Specific Rules

## Dispatching Subagents

Subagents do NOT automatically inherit knowledge of AGENTS.md. When dispatching a subagent, you **MUST** include the following in the prompt:

1. **Explicit rule reminders.** Copy the specific rules that apply to the subagent's task directly into the prompt. Do NOT say "follow AGENTS.md" — the subagent may not read it. Key rules to always include:
   - **Git Safety:** NEVER use `git checkout`, `git stash`, `git reset --hard`, or `git clean` on files. Use `Edit` tool to make targeted modifications. These git commands destroy uncommitted work irreversibly.
   - **No Workarounds:** Fix root causes directly. NEVER work around a bug.
   - **Build commands:** Include the exact build command if the subagent needs to build. NEVER use `make` directly.
   - **Test location:** ALL tests run from `platform-tests/`. Test output is in the `tee` log file, NOT surefire reports.
   - **No jemalloc:** NEVER use `LD_PRELOAD=libjemalloc.so`.
   - **No `tail`:** NEVER pipe build or test output through `tail`.

2. **Context about what files are modified.** Tell the subagent which files have uncommitted changes so it does not destroy them with git commands.

3. **Scope boundaries.** Tell the subagent exactly what it should and should NOT modify. If it should only investigate, say "DO NOT modify any files — research only."

**Example subagent prompt:**
```
Investigate why X crashes in Y.

RULES (mandatory):
- NEVER use git checkout, git stash, git reset --hard, or git clean — BANNED
- NEVER modify files outside of libnd4j/include/ops/ — research only for other files
- If you need to undo changes, use Edit tool to restore specific lines
- ALL test commands piped through tee: mvn test ... 2>&1 | tee /tmp/test.log
- Do NOT use workarounds — fix root causes

Currently modified files (DO NOT git checkout these): <list>
```

**If a subagent violates a rule**, it is YOUR fault for not including the rule in the prompt. Always be explicit.
