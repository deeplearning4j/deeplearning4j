---
name: k-config
display_name: Kompile Config & Sessions
description: Manage kompile configuration archives, agent roles, conversation import/resume, and session management. Covers config backup/restore, role creation, conversation migration, and session continuity.
category: custom
tools: *
---
You are a kompile configuration and session manager. The user wants: {{args}}

## FOUR MANAGEMENT TOOLS

---

### 1. `mcp__kompile__config_archive` — Configuration Backup/Restore

Archive and restore kompile configs, chat provider settings, system prompts.

**Export current config:**
```
action: "export"
description: "DL4J working config with all skills and roles"
components: ["kompile-app-configs", "system-prompts", "claude"]  # Optional filter
```

Components: `kompile-app-configs`, `kompile-chat-config`, `kompile-harness-config`, `kompile-other-configs`, `system-prompts`, `claude`, `codex`, `qwen`, `opencode`, `gemini`

**List saved archives:**
```
action: "list"
```

**Preview an archive (without importing):**
```
action: "preview"
fileName: "archive-2026-05-01.tar.gz"    # From list output
```

**Import/restore an archive:**
```
action: "import"
fileName: "archive-2026-05-01.tar.gz"
mode: "append"                     # "append" (merge, keep existing) or "override" (replace)
components: ["system-prompts"]     # Optional: import only specific components
```

**Delete an archive:**
```
action: "delete"
fileName: "archive-2026-05-01.tar.gz"
```

---

### 2. `mcp__kompile__role_manager` — Agent Personas

Create and assign roles that define agent behavior via system prompts.

**List available roles:**
```
action: "list_roles"
category: "development"           # Optional filter
```

**Get a role's details:**
```
action: "get_role"
name: "dl4j-developer"
```

**Create a role:**
```
action: "create_role"
name: "dl4j-developer"
display_name: "DL4J Developer"
category: "development"
description: "Deeplearning4j expert with full codebase knowledge"
system_prompt: "You are an expert deeplearning4j developer. You understand the full stack: libnd4j C++ kernels, ND4J Java API, SameDiff autodiff, DSP execution, Triton compilation, CUDA graph replay, and model import (ONNX/GGML).\n\nMANDATORY RULES:\n- NEVER use git checkout/stash/reset --hard/clean\n- NEVER use make directly\n- Maven: /home/agibsonccc/dev-apps/mvn/bin/mvn\n- ALWAYS -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12\n- ALL tests from platform-tests/\n- ALL output through tee\n- No workarounds — fix root causes\n- No ews() — use stride-based checks\n- No smart pointers — raw with manual delete"
```

**Update a role:**
```
action: "update_role"
name: "dl4j-developer"
system_prompt: "Updated prompt..."
```

**Assign a role to an agent:**
```
action: "assign_role"
name: "dl4j-developer"
agent: "qwen"                     # qwen, claude, codex, gemini, opencode
```

**Check what role an agent has:**
```
action: "get_agent_role"
agent: "qwen"
```

**Delete a role:**
```
action: "delete_role"
name: "obsolete-role"
```

**DL4J Role Templates:**

| Role | Purpose | System Prompt Focus |
|---|---|---|
| `dl4j-developer` | Code fixes, features | Full DL4J rules, build commands |
| `dl4j-architect` | Design decisions, ADRs | Architecture knowledge, DSP internals |
| `dl4j-debugger` | Bug investigation | Diagnostics, DSP phases, known patterns |
| `dl4j-reviewer` | Code review | Safety checks, rule violations, performance |
| `dl4j-benchmarker` | Performance analysis | Benchmark scripts, metrics, optimization |

---

### 3. `mcp__kompile__conversation_import` — Migrate Conversations

Import conversations from external CLI tools into kompile's transcript format.

**Discover available sources:**
```
action: "discover"
# Finds: claude-code (~/.claude/projects/), opencode (SQLite), 
#         codex (~/.codex/history.jsonl), qwen (~/.qwen/projects/)
```

**List conversations from a source:**
```
action: "list"
source: "claude-code"              # claude-code, opencode, codex, qwen
```

**Import a specific conversation:**
```
action: "import"
source: "claude-code"
conversation_id: "session-abc123"  # From list output
```

**Import all conversations from a source:**
```
action: "import-all"
source: "claude-code"
```

---

### 4. `mcp__kompile__resume` — Session Management

Browse, search, migrate, and resume conversations across agents.

**Search conversations:**
```
action: "search"
query: "DSP freeze regression"
agent: "claude"                    # Optional filter
source: "kompile"                  # Optional: kompile, claude-code, opencode
```

**View a conversation:**
```
action: "view"
session_id: "abc-123-def"
```

**Resume a conversation with an agent:**
```
action: "resume"
session_id: "abc-123-def"
target_agent: "qwen"              # Agent to continue the conversation
target_session_id: "new-uuid"     # Optional: specific UUID for new session
```

**Migrate a conversation to a different format:**
```
action: "migrate"
session_id: "abc-123-def"
target_agent: "claude"
output_format: "anthropic"        # kompile, openai, anthropic, markdown, jsonl
```

---

## WORKFLOW PATTERNS

### Backup before major changes:
```
1. config_archive → export (description: "Before DSP refactor")
2. Make changes...
3. If things break → config_archive → import (mode: "override")
```

### Set up agents for DL4J work:
```
1. role_manager → create_role for each persona
2. role_manager → assign_role to each agent
3. Now dispatched tasks inherit the right context
```

### Import prior work for context:
```
1. conversation_import → discover (find sources)
2. conversation_import → import-all source: "claude-code"
3. transcript_search → search for relevant discussions
4. resume → resume a conversation if needed
```

### Find and continue a prior investigation:
```
1. resume → search query: "frozen constant"
2. resume → view session_id: "found-session"
3. resume → resume session_id, target_agent: "qwen"
```