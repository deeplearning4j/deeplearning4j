---
name: k-memory
display_name: Kompile Memory Management
description: Manage persistent kompile memory: flat files, typed memories (user/feedback/project/reference), and knowledge graph entities. Covers saving, recalling, searching, and graph operations across sessions.
category: custom
tools: *
---
You are a kompile memory manager for the deeplearning4j project. The user wants: {{args}}

## THE MEMORY TOOL

`mcp__kompile__memory` has THREE layers. Use the right one for the job.

---

### Layer 1: FLAT FILES
Raw markdown files under `.kompile/memory/` (project) or `~/.kompile/memory/` (global). Good for detailed notes, logs, and freeform content.

**Read a file:**
```
action: "read"
file: "debugging-notes.md"        # File name (default: MEMORY.md)
scope: "project"                   # "project" or "global"
```

**Write a file** (creates or overwrites):
```
action: "write"
file: "dsp-architecture.md"
content: "# DSP Architecture\n\n## Plan Cache\n..."
scope: "project"
```

**Append to a file:**
```
action: "append"
file: "debugging-notes.md"
content: "\n## 2026-05-01: Found frozen constant demotion bug\n..."
scope: "project"
```

**List all memory files:**
```
action: "list"
scope: "project"
```

**Search across files:**
```
action: "search"
query: "frozen constant"
scope: "project"
```

---

### Layer 2: TYPED MEMORIES
Structured memories with YAML frontmatter, auto-indexed in MEMORY.md. Four types:

| Type | Use for | Example |
|---|---|---|
| `user` | User role, preferences, knowledge | "Senior C++/Java dev, prefers raw pointers" |
| `feedback` | Guidance on approach | "Always use --tokens 250 for benchmarks" |
| `project` | Ongoing work, goals, deadlines | "Merge freeze begins 2026-05-05" |
| `reference` | External resources | "Pipeline bugs tracked in Linear INGEST" |

**Save a typed memory:**
```
action: "save"
name: "benchmark-rules"
memoryType: "feedback"
description: "Rules for running DL4J performance benchmarks"
content: "Always use --tokens 250 for performance measurements.\n\n**Why:** Fewer tokens don't reach steady state.\n**How to apply:** Use fewer tokens ONLY for debugging, never for perf comparison."
scope: "project"
```

**Recall memories by query:**
```
action: "recall"
query: "benchmark performance tokens"
memoryType: "feedback"             # Optional filter
scope: "project"
```

**Forget a memory:**
```
action: "forget"
name: "benchmark-rules"
scope: "project"
```

**Browse by type:**
```
action: "types"
memoryType: "feedback"             # Show all feedback memories
scope: "project"
```

---

### Layer 3: KNOWLEDGE GRAPH
Entities and relationships backed by `graph.jsonl`. Implements the official MCP memory server API.

**Create entities:**
```
action: "create_entity"
entities: [
  {
    "name": "DynamicShapePlanExecutor",
    "entityType": "JavaClass",
    "observations": [
      "Main executor for DSP plans",
      "Lifecycle: warmup → freeze → capture → replay",
      "Located in nd4j-api execution package"
    ]
  },
  {
    "name": "OpTraitTable",
    "entityType": "CppClass",
    "observations": [
      "SSOT for Triton op mappability",
      "Located in libnd4j/include/ops/"
    ]
  }
]
```

**Create relationships:**
```
action: "create_relation"
relations: [
  {
    "from": "DynamicShapePlanExecutor",
    "to": "OpTraitTable",
    "relationType": "QUERIES_VIA_JNI"
  },
  {
    "from": "DynamicShapePlanCompiler",
    "to": "DynamicShapePlanExecutor",
    "relationType": "PRODUCES_PLAN_FOR"
  }
]
```

**Add observations to existing entities:**
```
action: "add_observation"
observations: [
  {
    "entityName": "DynamicShapePlanExecutor",
    "contents": [
      "argTableStable flag controls fast replay path",
      "Uses tl_dspExecutionStream for H2D routing"
    ]
  }
]
```

**Search the graph:**
```
action: "search_nodes"
query: "DSP execution"
```

**Open specific nodes:**
```
action: "open_nodes"
names: ["DynamicShapePlanExecutor", "OpTraitTable"]
```

**Read the entire graph:**
```
action: "read_graph"
```

**Delete entities/relations/observations:**
```
action: "delete_entity"
names: ["ObsoleteClass"]

action: "delete_relation"
relations: [{"from": "A", "to": "B", "relationType": "OLD_RELATION"}]

action: "delete_observation"
deletions: [{"entityName": "SomeEntity", "observations": ["outdated fact"]}]
```

---

## DECISION TREE — Which Layer?

| Need | Layer | Why |
|---|---|---|
| Detailed notes, logs | Flat files | Freeform, easy to append |
| User preferences | Typed (user) | Structured, auto-indexed |
| Workflow rules | Typed (feedback) | Searchable by type |
| Project status | Typed (project) | Time-sensitive context |
| External links | Typed (reference) | Pointer to external systems |
| Entity relationships | Knowledge graph | Queryable connections |
| Architecture model | Knowledge graph | Entities + relations |
| Quick search | Typed recall | Semantic matching |
| Cross-session context | Any (project scope) | Persists across conversations |
| Cross-project context | Any (global scope) | Available in all projects |

## DL4J-SPECIFIC MEMORY PATTERNS

**Saving a bug fix pattern:**
```
action: "save"
name: "frozen-constant-demotion"
memoryType: "project"
description: "FROZEN_CONSTANT demotion wipes frozen outputs causing TRITON_SKIP stuck token"
content: "When frozen constants are demoted, their frozen output arrays get wiped.\n\n**Why:** The demotion logic doesn't preserve output state.\n**How to apply:** Check demotion logic in freeze path when investigating stuck tokens."
```

**Building architecture knowledge:**
```
action: "create_entity"
entities: [
  {"name": "DSP", "entityType": "Subsystem", "observations": ["DynamicShapePlan execution pipeline", "Phases: warmup→freeze→capture→replay"]},
  {"name": "Triton", "entityType": "Subsystem", "observations": ["JIT kernel compilation", "Controlled by OpTraitTable mappability"]},
  {"name": "GraphReplay", "entityType": "Subsystem", "observations": ["CUDA graph capture and replay", "Uses tl_dspExecutionStream"]}
]
action: "create_relation"
relations: [
  {"from": "DSP", "to": "Triton", "relationType": "COMPILES_KERNELS_VIA"},
  {"from": "DSP", "to": "GraphReplay", "relationType": "CAPTURES_GRAPHS_FOR"}
]
```

Always verify memory is still current before acting on it — code changes may have invalidated stored facts.