---
name: k-files
display_name: Kompile File Operations
description: File operations via kompile tools: read, write, edit, patch, glob, grep, list. Covers reading, writing, targeted edits, unified diff patches, file discovery, content search, and directory listing.
category: custom
tools: *
---
You are a kompile file operations expert for deeplearning4j. The user wants: {{args}}

## SEVEN FILE TOOLS

These are kompile's MCP equivalents of standard file operations. Use them when dispatching work through kompile agents (they don't have access to Claude Code's built-in Read/Edit/etc.).

---

### 1. `mcp__kompile__read` — Read Files
```
file_path: "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DynamicShapePlanExecutor.java"
offset: 100                       # Optional: start line (1-based)
limit: 50                         # Optional: max lines (default: 2000)
```
Returns content with line numbers. Lines > 2000 chars are truncated.

---

### 2. `mcp__kompile__write` — Create/Overwrite Files
```
file_path: "platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/NewTest.java"
content: "package org.eclipse.deeplearning4j...;\n\npublic class NewTest {\n..."
```
Creates parent directories automatically. **Overwrites existing files** — use `edit` for modifications.

---

### 3. `mcp__kompile__edit` — Targeted String Replacement
```
file_path: "nd4j/.../DynamicShapePlanExecutor.java"
old_string: "if (ews() == 1) {"    # Must be UNIQUE in the file
new_string: "if (shape::strideDescendingCAscendingF(shapeInfo)) {"
replace_all: false                  # true to replace ALL occurrences
```
**Rules:**
- `old_string` must be unique — provide more context if ambiguous
- Always `read` the file first to verify the exact string
- Use `replace_all: true` for renaming variables/methods across a file

---

### 4. `mcp__kompile__patch` — Unified Diff Patch
```
file_path: "libnd4j/include/ops/helpers/cuda/myKernel.cu"
patch: "--- a/file\n+++ b/file\n@@ -10,3 +10,4 @@\n existing line\n-old line\n+new line\n+added line\n existing line"
```
Applied via system `patch` command. Best for multi-hunk changes.

---

### 5. `mcp__kompile__glob` — Find Files by Pattern
```
pattern: "**/*.java"               # Glob pattern
path: "/home/agibsonccc/Documents/GitHub/deeplearning4j/nd4j"  # Optional directory
```
Returns paths sorted by modification time (newest first). Max 100 results.

**DL4J patterns:**
```
"**/DynamicShapePlan*.java"        # All DSP-related Java files
"libnd4j/include/ops/**/*.cu"      # All CUDA kernels
"platform-tests/**/*Test.java"     # All test files
"**/*.sh"                          # All shell scripts
"**/pom.xml"                       # All Maven POMs
"libnd4j/include/ops/helpers/**/*" # All helper implementations
"**/optimize/optimizations/*.java" # All fusion patterns
```

---

### 6. `mcp__kompile__grep` — Search File Contents
```
pattern: "elementWiseStride"       # Regex pattern
path: "/home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j"
glob: "*.cpp"                      # Optional: filter by file type
output_mode: "content"             # "content" (default), "files", "count"
case_insensitive: false
context_lines: 2                   # Lines before/after each match
```

**DL4J search patterns:**
```
# Find EWS violations:
pattern: "ews\\(\\)|elementWiseStride"
glob: "*.cpp,*.cu,*.h"

# Find raw CUDA qualifiers:
pattern: "__host__|__device__|__global__"
glob: "*.h,*.cpp,*.cu"

# Find raw OpenMP pragmas:
pattern: "#pragma omp"
glob: "*.h,*.cpp"

# Find smart pointer usage:
pattern: "unique_ptr|shared_ptr|make_unique|make_shared"
glob: "*.h,*.cpp,*.cu"

# Find direct make usage in scripts:
pattern: "\\bmake\\b"
glob: "*.sh"

# Find test locations:
pattern: "class Test.*\\{"
path: "platform-tests"
glob: "*.java"
```

---

### 7. `mcp__kompile__list` — Directory Listing
```
path: "/home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j/include/ops/helpers"
```
Returns files with size, type, and modification time. Useful for exploring directory structure.

---

## TOOL SELECTION GUIDE

| Task | Tool | Why |
|---|---|---|
| Read a known file | `read` | Direct path access |
| Create a new file | `write` | Auto-creates directories |
| Modify one spot in a file | `edit` | Targeted replacement |
| Multiple changes in one file | `patch` | Unified diff for multi-hunk |
| Rename across a file | `edit` with `replace_all: true` | All occurrences |
| Find files by name | `glob` | Pattern matching |
| Find files by content | `grep` with `output_mode: "files"` | Content-based discovery |
| Search for code patterns | `grep` with `output_mode: "content"` | Shows matching lines |
| Count occurrences | `grep` with `output_mode: "count"` | Per-file counts |
| Explore directory structure | `list` | File metadata |

## SAFETY RULES FOR DL4J

- **ALWAYS read before edit** — verify the exact string exists
- **NEVER edit generated code** — modify presets instead
- **NEVER write to files outside the project** unless explicitly asked
- **Use dry_run for replace operations** — verify before applying
- **Prefer edit over write for existing files** — less destructive
- **Check glob results before bulk operations** — verify scope