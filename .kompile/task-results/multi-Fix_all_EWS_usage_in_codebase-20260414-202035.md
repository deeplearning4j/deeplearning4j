# Multi-Task Results: Fix all EWS usage in codebase

**Subtasks:** 6

---

## matmul-ews (qwen)

Subagent 'qwen' completed in 60.7s

## Summary
All 4 occurrences fixed. Replaced:
- `x->ews() == 1` → `shape::strideDescendingCAscendingF(x->shapeInfo())` (line 90)
- `z->ews() == 1` → `shape::strideDescendingCAscendingF(z->shapeInfo())` (line 96)
- `y->ews() == 1` → `shape::strideDescendingCAscendingF(y->shapeInfo())` (line 109)
- `z->ews() == 1` → `shape::strideDescendingCAscendingF(z->shapeInfo())` (line 115)

The `reshape()` view/copy decision now uses the correct contiguity check that works reliably for views, non-contiguous arrays, and DSP-allocated tensors.

**Full output (523 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260414-201507.md`
Use the `read` tool to access the full result if needed.

---

## cpu-helpers-ews (qwen)

Subagent 'qwen' completed in 211.1s

## Summary
All 5 files have been updated. Here's a summary of the changes:

1. **adjust_saturation.cpp:40** — Replaced `input->ews() == 1 && output->ews() == 1` with `shape::strideDescendingCAscendingF(input->shapeInfo()) && shape::strideDescendingCAscendingF(output->shapeInfo())`

2. **adjust_hue.cpp:39** — Same pattern as adjust_saturation.

3. **randomShuffle.cpp:88** — Replaced `const sd::LongType ews = arr->ews()` with `const bool isContiguous = shape::strideDescendingCAscendingF(arr->shapeInfo())`. Updated the `ews` usage in `fisherYates` and `mergeShuffle` calls to use `isContiguous ? 1 : arr->stridesOf()[0]` as the stride value, and removed the `* ews` offset multiplication since contiguous arrays have stride 1.

4. **lstm.cpp:253** — Replaced all six `->ews() == 1` checks (`z`, `i`, `c`, `cLast`, `f`, `h`) with `shape::strideDescendingCAscendingF(x->shapeInfo())`.

5. **random.cpp:133,183,184** — Replaced `output->ews() == 1` and `lambda->ews() == 1` with `shape::strideDescendingCAscendingF(...)` calls.

**Full output (1016 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260414-201737.md`
Use the `read` tool to access the full result if needed.

---

## ndarray-lambda-ews (qwen)

Subagent 'qwen' completed in 88.8s

## Summary
Done. Both EWS usages have been replaced:

- **Line 155**: `this->ews() == 1 && target->ews() == 1` → `shape::strideDescendingCAscendingF(this->shapeInfo()) && shape::strideDescendingCAscendingF(target->shapeInfo())`
- **Line 372**: Same replacement

**Full output (249 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260414-201535.md`
Use the `read` tool to access the full result if needed.

---

## ctc-beam-ews (qwen)

Subagent 'qwen' completed in 56.8s

## Summary
Done. Replaced all three deprecated `ews()` checks:

- **Condition**: `ews() == 1` → `shape::strideDescendingCAscendingF(shapeInfo())`
- **Error messages**: `%d == ews(1)` → `contiguous=%d`, printing the stride check result and `ordering()`

**Full output (240 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260414-201503.md`
Use the `read` tool to access the full result if needed.

---

## mlir-platform-ews (qwen)

Subagent 'qwen' completed in 305.8s

## Summary
All 7 files have been fixed. Replaced all `ews() == 1 || ews() == 0` checks with `shape::strideDescendingCAscendingF(shapeInfo())` across:

| File | Occurrences fixed |
|---|---|
| `activations_extended.cpp` | 3 (hardswish, relu6, log_softmax) |
| `activations.cpp` | 3 (relu, sigmoid, softmax) |
| `reductions.cpp` | 9 (sum, mean, max, min, prod, variance, stdev, norm2, logsumexp) |
| `unary_ops.cpp` | 4 (exp, sqrt, abs, square) |
| `comparison_ops.cpp` | 3 (equals, greater, less) |
| `binary_ops.cpp` | 4 (add x+y, multiply x+y) |
| `matmul.cpp` | 2 (a, b) |

**Total: 28 occurrences replaced.** Zero `ews()` usages remain in the MLIR platform directory.

**Full output (659 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260414-201912.md`
Use the `read` tool to access the full result if needed.

---

## misc-ews (qwen)

Subagent 'qwen' completed in 389.2s

## Summary
All three files are fixed. Here's a summary of changes:

1. **`MlxIRBuilder.cpp`** (line 280): `arr->ews() == 1` → `shape::strideDescendingCAscendingF(arr->shapeInfo())`. Added `#include <helpers/shape.h>`.

2. **`NativeDynamicShapePlan_gpubackend.cpp`** (line 360): `out->ews()` → `shape::strideDescendingCAscendingF(out->shapeInfo()) ? 1 : 0`, stored in `outputEws` with a comment explaining EWS is deprecated. The downstream consumer at line 405 (`shapeInfo[len-2] = recipe.outputEws`) now correctly writes `1` for contiguous arrays.

3. **`ViewRecipe.h`** (line 72): Updated field comment to document the new semantics (`1` if contiguous, `0` otherwise).

4. **`Context.cpp`** (lines 362, 376): Both debug `sd_printf` calls changed from `ews: [%i]` with `array->ews()` to `contiguous: [%i]` with `shape::strideDescendingCAscendingF(array->shapeInfo()) ? 1 : 0`. Added `#include <helpers/shape.h>`.

**Full output (901 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260414-202035.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 6/6 subtasks completed successfully.