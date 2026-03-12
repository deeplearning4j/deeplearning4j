# Test Verification Journal

## Fixes Applied So Far
1. **Shape.wholeArrayDimension()** - Added Integer.MAX_VALUE sentinel recognition
2. **Shape.normalizeAxis()** - Added Integer.MAX_VALUE → empty array conversion
3. **CpuOpContext.setIArguments/setBArguments/setTArguments/setDArguments** - Clear native context when empty args
4. **reduce_prod_bp** (C++) - Use applyTrueBroadcast instead of assign for scalar→tensor
5. **SymbolicShapeRanges.cpp** - Copied from gpu/ to cpu/ for CPU linker
6. **JCublasNDArrayFactory.pullRows()** - Fixed INT32→INT64 index type mismatch in CUDA kernel
7. **reduce_mean_bp** (C++) - Use applyTrueBroadcast instead of `*=` for broadcast multiply
8. **reduce_norm1_bp** (C++) - Same fix as #7
9. **reduce_norm2_bp** (C++) - Same fix as #7
10. **reduceStDev_bp** (C++) - Same fix as #7
11. **reduceVariance_bp** (C++) - Same fix as #7
12. **create_view DECLARE_SHAPE_FN** - Implemented proper shape computation (was returning input shape)
14. **NDArray constructors 6, 12, 14** - Added `_offset = offset` (offset parameter was ignored, views read from position 0)
15. **create_view CUSTOM_OP_IMPL** - Fixed `output->assign(viewArray)` → `output->assign(&viewArray)` (pointer not value)
16. **create_view negative index handling** - Added `if (end < 0) end = dimSize;` for negative interval ends
17. **create_view stride reading** - Read actual stride from `indexIndices[2]` instead of `indexVector[2]` (hardcoded header value)
18. **create_view view strides** - Use ShapeDescriptor with computed outputStrides instead of C-order strides for view NDArray
13. **Validation tests** - Replaced unreliable `sd.grad("in").getArr()` with `g.get("in")` from calculateGradients return map (matches GradCheckUtil pattern)
19. **shape::maxIndToMinInd** (shape.h) - Off-by-one `[i+1]` → `[i]` when minRank != maxRank (broke prelu alpha mapping)
20. **pullRowsGeneric** (NativeOps.cpp) - Used idx2 (row index) instead of i (element index) for INDEX2COORDS; used hX/hZ instead of rX/rZ (TAD-offset pointers)
21. **Loops.h bounds check** - Removed invalid `xOffset >= xLen` / `zOffset >= zLen` check that compared physical stride-based offsets against logical element count (broke strided views)
22. **ForwardExecutionDAG.java** - Made cycle detection control-flow-aware (CONTROL_FLOW_OP cycles are expected)
23. **einsum.cpp** - Added missing `#include <ops/declarable/DeclarableOp.h>` for conditionHelper

## C++ Build Status
- CPU rebuild: COMPLETE (with all fixes including #19-#21)
- CUDA: NOT rebuilt yet (fixes #4, #7-#11, #19-#21 not in .so yet)

## SameDiffTests — Complete Isolated Test Results (CPU)

| # | Test | CPU | Category | Root Cause / Status |
|---|------|-----|----------|-------------------|
| 1 | testGatherOp | **PASS** | pullRows | FIXED: pullRowsGeneric inner loop (fix #20) |
| 2 | testActivationBackprop | **PASS** | gradient | PASS on CPU. CUDA-ONLY issue: SOFTPLUS grad returns softplus(x) not derivative |
| 3 | validateProdDiff | **PASS** | test infra | FIXED: test used unreliable `sd.grad().getArr()`, now uses `g.get("in")` |
| 4 | validateStdevDiff | **PASS** | test infra | FIXED: same as #3 |
| 5 | testMseBackwards | **PASS** | reduce_mean_bp | FIXED: applyTrueBroadcast for broadcast multiply |
| 6 | testCreateViewBp | **PASS** | create_view | FIXED: previous create_view fixes resolved this |
| 7 | testCtc | **PASS** | ctc_loss | FIXED: CTC loss implemented |
| 8 | testPReLU | **PASS** | prelu | FIXED: shape::maxIndToMinInd off-by-one (fix #19) |
| 9 | testIndexInterval2 | **PASS** | create_view | FIXED: Loops.h bounds check broke strided views (fix #21) |
| 10 | testPoint | **PASS** | create_view | FIXED: NDArray offset constructors (fix #14) |
| 11 | testIndexInterval | **PASS** | create_view | FIXED: negative index handling (fix #16) |
| 12 | testWhile | ERROR | control flow | "Operation not ready. Missing dependencies" — needs frame-aware execution engine |
| 13 | testForLoop | ERROR | control flow | Same as #12 |
| 14 | testIf | ERROR | control flow | "VALUE NOT FOUND IN CONTEXT" — needs control flow frame execution |
| 15 | testExecDiffShapesIndexAccumAlongDim | **PASS** | indexAccum | FIXED |
| 16 | validateMinDiff | **PASS** | test infra | FIXED: same as #3 |
| 17 | validateMaxDiff | **PASS** | test infra | FIXED: same as #3 |
| 18 | validateVarDiff | **PASS** | already correct | Already used `g.get("in")` pattern |

## Summary
- **15/18 PASS on CPU**: All non-control-flow tests pass
- **3 ERRORS**: #12 (testWhile), #13 (testForLoop), #14 (testIf) — control flow needs frame-aware execution engine (Enter/Exit/Switch/Merge/NextIteration ops)
- The old LogicWhile/LogicConditional implementations were deleted during DSP migration. These need to be reimplemented in the ForwardExecutionDAG framework.

## Key Pattern: Gradient Retrieval
- **CORRECT**: `Map<String,INDArray> g = sd.calculateGradients(...); INDArray grad = g.get("varName");`
- **UNRELIABLE**: `sd.grad("varName").getArr()` — returns null because gradient arrays aren't stored back on SDVariables
- GradCheckUtil uses the correct pattern exclusively

## Remaining Work
1. Control flow (#12-#14) — requires frame-aware execution engine (substantial feature)
2. CUDA rebuild needed to pick up fixes #19-#21
