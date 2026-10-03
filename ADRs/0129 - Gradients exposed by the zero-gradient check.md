# ADR 0129: Gradients exposed by the zero-gradient check

## Status

Accepted (2026-10-03).

## Context

ADR 0128 changed `GradCheckUtil.checkGradients` to treat only two zero gradients
as agreeing. Until then an analytic gradient of exactly 0 passed against any
numerical gradient, and the op validation suites hid several broken gradients
behind it:

- `replaceWhere(x, y, condition)` gave `y` the gradient of `x`. SameDiff builds
  the gradient graph on a copy, and the copy is made through FlatBuffers, which
  stores a legacy op as its op type and number. `CompareAndReplace` (the array
  form) and the one-input `CompareAndSet` (the value form) are both
  `TRANSFORM_SAME` 13, and the number-to-class table named only `CompareAndSet`,
  so every copy rebuilt `CompareAndReplace` as `CompareAndSet`. A two-input
  branch in `CompareAndSet.doDiff` then handed both inputs the "not replaced"
  gradient.
- `minimum_bp` returned zeros on CUDA. Its helper applied `std::function`
  lambdas, which the CUDA backend runs on the host, and then registered the
  outputs as device-written, so the host result was dropped for the untouched
  device copy. The CPU `minimum_bp` and `maximum_bp` compared the two `NDArray*`
  pointers instead of the values for a scalar `y`, and for a broadcast `y`
  compared against `x`'s gradient instead of `x`.
- `weighted_cross_entropy_with_logits` with per-class weights overwrote its
  weights and targets inputs and squared the targets on the CPU, and ignored the
  weights and dropped its host result on CUDA.
- `softmax_cross_entropy_loss_with_logits_grad` added `1e-6` to every label, so
  rows whose labels are all 0 (loss 0 whatever the logits) got a gradient of
  `softmax * classes * 1e-6`.
- The NDArray lambda loops (`applyLambda`, `applyPairwiseLambda`,
  `applyTriplewiseLambda`, the indexed forms; CPU and the CUDA host loop) added
  each array's view offset to `bufferAsT()`, which already points at the view's
  first element: every view with a non-zero offset was read and written at twice
  its offset.

A test disabled since 2023 (`SameDiffSpecifiedLossVarsTests.testTrainingDifferentLosses`,
its assertions rewritten to expect that training changes nothing) hid one more:
`fit` and `calculateGradients` reuse an existing gradient function, and changing
the loss variables never dropped it, so after `setLossVariables` training and
gradients still came from the gradient function of the previous losses.

## Decision

### Ties in elementwise extrema

For `z = max(x, y)` (and `min`), `dL/dz` goes to the input `z` takes its value
from; where `x == y` each input takes half. The two gradients then always sum to
`dL/dz` (`max(x, x)` has derivative 1), the same rule ADR 0128 gives
`scatter_max` and `scatter_min`. Previously both inputs took the whole gradient
at a tie.

`maximum_bp` and `minimum_bp` share one implementation
(`helpers/impl/minimax.cpp`) built from array operations: broadcast comparisons,
casts, products and a sum over each input's broadcast axes. It replaces the CPU
lambda version and the two CUDA files.

### Helpers compute with array operations on the arrays' device

A CUDA helper must not compute through the NDArray lambda API: the function runs
on the host, a later `registerSpecialUse` on its output discards the result, and
a CUDA graph capturing the op records none of the work. `apply_sgd`,
`thresholdedrelu` and `weighted_cross_entropy_with_logits` now use scalar,
pairwise and broadcast operations (`thresholdedrelu` keeps NaN and `-inf` at 0
through the pairwise `CompareAndSet`), and `weighted_cross_entropy_with_logits`
shares one implementation between CPU and CUDA
(`helpers/impl/weighted_cross_entropy_with_logits.cpp`), writing its output only
after reading its inputs because the op may run in place.

### Element offsets of views

`bufferAsT()` and `specialBuffer()` point at a view's first element; offsets
from them come from the strides alone. `NDArray::getOffset(i)` counts from the
start of the data buffer and is only for the unshifted base. The lambda loops
and the CUDA buffer printer follow this.

### Legacy op identity in FlatBuffers

A legacy op node is rebuilt from the class registered under its recorded op
name when that class is a legacy op of the recorded type; the type and number
decide only for nodes without a name and for names that belong to another kind
of op (the custom `DropOut` wrapper shares `dropout` with the legacy random op).
`CompareAndReplace` and `CompareAndSet` restore their condition mode from the
enum name a deserialized graph holds. A SameDiff `CompareAndSet` has one input,
and its gradient is that input's alone.

### Ops rebuilt from their arguments

Every op the gradient graph copies must restore from its arguments every field its
`doDiff` reads. `CumProd` restored nothing (its gradient scanned no axes),
`CumSum` lost `reverse` and read its axes from the wrong argument (and `addArgs`
repeated the axis list once per axis), `ClipByNorm` and `ClipByAvgNorm` lost
their dimensions and clip value. Each now implements `configureFromArguments`.

### tensormmul

`tensormmul_bp` folds the operands into matrices and forms each gradient as one
product (`dA = dC * B`, `dB = A^T * dC` over the free and contracted axes). A
scalar gradient at the output (a loss variable that is not itself a scalar) is
spread over every output element; it used to leave both gradients 0.

The boolean arguments `transposeX`, `transposeY`, `transposeZ`, which the native
op never read, transpose (reverse the axes of, as numpy's `.T`) the first input,
the second input and the result; the contracted axes refer to the transposed
inputs. The forward op reads the inputs and writes the result through transposed
views, and the gradient differentiates the same product through the same views.

### dot_product_attention_v2: Keras semantics and what the backward reads

The op mirrors Keras' `Attention` layer: `weights = softmax(scale * Q K^T + value
and causal mask)`, `result = (dropout(weights) @ V) * queryMask[..., None]`, and
the weights it returns are not query-masked. It used to multiply the returned
weights by the query mask instead (leaving the result unmasked) and then read
those weights back in the backward as the softmax output: every gradient of a
masked query was zero.

Dropout's argument is the drop probability (the Keras rate). The `dropout` op
takes a keep probability unless its inverted flag is set, so the attention
helper sets the flag, and kept weights are scaled by `1 / (1 - rate)` as Keras
does. The dropout output (op output 3) holds that multiplier: 0 or
`1 / (1 - rate)`, and 1 everywhere when no dropout ran. The `dropout` op no
longer zeroes its outputs before writing them (zeroing an in-place output erased
the input, so attention dropout produced all zeros), writes every element, and
its gradient passes through at keep probability 1 (it returned zeros).

The backward differentiates what the forward recorded: the weights and logits it
returned (op outputs 1 and 2) already carry every mask, the causal mask, dropout
and an additive bias, and `AttentionHelper::doAttentionBp` prepares the masks
exactly as `doAttention` does. Rank 4 runs it on the head-major
`[batch * heads, seq, dim]` layout the forward's helper path uses (KV heads
repeated for grouped-query attention, their gradients summed back). The flash
recomputation, which knows no masks, dropout or bias, runs only when the forward's
weights were not materialized, and then masks or dropout are an error.

Gradient checks start every forward pass (analytic and numerical) from the same
`Nd4j.getRandom()` state, so random ops draw the same values in each and the two
gradients are taken of one function.

### The gradient function follows the loss variables

A gradient function differentiates the loss variables it was built for. When
`setLossVariables` or `addLossVariable` changes the set, an existing gradient
function is dropped and the next `fit` or gradient calculation builds it for the
current losses (order does not matter: the losses are summed). A gradient
function restored with a graph (a load or `dup()`) carries no record of its
losses, so the loader restores the loss variables before the sub-instances: the
restored function differentiates the losses present when it appears, and only a
later change drops it.

### Scans and clipping by norm

`cumprod_bp` is the cumulative sum of `g * y` against the scan direction,
exclusive when the forward scan is, divided by `x` (TensorFlow's form, undefined
at `x = 0`); without axes it scans the whole array as the forward op does.

For `y = x * c / a` with `a > c`, where `a` is the norm `|x|` or the average norm
`|x| / n`, `dL/dx = (c / a) * (g - x * dot(g, x) / |x|^2)`: the second term
divides by the plain norm squared also for the average norm (it used `a^2`). The
CUDA kernel takes `dot(g, x)` per tensor instead of the sum of `x`, which equals
it only for a uniform gradient. `ClipByAvgNorm` gains its gradient op
`ClipByAvgNormBp` (`clipbyavgnorm_bp`).

## Consequences

- Graphs saved before this change load with the right legacy op classes,
  including `CompareAndReplace`.
- At a tie of `max`/`min` each input now receives half of the gradient.
- `apply_sgd`, `thresholdedrelu` and the weighted cross entropy no longer make a
  host round trip on CUDA and can be captured in CUDA graphs.

- `tensorMmul` with transpose flags now computes what the flags say.
- Changing the loss variables rebuilds the gradient function on its next use;
  updater state is kept.

Tests: `LegacyOpSerdeIdentityTest`, `ExtremumGradientTest`,
`ClosedFormHelperOpsTest`, `LambdaHelperViewTest`, `TensorMmulGradientTest`,
`TensorMmulTransposeTest`, `CumProdGradientTest`, `CumSumGradientTest`,
`ClipByNormGradientTest` (now also the average norm),
`SameDiffSpecifiedLossVarsTests` (`testTrainingDifferentLosses` enabled with its
original assertions, `testGradientsFollowLossChange`), and the op validation
suites (`TestTransformOpValidation.testReplaceWhereArray` now checks its output
too).
