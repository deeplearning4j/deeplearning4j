# ADR 0128: Scatter and gather index semantics

## Status

Accepted (2026-10-03).

## Context

The scatter ops (`scatter_add`, `scatter_sub`, `scatter_mul`, `scatter_div`,
`scatter_upd`, `scatter_max`, `scatter_min`, the `scatter_nd*` family) and
`gather` left several questions to whichever kernel ran:

- **Repeated indices.** CUDA's per-update kernel let threads holding updates for
  one destination race: a plain read-modify-write lost all but one of them, for
  every op including addition. The CPU helper split the indices across threads
  the same way unless `lock` was set. `TestMiscOpValidation.testScatterOpGradients`
  ran with its gradient check disabled as "known failures".
- **Ordering.** The ordered ("lock") CUDA kernels ran one thread per output
  element, each scanning every index: O(output length x indices), which made
  `scatter_upd` (lock on by default) and `get_rows_bp` over an embedding table
  billions of iterations.
- **Layouts.** The ops accept updates shaped `indices.shape + output.shape[1:]`,
  `[indices.length] + output.shape[1:]` for vector indices, or `indices.shape`
  for a vector output. CUDA's 1D shortcut read `x[i]` past the end for an output
  of shape `[1, n]`, and its general path read rank-2 vector indices with the
  updates' coordinates.
- **Out-of-range indices.** Unvalidated (`checkIndices` false, and always during
  DSP replay) they wrote out of bounds on CUDA and threw or corrupted on CPU.
  `gather` clamped them to the nearest slice on some paths (another slice's
  data) and skipped them on others (an unwritten output).
- **Empty indices.** Under the default `EMPTY_SKIP` the ops never ran, leaving a
  non-in-place output unwritten instead of equal to the input; `scatter_nd`
  relied on a zeroed allocation.
- **Types.** CUDA read the updates in the updates' type and wrote the output as
  that type, overrunning a narrower output.
- **Gradients.** `ScatterMul`, `ScatterDiv`, `ScatterUpdate`, `ScatterMax` and
  `ScatterMin` differentiated as though no index repeated and no value tied; the
  `ScatterNd*` ops had no gradient at all. `GradCheckUtil` counted a gradient
  check as passing whenever either gradient was exactly zero, so an analytic
  gradient of 0 against a non-zero numerical one passed.

## Decision

### Pairing

Index k (the k-th element of `indices` in logical order) names slice
`indices[k]` along the output's first dimension; update slice k is the k-th run
of `sliceLen = output.length / output.shape[0]` elements of `updates` in logical
order; element p of a slice is its p-th element in logical order. This one rule
serves every accepted layout. For `scatter_nd*`, index row r (the r-th run of
`indexDepth` elements) names the output's leading `indexDepth` coordinates,
flattened to a destination slice.

### Repeated indices

Updates sharing a destination apply one after another in index order, for every
op, whether or not `lock` is set: the last write wins for the update ops, and
products, quotients, maxima and minima see every update.

- CUDA applies independent updates one thread per update. Addition and
  subtraction accumulate there with atomics. For the other ops a kernel marks
  each destination in a bitmap and sets a device flag when one repeats; both the
  per-update kernel and the ordered kernel are launched and the flag chooses
  which one does the work. The flag is never read on the host: no stream
  synchronization, and the choice is recorded under graph capture.
- The ordered kernels give each thread one element position p of every slice
  and walk the indices in order: O(sliceLen x indices) work instead of
  O(output length x indices).
- The CPU helpers detect a repeat on the host and then run the same ordered
  walk, parallel over slice positions.
- `lock` asks for index-order accumulation also for addition and subtraction:
  bitwise reproducible sums (`get_rows_bp` sets it). Without it CUDA's atomic
  sums may round differently from run to run.

### Graph backends

- Triton lowers `scatter_nd` and `scatter_nd_update` with one owner per output
  element: a program loads its block (zeros or the input), walks every index
  row in order, applies the rows naming its lanes, and stores the block once.
  No phases and no atomics. This costs N x rows lane tests, which suits short
  walks (KV-cache and window updates). A phased lowering would need a launch
  barrier between a copy phase and a row-parallel scatter phase.
- Vulkan has no host helpers, so every scatter op is a device kernel
  (`VulkanLoweringContract::INDEXED_SLICE_UPDATE`, lowered by `IndexedSliceUpdateToSpirv`;
  eager execution and DSP replay record the same module). `scatter_add`, `scatter_sub`,
  `scatter_mul`, `scatter_div`, `scatter_upd`, `scatter_max`, `scatter_min`,
  `scatter_nd_add`, `scatter_nd_sub`, `scatter_nd_update` and `get_rows_bp` share one
  schedule whose combine is the op's recipe. There is one invocation per element
  position of an output slice. It writes its element of every slice (a copy of the
  reference input, or zeros for `get_rows_bp`; nothing when the output is the input)
  and then walks the index rows in order, applying the combine to the slice each row
  names and skipping a row with a coordinate outside the output (compared in the
  indices' own width, so a 64-bit index cannot wrap into range). An invocation never
  touches another's elements, so repeated indices apply in index order with no
  atomics and no flag, `lock` or not. The cost per invocation is O(slices + index
  rows), the positions run in parallel (a `[numRows, D]` table updated by N rows runs
  D invocations of numRows + N steps), so the walk is serial per element position, as
  CUDA's ordered kernel is. The updates are cast to the output's type first; indices
  are 32-bit, or 64-bit on a device with shaderInt64, and 64-bit or half payloads need
  shaderInt64, shaderFloat64 or shaderFloat16 with 16-bit storage. With no index rows
  the output is a copy of the input and the indices and updates are not even bound
  (`scatter_nd` and `get_rows_bp` write zeros). The device cannot raise the error that
  `checkIndices` asks for, so a scatter op with `checkIndices` true is rejected, not run
  unchecked. `scatter_nd` itself (zeros plus add, `IndexedAccumulationToSpirv`) keeps
  its single serial invocation, with the same range rule and the same cast. A
  buffer shared between the output and the indices or updates, or a partly
  overlapping view of the reference, is rejected.
- Vulkan's `gather` takes any axis and indices of any rank (the second input, 32- or
  64-bit) or the integer-argument form (`[axis, index, ...]`, the indices frozen
  into the kernel, at most 2048 of them), one invocation per output element (per
  output position outside the gathered axis in the integer-argument form). An index
  outside the axis gathers zeros, where the lowering used to clamp it to the nearest
  slice. `checkIndices` is not honored there (a device kernel has no error channel and
  its default is true), so an out-of-range index always gathers zeros.

### Out-of-range and empty indices

An unvalidated index outside the output is skipped by every scatter op and
backend, and `gather` writes zeros for it (TensorFlow's GPU semantics), on CPU,
CUDA and Vulkan alike. `checkIndices` still turns either into an error on CPU and
CUDA; Vulkan cannot raise it (see Graph backends).

With no indices the output is the input (`scatter_nd`: zeros). The scatter ops
declare `EMPTY_EXECUTE`, and `scatter_nd` clears its own output.

### Types

Updates of another type are cast to the output's type before the kernels run.

### Gradients

With L the loss and `g = dL/dout`:

- mul: `dL/du_k = g * ref * (the other updates at u_k's index)`, computed from
  the product of the non-zero updates so that no zero is divided by.
- div: `dL/du_k = -g * out / u_k`.
- update: only the update written last at an index receives `g`; the replaced
  slices of ref receive 0.
- max / min: `g` is split evenly among ref and the updates that reach the
  extreme at an element.
- `scatter_nd_add/sub`: `±gatherNd(g)`; `scatter_nd_update` as update;
  `scatter_nd`: `gatherNd(g)`.

`GradCheckUtil.checkGradients` treats only two zero gradients as agreeing; one
zero against a non-zero gradient passes only under the minimum absolute error.

## Consequences

- Scatter results no longer depend on thread scheduling; a race-free result
  needs no `lock`, and `lock` costs O(sliceLen x indices) rather than
  O(output x indices).
- Ops with repeated indices pay one bitmap pass over the indices on CUDA
  (no synchronization) and one host pass on CPU.
- Gradient checks may now fail where an op's gradient was silently zero.

Tests: `ScatterSemanticsTest`, `GatherOutOfRangeTest`, `ScatterGradientTest`,
`TritonScatterNdLoweringTest`, and
`TestMiscOpValidation.testScatterOpGradients` (now with repeated indices and the
gradient check on).
