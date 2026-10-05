# ADR 0131: Framework-allocated op outputs are dense

## Status

Accepted (2026-10-04).

## Context

A shape function describes each output of an op with a shape information
(shape, strides, extras, order). Many of them describe an output with the shape
information of an input:

- 156 ops in 109 `generic/` files return an input's shape information as it is
  (`CONSTANT(inShape)`, `bufferForShapeInfo(inShape)->primary()`, `COPY_SHAPE`,
  the raw input pointer: every `*_bp` op, `check_numerics`, `identity_n`,
  `expose`, the `to_*` casts, `lu`, `solve`, `kv_cache_update`, ...), 15 more
  copy it with only the data type changed (`castToDataType`,
  `copyShapeInfoAndType` with `copyStrides`), and so do the legacy transform,
  scalar, pairwise and broadcast op classes, `DIVERGENT_OP_IMPL` and the nine
  boolean broadcastable ops;
- 26 explicit shape functions and the 129 ops of `OP_IMPL` and
  `CONFIGURABLE_OP_IMPL` (every activation, every updater, `layer_norm`,
  `cumsum`, `softmax`, ...) recompute dense strides but copy the input's extras
  word, which holds the view flag, the needs-copy flag and the copy-offset flags.

When the framework allocates an output from such a descriptor it keeps the
strides and allocates `length` elements. For a stepped view (a [4, 70] slice of
every second column of a [4, 141] array has strides [141, 2]) the output
addresses offset 3 * 141 + 69 * 2 = 561 of a 280-element buffer: every write past
element 279 corrupts the heap on CPU or device memory on CUDA. `fused_layer_norm`
was the first op found writing there (it now builds a dense descriptor itself).
Both executioners' output allocation, the SameDiff memory managers and the
dynamic shape plan did the same:

- Java: `Nd4j.createFromDescriptor` (the three `NDArrayFactory` implementations
  and the executioner twins) adopts the descriptor as the array's shape
  information, and allocates `Shape.length` elements. `CustomOp.initializeOutputs`,
  `DynamicCustomOp.computeArrays`, `OpExecutioner.allocateOutputArrays`,
  `SameDiff.executeCustomOpEagerly`, constant folding, every `SessionMemMgr`
  and `MultiBackendWorkspaceSessionMemMgr` (which wraps a workspace buffer of
  `length` elements in the descriptor) call it. SameDiff's `InferenceSession`
  already allocated C-order outputs from shape and type only, but handed
  F-ordered and empty ones to the managers.
- Native: `DeclarableOp::prepareOutputs` (`new NDArray(out, true, ...)`, twice)
  and the dynamic shape plan: the shape pre-pass placeholders (which the first
  execution reuses as the slots' outputs), the fallback allocation of a
  view-capable op whose input cannot be viewed (a stepped or F-ordered input),
  secondary and untracked outputs, the maximum-size and capacity-shift paths.
  The main path of the plan (`new NDArray(order, contigShape, dt)`) was dense
  already for ops that are not view-capable.

The view flag does harm of its own. `~NDArray` frees its buffer only
`if (_ownsBuffer && !isView)`, so an owner carrying the flag leaks it (the plan
had cleared the flag by hand in one place and described the leak in a comment).
The plan skips the pre-zeroing of an output that is a view or carries a
copy-offset flag, so a sparse-output op's fresh buffer was not zeroed. Java's
`closeable()` and `ArrayCacheMemoryMgr` treat a flagged array as one that does
not own its buffer.

A shape function's descriptor is also the description of a view: `permute`,
`transpose`, `reshape_no_copy` (and `kv_scatter`, `paged_kv_append`) return the
input's strides plus `ARRAY_COPY_OFFSET_INPUT_0`, and the Java executors and the
plan create their output over the input's buffer from it. Those strides are the
point of the descriptor and must stay.

## Decision

**An output the framework allocates is a new, dense array in the descriptor's
shape, data type and order.** The empty flag stays; the strides are the dense
ones of the order, and the view, needs-copy, padded-buffer and copy-offset flags
are dropped. An empty descriptor allocates nothing and is returned as it is, and
so is a descriptor that already describes such an array (nothing is interned or
allocated then). An output that is a view of an input is not allocated from its
descriptor and keeps it.

The rule lives where a descriptor becomes an allocation, through one helper per
language:

- Java: `Shape.allocationShapeInfo(DataBuffer)` (and `Shape.isPackedInOrder`).
  `CpuNDArrayFactory`, `JCublasNDArrayFactory` and `VulkanNDArrayFactory`
  `createFromDescriptor`, the three executioner twins and
  `MultiBackendWorkspaceSessionMemMgr` use it, which covers every caller above.
  `NDArrayFactory.createFromDescriptor` documents the contract.
- Native: `denseOutputShapeInfo(const LongType*)` in
  `libnd4j/include/helpers/DenseOutputShape.h` (a header of its own, so that no
  widely included header changes). `DeclarableOp::prepareOutputs` and every
  allocation site of the dynamic shape plan use it (one placeholder keeps a view's
  strides, see below). The plan also interns the
  cached output shapes of every output that is not the primary output of a
  view-capable op in their dense form, so the cached shape, the strides compared
  with an existing wrapper and the allocated array agree (the Triton builder
  already assumed C-contiguous cached shapes).

The shape functions do not change. The rule makes it unnecessary: the descriptor
of an output the framework allocates is only read for its shape, data type, order
and empty flag, and the descriptor of a view output has to keep its strides.
Changing the 156 + 15 + 26 shape functions and the macros to build dense
descriptors would be a far larger change (about 160 files), would not protect a
shape function written later, and would need an exemption for the view-producing
ones. The `NDArray` constructors (`copyStrides = true`) are not changed either:
hundreds of helper call sites use them to give a new array the layout of an
existing one.

### Dense, not packed

A descriptor whose strides are a permutation of packed ones (an array from
`permute`) would fit its buffer, so keeping them would be safe. Dense in the
descriptor's order is simpler to state and what every kernel that indexes an
output linearly assumes (`kv_cache_update` copies at flat offsets), and it is what
the plan's main path and `InferenceSession` already did.

### The one placeholder that keeps a view's strides

While the plan infers the shapes of a graph it publishes a placeholder for every
output, and the shape functions of the ops after it read the strides of those
placeholders (the view descriptor of a `permute`, `transpose`, `expand_dims` or
`squeeze` derives its strides from its input's). The primary output of a
view-capable op is a view at run time, with the strides its shape function
describes, so inference sees the layout the slot will have only if its placeholder
keeps them. It does, exactly as it did before this ADR, while the strides address
only its own buffer, which is what a buffer of `length` elements can hold (a
permutation of dense strides: `permute`, `transpose`, an `expand_dims` or
`reshape` of a dense array). Strides that address more (a slice of a wider array)
cannot be allocated, and that placeholder is dense like every other output. The
view, copy and copy-offset flags go either way, because the placeholder owns its
buffer. This is `viewStandInShapeInfo` in
`libnd4j/include/helpers/DenseOutputShape.h`; every output the plan allocates at
run time, including the fallback of a view-capable op whose input cannot be
viewed, is dense.

## Consequences

- Outputs of ops over views, F-ordered and permuted inputs are dense in their
  order: CPU, CUDA, Vulkan and the plan. A stepped input no longer makes the op
  write past its output.
- Allocated outputs no longer carry the view flag: they are closed and freed with
  their buffers, and the plan pre-zeroes the outputs that need it.
- `kv_scatter` and `paged_kv_append` describe an output that aliases input 0 and
  need a view; an output allocated for them is a new array (as it already was
  in the plan and in native-only execution).
- Descriptors returned by `calculateOutputShape` (and the JNI
  `calculateOutputShapes2`) are unchanged, so Java code that reads them sees what
  it saw.
- Platform tests: `DenseOpOutputTest` (every allocation route over every input
  layout, the plan on and off, outputs that are views stay views).
