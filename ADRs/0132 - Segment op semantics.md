# ADR 0132: Segment op semantics

## Status

Accepted (2026-10-04). Renumber on merge if 0132 is taken (0131 is the dense-outputs ADR of the same batch).

## Context

`segment_{sum,mean,max,min,prod}`, `unsorted_segment_{sum,mean,max,min,prod,sqrt_n}`, their
`_bp` ops, `segment_softmax` and `sequence_mask` disagreed between the CPU and CUDA artifacts
and with TensorFlow, and CUDA failed outright past small sizes:

- **Launches.** The vector kernels of the sorted, unsorted and sqrt_n ops launched a block per
  class with a thread per id (rounded up to a power of two), so they failed from 1025 ids. They
  also requested `numClasses * 32 + 32` bytes of dynamic shared memory (`(numClasses + 1) * 64` for
  sqrt_n), past the 48 KB limit from 1536 classes (768 for sqrt_n), where a tree reduction needs one
  partial per thread. The id validation put one thread per id in a single block (1025 again), and so
  did `sequence_mask` for its lengths. The vector backprop launched 1 + the gradient's length threads
  (1025 for a gradient of 1024); the N-D backprop had its blocks and threads swapped and failed once
  the input had more than 1024 elements.
- **Validation.** CUDA's sorted-id check was `return true`; its unsorted check seeded the maximum it
  looks for with the expected value, so an id equal to the number of classes passed, and no backend
  rejected a negative id, which then indexed out of bounds in the kernels. The check used the legacy
  `cudaMemcpy` and a pool block from the default stream (illegal inside a capture).
- **Values.** An empty segment was 0 on CPU and the "neutral" value of the reduction on CUDA: `-max` of
  the type for a max (which is 1 for an unsigned type and one above the lowest value for a signed one)
  and `infOrMax` for a min (+inf for float and double). The CPU unsorted max of an empty class was -max
  for a vector and +max for rows x columns. The CPU sorted max, min and prod never wrote a first
  segment of one row, nor anything for a single row.
- **Accumulation.** The N-D forward kernels combined into the output through `templatemath.h` atomics:
  the 8 and 16 bit `sd_atomicMin` works on a local copy and stores the old value back (the minimum is
  never stored; the HALF and BFLOAT16 minimum backprop then found no element equal to it and returned
  zeros), and `sd_atomicAdd<bfloat16>` updates the neighbouring half-word. A CUDA mean divided every
  element before its atomic add, so an integer mean was truncated per element.
- **Layouts.** The CPU vector paths read `buffer[i]` and its N-D paths read each TAD's elements
  linearly (wrong for F ordered, permuted and stepped inputs); the CUDA validation and range kernels
  read the ids linearly; `unsorted_segment_sqrt_n` read `dataBuffer()`, which drops a view's offset.
  The ops did not check their output types (a FLOAT output for an INT sum was reinterpreted).
- **Backprop.** Max/min backprops compared with a tolerance (1e-6 / 1e-5, differing by backend); the
  CUDA sqrt_n backprop divided by a FLOAT square root for DOUBLE; the product's gradient, prod / x,
  is 0 / 0 = NaN for every zero element.
- **sequence_mask.** CUDA compared an unsigned 32 bit index with the length (a negative int length was
  true everywhere), indexed with 32 bit values and wrote only the true elements; the execution took
  the width from the position of the longest length (when a `maxlen` input was not larger than it) or
  from an integer argument that the shape function reads as the data type.
- **segment_softmax.** `gridDim.y = features` (at most 65535), ids read as `int*` on both backends
  whatever their type, features indexed with two strides (wrong from rank 3), every block scanned the
  ids from the start for its segment, and the CPU helper never used K.

## Decision

### One definition of the arithmetic

`helpers/segment_semantics.h` holds the policy structs (`SegSum`, `SegProd`, `SegMax`, `SegMin`,
`SegMean`, `SegSqrtN`; `GradSum`, `GradMean`, `GradSqrtN`, `GradCompare`, `GradProd`), the accumulator
traits, the layout descriptor `SegMat` and the host range functions. The CPU helpers call the host
range functions; the CUDA kernels call the same policy functions per element, so the backends
cannot drift apart.

| | sorted (`segment_*`) | unsorted (`unsorted_segment_*`) |
|---|---|---|
| empty segment: sum, mean, sqrt_n | 0 | 0 |
| empty segment: prod | 1 | 1 |
| empty segment: max / min | 0 / 0 | lowest / highest value of the type (-FLT_MAX / FLT_MAX; INT_MIN / INT_MAX; 0 / 255 for UINT8) |
| segment with rows | exactly those rows reduced (max of only -inf is -inf) | same |
| NaN | a NaN makes a max or min NaN, wherever it is | same |

Accumulators: HALF, BFLOAT16 and FLOAT sum and multiply in FLOAT, DOUBLE in DOUBLE; integers sum
and multiply modulo 2^32 (up to 32 bits) or 2^64, which is the narrow type's wrap-around; max and
min compare in a type that keeps the order (FLOAT for the 16 bit floats, 32/64 bit integers for the
integers); mean and sqrt_n of integers sum in DOUBLE. A mean is `sum / count` and a sqrt_n is
`sum / sqrt(count)`, once, on the finished accumulator, in the accumulator type. The output of
sum/prod/max/min has the input's type (the ops reject another); a mean's output is floating point
(an integer input gives the default floating type, FLOAT, unless the caller supplies another
floating output).

### Ids

Ids are any integer dtype, rank and layout (read through their shape and strides, as one logical
sequence); both backends read them into one dense int64 sequence first. Sorted ops: no negative id and
no id smaller than the one before it (the number of classes is the last id + 1). Unsorted ops: every
id in `[0, numSegments)`. Violations throw, naming the offending id; there is no TensorFlow-style
silent drop of negative ids. The kernels never index with an id outside the range of classes (they
drop the row), because during a CUDA graph capture the host cannot read the validation back.
Backprops do not validate (a gradient graph would synchronize on every call); an out-of-range id gets
a zero gradient, an unsorted `prod_bp` keeps its up-front check.

### Backprops

The gradient of a max/min goes, whole, to every element equal to the segment's extremum (an exact
comparison; the extremum is one of the elements); sum broadcasts, mean divides by the count, sqrt_n
by its square root. The gradient of a product is the product of the other elements of the segment
(TensorFlow's three cases): without a zero in the segment's column it is `prod / x`; with exactly one
zero only that element has a gradient, the product of the nonzero ones; with two or more none has. It
is built from two reductions of the input (`SegProdNonZero`, the product of the nonzero elements, and
`SegZeroCount`) instead of dividing the product by the element. The second output of every `_bp` op is
a copy of the ids. The gradient must have the rank of the input and equal trailing dimensions, and the
output's type.

### CUDA

- The launch producers of `LaunchDims.cu` for the family return `(blocks, threads, shmem 0)`;
  every kernel strides over its elements with 64 bit indices, so they only cap the grid. Environment
  overrides (`GRID_SIZE_<KEY>`, `BLOCK_SIZE_<KEY>`) are validated.
- Sorted ids: boundaries per class from one kernel (no atomics), then a gather. A vector input is
  reduced by one block per segment with a fixed shared-memory tree (deterministic per launch); rows x
  columns inputs by one thread per output element in row order (bit-identical to the CPU).
- Unsorted ids: a scatter into a `[classes, columns]` accumulator of the accumulator type with
  atomics that do not go through `templatemath.h` (native `atomicAdd` where it exists, a
  compare-and-swap loop on the bit pattern otherwise), then one kernel that finishes every segment.
  Float results depend on the atomic order (to rounding).
- Validation: a check kernel records the first violation; one 24 byte report is copied back with a
  stream-ordered `cudaMemcpyAsync` and a stream synchronization (skipped during a graph capture).
  Temporaries come from `PointersManager` (capture-aware, stream-ordered frees); no launch
  synchronizes. No op reads an id on the host (a host read would synchronize a captured stream).
- `sequence_mask` writes every element, one thread per output element.

### sequence_mask

The width is decided by the shape function only (the second input or the first integer argument, never
below the longest length; a length below zero counts as zero) and the helpers take it from the
output. `mask[..., j] = j < length` with a signed 64 bit comparison for signed types and an
unsigned one for unsigned types; every element is written.

### segment_softmax

A segment is the set of rows with one id; the ids must be sorted and lie in `[0, K)` (validated with the
sorted check and the range check, both capture-aware), the logits are read through their strides for
any rank. One block (CUDA) / one task (CPU) per (segment, feature) pair, grid-stride; arithmetic in
FLOAT (DOUBLE for DOUBLE); the per-segment row ranges come from the ids once, not from a scan per
block.

## Consequences

- Sorted max/min of an empty segment is 0 on CUDA (it was -max / inf), unsorted max/min of an empty
  segment is the lowest/highest value of the type on both backends (CPU N-D was +max / max for
  max).
- Invalid ids now throw on CUDA (they used to be accepted, `id == numSegments` included) and on CPU
  for negative ids; `segment_softmax` rejects unsorted ids and ids >= K.
- An integer `segment_mean` / `unsorted_segment_mean` returns FLOAT (it threw: the descriptor allows
  floating outputs only); outputs of another type than the op accepts are rejected instead of being
  written through a reinterpreted pointer.
- Max/min backprops no longer give gradient to elements merely close to the extremum; the gradient of
  a product through a zero element is the product of the others (it was NaN).
- Not changed: tie handling gives each tied element the whole gradient (TensorFlow divides it);
  `platform/mlir/embedding/embedding_ops.cpp` (HAVE_MLIR builds) and `platform/mps/mps_embedding.mm`
  (Apple MPS builds) register their own helpers for `segment_{sum,mean,max,min}` (MPS also `segment_prod`)
  and `unsorted_segment_sum` that bypass the generic ops' validation and these semantics, and the
  OpenVINO graph backend composes `sequence_mask` from its first integer argument alone.
- Limits: the CUDA sorted rows x columns gather has one thread per output element, so a few classes
  with very many rows parallelize poorly (correct, and as fast per element as the CPU); the unsorted
  ops' float sums depend on the atomic order; a vector's sorted sum uses a fixed tree whose rounding
  differs from the CPU's sequential sum in the last bits (exact for integers and for the small
  integer-valued floats the tests use).

## Testing

`platform-tests` `SegmentSemanticsTest`: every op, sorted and unsorted, over every numeric type,
vector / matrix / rank 3 shapes, ids of both widths, against a Java reference; 1025+ ids over 767 / 1535
/ 1536 / 2048 classes; backprops (gradient of 1024, data of 1025 elements, ties, products through
zeros); empty-segment values; unsigned zeros; HALF/BFLOAT16 accumulation; integer wrap and exact integer
means; NaN and infinities; invalid ids; strided, offset, F ordered and transposed inputs, stepped ids,
`[1, N]`, caller supplied output views; row-order bit-exactness; `sequence_mask` (1025 lengths, wide
masks, negative and unsigned lengths, layouts); `segment_softmax` (70001 features, INT64 ids, rank 3,
views, backprop).
