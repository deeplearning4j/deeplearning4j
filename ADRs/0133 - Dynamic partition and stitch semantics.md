# ADR 0133: Dynamic partition and stitch semantics

## Status

Accepted (2026-10-04). Renumber on merge if 0133 is taken (0131 and 0132 are ADRs of the same batch).

## Context

`dynamic_partition`, `dynamic_partition_bp` and `dynamic_stitch` disagreed between the CPU and CUDA
artifacts, and CUDA left most of the work undone:

- **dynamic_partition_bp.** The op body never called `helpers::dynamicPartitionFunctorBP`. It looped on the
  host with `e<double>(i)` / `p(i, v)` per element (on CUDA a device-to-host and a host-to-device transfer per
  element) and moved scalars, so for data with slices of a row or more only the first elements of the
  gradient were written. The CUDA helper had an empty body, and the CPU helper addressed raw buffers
  (views and F order wrong) and wrote an output (`outputList[1]`) the op does not have.
- **dynamic_stitch_bp.** There is no such op: `DynamicStitch.doDiff` builds the gradient with `gather`.
  `helpers::dynamicStitchFunctorBP` had no caller and threw "Not implemented yet" on both backends.
- **Launches.** The CUDA stitch kernel for slices never read `blockIdx`: input e was copied by thread e only,
  which copied only the slice elements j = e mod blockDim, so every row longer than one element was left
  unwritten. The scalar stitch kernel raced between inputs sharing an index. The scalar partition kernel
  sized its dynamic shared memory from the literal 256 instead of the block size, scanned each tile with
  one thread, and wrote its outputs ignoring their strides; the slice kernel scanned every index serially
  per block.
- **Shape functions.** The dimensions of a slice were read at `outRank + i - 1` (partition) and `i`
  (stitch) instead of after the indices' own dimensions. The partition read a stride or the wrong
  dimension (right only for C-ordered rank-two data with vector indices, or when the data's rank is twice
  the indices' rank minus one); the stitch was right only when its first indices are a vector. The
  partition sizes took one pass over the indices per partition; the stitch took the maximum of
  zero-length index arrays.
- **Empty arrays.** A partition that got no slice has zero-length arrays and `NDArray::isEmpty()` is true
  for them, so the default `EMPTY_SKIP` never ran `dynamic_stitch` (the other partitions were not
  stitched) or `dynamic_partition_bp` (the gradient was never written).
- **Gaps.** A stitched row that no index names was never written: the output holds whatever the recycled
  allocation held.

## Decision

### Pairing

The data has the shape of the indices followed by the dimensions of a slice. Element e of the indices (in
logical, C order) names slice e: the e-th run of `sliceLength = data.length / indices.length` elements of
the data in logical order, whatever the layouts of the operands. Every operand is read and written through
its own shape and strides, from its buffer pointer (which includes a view's offset).

### dynamic_partition

Partition p receives the slices whose index is p, in index order: the output has shape
`[count(p), <slice dimensions>]` and a slice's position is the number of earlier slices of its partition. An
index outside `[0, numPartitions)` names no partition: its slice is dropped. Indices are read as integers
(CPU: any type; CUDA: INT32 and INT64).

### dynamic_partition_bp

Slice e of the data's gradient is the slice at the position of e in the gradient of partition `indices[e]`;
a slice that names no partition (and was therefore dropped by the forward op) has a zero gradient. The op
checks the types, the rank and the slice dimensions of every gradient, and calls the helper, which writes
every element of the output.

### dynamic_stitch

Row r of the output is the slice of the last input (in input order) whose indices name r. Within one input
the CPU lets the later slice win; on CUDA the order of slices naming the same row inside one input is
unspecified (they are written in parallel). A row that no index names is zero on both backends. An index
outside `[0, rows)` is a validation error on the CPU; the CUDA kernels drop such a slice (a host read of the
indices would synchronize a captured stream). The shape function takes the number of rows from the largest
index and skips zero-length index arrays.

### dynamic_stitch_bp

The unreachable helper `dynamicStitchFunctorBP` is removed from both backends and `helpers/dynamic.h`; the
gradient of the stitch stays a gather in `DynamicStitch.doDiff` (covered by a SameDiff test).

### Empty arrays

`dynamic_stitch` and `dynamic_partition_bp` declare `emptyHandling() = EMPTY_EXECUTE`
(`headers/parity_ops.h`: the `DECLARE_CUSTOM_OP` expansion plus the override, as `decoder_masked_mha`
does), and write a zero-length operand's share as nothing.

### CUDA

One block per partition (grid-stride over the partitions) ranks the indices in tiles with warp votes
(`__ballot_sync`, `__popc`), so equal partitions keep their order and every position is deterministic.
One grid-stride kernel then moves whole slices (partition, backprop) or each input's slices in input
order (stitch, one launch per input on the context stream, preceded by a kernel that zeroes the output). The
launch dimensions come from the `dynamic_partition_tad` and `dynamic_stitch_tad` keys (blocks, threads,
no dynamic shared memory); the positions and the operand tables are `PointersManager` temporaries
(capture-aware, freed stream-ordered behind the kernels); no per-element host transfer and no
synchronization.

## Consequences

- `headers/parity_ops.h` changed: every translation unit that includes it recompiles.
- The CUDA and CPU artifacts agree except for repeated indices inside one stitch input (unspecified on CUDA),
  an out-of-range stitch index (an error on the CPU, a dropped slice on CUDA) and the index types the
  helpers accept.
- Tests: `DynamicPartitionStitchTest` (both backends): closed forms for partition, its gradient (also
  through SameDiff) and stitch, sizes 1 to 70001, slices of one to three dimensions, rank-two indices,
  every layout, indices outside the partitions, unused partitions, a partition-stitch round trip and the
  stitch gradient.
