# ADR 0134: Image resize and crop_and_resize kernels

## Status

Accepted (2026-10-04). Renumber on merge if 0134 is taken (0131 to 0133 are ADRs of the same batch).

## Context

The image resizes of `helpers/cuda/image_resize.cu` and `helpers/cpu/image_resize.cpp` (`resize_nearest_neighbor`,
`resize_bilinear`, `resize_bicubic`, `resize_area`, `resize_images`, `image_resize`) and `crop_and_resize`
worked only for dense C-ordered images of a few sizes:

- **Nearest launch.** The kernel handled one image per block with the rows on `threadIdx.x`, and
  `resizeNeighborDims` returned `(outH * outW, batch)` for a `<<<x, y>>>` launch: images from index
  `outH * outW` on were never written (silently), a batch of 1025 or more was an invalid launch, and the
  intended launch (a block per image, a thread per pixel) is invalid from 33 x 33 outputs on.
- **Views and layouts.** The nearest, bilinear and bicubic kernels read and wrote `getDataBuffer()`
  pointers (a view's offset dropped) and indexed dense C order; the area resize wrote its output as a dense C
  array although the shape function keeps the input's order.
- **Bicubic.** The advance values were computed by one thread after a `__syncthreads()` that does not order
  other blocks (wrong from 129 output columns); the main kernel needed one thread per block, took 512 bytes
  of dynamic shared memory for `16 * channels` bytes (an overflow from 33 channels) and its 3-channel branch
  was selected for 100 (a typo). Both backends wrote floats through the output's buffer whatever its type
  (the op allows DOUBLE), and `resizeBicubicFunctor`, which `resize_images` calls, was an empty stub.
- **Area.** The CUDA row cache was `[batch, outH, outW]` while every output row needs
  `ceil(heightScale) + 1` entries: a row longer than the output width overran the pool; the launches were
  literal `<<<128, 128, 2048>>>` and `<<<128, 128, 256>>>` (the named keys were ignored).
- **Capture.** Temporaries were released with `CudaMemoryPool::free` after a synchronization that is skipped
  during a capture, while the recorded kernels still referenced them.
- **crop_and_resize.** The box indices were read linearly; an index outside the batch read out of bounds (below
  zero) or left the crop unwritten; the sample positions were float for DOUBLE images and the horizontal
  weight of an integer image was computed as an integer (0).

## Decision

### Layout

Every kernel and loop reads the input and writes the output through their strides, from the arrays' own
buffer pointers (the offset of a view included): C, F, permuted and stepped layouts give the same results.
The area resize accumulates into a dense array and assigns it to an output of another layout. A 3D image is
a batch of one; the size arguments of `resize_nearest_neighbor` are `(height, width)`, as its size input.
Bicubic column/channel offsets remain `LongType` through the shared interpolation accessor (no 32-bit narrowing).

### Launches

One thread per output pixel, striding over `batch * outH * outW`, the channels in a loop (nearest,
bilinear, bicubic); one thread per output row of an image (area); a block per box with its threads striding
over the crop's pixels (crop). The dimensions come from `resizeNeighborDims` (blocks = min(ceil(pixels / 256),
65535), 256 threads, no dynamic shared memory) and the named keys `image_resize`,
`image_resize_interp_weights` and `image_resize_fill_interp` (x = blocks, y = threads). Environment overrides
of `resizeNeighborDims` follow the keys' names: `GRID_SIZE_IMAGE_RESIZE_NEIGHBOR` is the number of blocks and
`BLOCK_SIZE_IMAGE_RESIZE_NEIGHBOR` the threads (the old producer had them the other way round).

### Arithmetic

- Nearest copies; the source index of an output index is the scaler's value rounded by the nearest mode
  (floor, round prefer floor, round prefer ceil, ceil) and clamped to the image.
- Bilinear interpolates in double on both backends and converts once into the output storage type. Each scalar
  subtraction, multiplication and addition rounds separately in DOUBLE (no implicit multiply-add contraction).
- Bicubic interpolates in float, rounding each product and each left-to-right sum separately, with the
  host-computed coefficient table and the Keys weights of
  `getWeightsAndIndices`, and stores the result as the output's type, FLOAT32 or DOUBLE (another output type
  and an unknown coordinate mode are `BAD_INPUT`). `resizeBicubicFunctor` (`resize_images`) is the legacy
  resize: asymmetric coordinates, the borders repeated, the ordinary coefficient -0.75.
- CPU bicubic coefficient-table workers own disjoint half-open index ranges over all `kTableSize + 1` entries,
  including the final endpoint; adjacent workers no longer concurrently write the same entry.
- Area keeps a cache of `ceil(heightScale) + 2` input rows per output row (the margin covers the rounding of
  the float coordinates).
- `crop_and_resize` computes positions and interpolation in double when the images or the boxes are DOUBLE and
  in float otherwise. Every scalar operation rounds separately in that computation type, including coordinate
  scaling and interpolation, using the existing `reproducible_math.h` primitives shared by CPU and CUDA. Output
  storage retains the image dtype, as does Java output inference. A box whose index names no image of the batch,
  and every position outside the image (including NaN/nonfinite coordinates), gives the extrapolation value on
  both backends, before any floor/ceil/index conversion.
- Boxes must be floating-point; indices and crop sizes must be integers. Image rank 4, boxes shape `[N,4]`,
  sufficient box indices, positive image height/width, positive crop height/width, and the optional method
  (`0` bilinear or `1` nearest) are validated before shape metadata is indexed or outputs are allocated. Unknown
  methods are rejected, not interpreted as nearest. TensorFlow Java import also rejects unknown method names.

### Temporaries

CUDA temporaries (weights, tables, the cache pool, the resize state) are `PointersManager` allocations
(capture workspace while recording, freed stream-ordered behind the kernels otherwise); launch errors are
checked unless a graph capture is active; no launch synchronizes. The one barrier is before the copy of a
staged area result into an output that is not a dense array: an NDArray op on the output's own context, which
need not be the helper's stream (skipped during a capture).

## Consequences

- `image_resize_v2.cu` (the antialiasing spans path) is unchanged.
- A script that set the `GRID_SIZE_` or `BLOCK_SIZE_IMAGE_RESIZE_NEIGHBOR` overrides gets the other
  dimension.
- Tests: `ImageResizeParityTest` (both backends): the goldens of the native tests, batches up to 2049 against
  outputs of one to eight pixels, every coordinate and nearest mode, every layout of input and output, all
  types, 3D images, 1025 images, 300 output columns, 100 channels, heavy shrinks, a DOUBLE bicubic output,
  1500 boxes and 1030 images for the crop. Exact cancellation discriminators separate ordered arithmetic from
  contracted arithmetic: crop `[-1024,.125]` at `8191/8192` must produce zero in FLOAT rather than `-2^-16`;
  bicubic `[0,-(1+2^-23),1+2^-23,0]` at source `1.5` must produce zero rather than `3*2^-28`. Mixed FLOAT/DOUBLE
  crop cases additionally verify the chosen computation type. Invalid-rank/dtype/method/size and NaN/Inf tests
  cover shape inference and execution boundaries. These additions require the parent-owned CPU/CUDA build and
  test gate; source review alone does not establish the original randomized CUDA mismatches' exact operands.
