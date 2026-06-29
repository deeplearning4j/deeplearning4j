/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ndarray;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.shape.Gather;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMax;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentMean;
import org.nd4j.linalg.api.ops.impl.transforms.segment.UnsortedSegmentSum;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Sparse-gradient embedding lookup and embedding-bag pooling utilities.
 *
 * <p>All forward and backward computations are expressed purely via existing ND4J
 * differentiable ops (Gather, UnsortedSegment{Sum,Mean,Max}). There is
 * no new C++ op and no host-side element loop over embedding data.
 *
 * <h2>Eager (INDArray) API</h2>
 * <ul>
 *   <li>{@link #lookup} – gather rows of an embedding table by index.</li>
 *   <li>{@link #lookupDenseGrad} – accumulate upstream gradients into a dense
 *       [vocabSize, dim] gradient matrix via UnsortedSegmentSum (handles duplicate
 *       indices correctly). Only used rows receive non-zero gradients; unused rows
 *       remain zero.</li>
 *   <li>{@link #toSparseGradient} – return the lookup gradient as a COO
 *       {@link SparseNDArray} (shape [vocabSize, dim]) so only the touched
 *       N*dim values are materialised—critical for large-vocab training.</li>
 *   <li>{@link #embeddingBag} – pooled embedding lookup (SUM / MEAN / MAX)
 *       using unsorted segment reduce.</li>
 * </ul>
 *
 * <h2>SameDiff (differentiable graph) API</h2>
 * <ul>
 *   <li>{@link #sdLookup} – embedding lookup wired into a SameDiff graph.
 *       Gradient w.r.t. the table is computed by Gather's doDiff
 *       (an UnsortedSegmentSum over the upstream gradient), so duplicate indices
 *       accumulate correctly and only used rows receive non-zero gradient.</li>
 *   <li>{@link #sdEmbeddingBag} – embedding-bag pooling wired into a SameDiff
 *       graph.  Differentiable through Gather + UnsortedSegment{Sum,Mean,Max},
 *       all of which have registered doDiff implementations.</li>
 * </ul>
 *
 * <h2>Pool modes</h2>
 * <ul>
 *   <li>{@code MODE_SUM  = 0} – sum pooling (UnsortedSegmentSum)</li>
 *   <li>{@code MODE_MEAN = 1} – mean pooling (UnsortedSegmentMean)</li>
 *   <li>{@code MODE_MAX  = 2} – max pooling (UnsortedSegmentMax)</li>
 * </ul>
 */
public final class SparseEmbedding {

    /** Embedding-bag pooling mode: sum the gathered rows within each bag. */
    public static final int MODE_SUM  = 0;
    /** Embedding-bag pooling mode: average the gathered rows within each bag. */
    public static final int MODE_MEAN = 1;
    /** Embedding-bag pooling mode: element-wise max of gathered rows within each bag. */
    public static final int MODE_MAX  = 2;

    private SparseEmbedding() { /* utility class */ }

    // -----------------------------------------------------------------------
    // Eager (INDArray) forward ops
    // -----------------------------------------------------------------------

    /**
     * Gather rows of {@code table} at the given {@code indices}.
     *
     * <p>Equivalent to PyTorch {@code F.embedding(indices, table)}.
     * Internally runs the native {@code gather} op (axis 0) so execution is
     * on whichever backend (CPU / CUDA) is currently active.
     *
     * @param table   embedding matrix [vocabSize, dim] — any float dtype
     * @param indices 1D lookup indices [N] — INT32 or INT64
     * @return        gathered rows [N, dim], same dtype as {@code table}
     */
    public static INDArray lookup(INDArray table, INDArray indices) {
        Preconditions.checkNotNull(table,   "table must not be null");
        Preconditions.checkNotNull(indices, "indices must not be null");
        Preconditions.checkArgument(table.rank() == 2,
                "table must be rank-2 [vocabSize, dim], got rank %d", table.rank());
        Preconditions.checkArgument(indices.rank() == 1,
                "indices must be rank-1 [N], got rank %d", indices.rank());
        return Nd4j.exec(new Gather(table, indices, 0))[0];
    }

    /**
     * Compute the dense gradient of an embedding lookup w.r.t. the table.
     *
     * <p>Accumulates upstream gradient rows into a dense [vocabSize, dim] gradient
     * matrix via UnsortedSegmentSum. Rows whose indices appear more than once receive
     * the summed gradient. Rows not present in {@code indices} remain zero.
     *
     * <p>This creates a [vocabSize, dim] allocation. For large vocabularies prefer
     * {@link #toSparseGradient}, which returns a COO tensor containing only the
     * N*dim non-zero elements.
     *
     * @param upstreamGrad upstream gradient [N, dim] — same dtype as table
     * @param indices      1D lookup indices [N] used in the forward pass — INT32 or INT64
     * @param vocabSize    number of rows in the embedding table (V)
     * @return             dense gradient [vocabSize, dim]; only used rows are non-zero
     */
    public static INDArray lookupDenseGrad(INDArray upstreamGrad, INDArray indices, long vocabSize) {
        Preconditions.checkNotNull(upstreamGrad, "upstreamGrad must not be null");
        Preconditions.checkNotNull(indices,      "indices must not be null");
        Preconditions.checkArgument(upstreamGrad.rank() == 2,
                "upstreamGrad must be rank-2 [N, dim], got rank %d", upstreamGrad.rank());
        Preconditions.checkArgument(indices.rank() == 1,
                "indices must be rank-1 [N], got rank %d", indices.rank());
        Preconditions.checkArgument(vocabSize > 0, "vocabSize must be positive, got %d", vocabSize);

        // UnsortedSegmentSum treats indices as segment IDs:
        //   output[idx] = sum of upstreamGrad[i] for all i where indices[i] == idx
        // This correctly accumulates duplicate indices (rows looked up more than once
        // receive the sum of all upstream gradient rows, not the last one). Unused rows
        // are left at zero by the native op. No ref/zeros array needed.
        INDArray idx32 = indices.dataType() == DataType.INT32
                ? indices
                : indices.castTo(DataType.INT32);
        return Nd4j.exec(new UnsortedSegmentSum(upstreamGrad, idx32, (int) vocabSize))[0];
    }

    /**
     * Return the lookup gradient as a COO {@link SparseNDArray}.
     *
     * <p>The logical shape is [vocabSize, dim]. Only the N*dim elements
     * corresponding to the looked-up rows are stored — unused rows occupy no
     * memory. This is the memory-efficient representation for large-vocab
     * sparse weight updates.
     *
     * <p>If the same index appears multiple times, both occurrences are included
     * as separate COO entries. To deduplicate (accumulate repeated index gradients)
     * convert to dense with {@link SparseNDArray#toDense()} or accumulate into the
     * weight table with {@link #lookupDenseGrad}.
     *
     * <p>Implementation detail: the COO (row, col) index arrays are built with
     * {@link Nd4j#tile} and {@link Nd4j#concat}, so no host-side element loop
     * over embedding values occurs.
     *
     * @param upstreamGrad upstream gradient [N, dim]
     * @param indices      1D lookup indices [N] — INT32 or INT64
     * @param vocabSize    number of rows V in the embedding table
     * @return             COO SparseNDArray with shape [vocabSize, dim] and nnz = N*dim
     */
    public static SparseNDArray toSparseGradient(INDArray upstreamGrad, INDArray indices, long vocabSize) {
        Preconditions.checkNotNull(upstreamGrad, "upstreamGrad must not be null");
        Preconditions.checkNotNull(indices,      "indices must not be null");
        Preconditions.checkArgument(upstreamGrad.rank() == 2,
                "upstreamGrad must be rank-2 [N, dim], got rank %d", upstreamGrad.rank());
        Preconditions.checkArgument(indices.rank() == 1,
                "indices must be rank-1 [N], got rank %d", indices.rank());
        Preconditions.checkArgument(vocabSize > 0, "vocabSize must be positive, got %d", vocabSize);

        long N   = indices.length();
        long dim = upstreamGrad.size(1);
        long nnz = N * dim;

        // Row indices for COO: each indices[i] repeated `dim` times.
        // indices.reshape(N, 1) tiled to [N, dim], then flattened to [nnz, 1].
        INDArray rowBlock = Nd4j.tile(
                indices.castTo(DataType.INT64).reshape(N, 1),
                new int[]{1, (int) dim}
        ).reshape(nnz, 1);

        // Column indices: [0, 1, ..., dim-1] tiled N times → [N, dim] → [nnz, 1].
        INDArray colTemplate = Nd4j.tile(
                Nd4j.arange(dim).castTo(DataType.INT64).reshape(1, dim),
                new int[]{(int) N, 1}
        ).reshape(nnz, 1);

        // COO index matrix [nnz, 2] = concat([rowBlock, colTemplate], axis=1)
        INDArray cooIndices = Nd4j.concat(1, rowBlock, colTemplate);

        // Non-zero values: upstreamGrad row-major flatten to [nnz]
        INDArray values = upstreamGrad.reshape(nnz);

        return new SparseNDArray(cooIndices, values, new long[]{vocabSize, dim}, SparseFormat.COO);
    }

    // -----------------------------------------------------------------------
    // Eager (INDArray) embedding-bag pooling
    // -----------------------------------------------------------------------

    /**
     * Pooled embedding lookup (embedding-bag).
     *
     * <p>Semantics match PyTorch {@code nn.EmbeddingBag}:
     * <ol>
     *   <li>Gather rows of {@code table} at {@code indices}  → gathered [nnz, dim]</li>
     *   <li>Reduce each bag using {@code segmentIds}         → output  [numBags, dim]</li>
     * </ol>
     *
     * <p>The segment reduce (step 2) is performed by one of the existing ND4J
     * {@code unsorted_segment_*} ops, so {@code segmentIds} does not need to be
     * sorted and may have gaps.
     *
     * @param table      embedding matrix [vocabSize, dim]
     * @param indices    1D indices [nnz] of rows to look up — INT32 or INT64
     * @param segmentIds 1D bag assignment [nnz] with values in [0, numBags) — INT32 or INT64
     * @param numBags    number of output bags
     * @param mode       {@link #MODE_SUM}, {@link #MODE_MEAN}, or {@link #MODE_MAX}
     * @return           pooled output [numBags, dim], same dtype as {@code table}
     */
    public static INDArray embeddingBag(INDArray table, INDArray indices,
                                        INDArray segmentIds, int numBags, int mode) {
        Preconditions.checkNotNull(table,      "table must not be null");
        Preconditions.checkNotNull(indices,    "indices must not be null");
        Preconditions.checkNotNull(segmentIds, "segmentIds must not be null");
        Preconditions.checkArgument(table.rank() == 2,
                "table must be rank-2 [vocabSize, dim], got rank %d", table.rank());
        Preconditions.checkArgument(indices.rank() == 1,
                "indices must be rank-1 [nnz], got rank %d", indices.rank());
        Preconditions.checkArgument(segmentIds.rank() == 1,
                "segmentIds must be rank-1 [nnz], got rank %d", segmentIds.rank());
        Preconditions.checkArgument(indices.length() == segmentIds.length(),
                "indices and segmentIds must have the same length, got %d vs %d",
                indices.length(), segmentIds.length());
        Preconditions.checkArgument(numBags > 0, "numBags must be positive, got %d", numBags);
        validateMode(mode);

        // Step 1: gather rows [nnz, dim]
        INDArray gathered = lookup(table, indices);

        // Cast segmentIds to INT32 as required by segment ops
        INDArray segIds32 = segmentIds.dataType() == DataType.INT32
                ? segmentIds
                : segmentIds.castTo(DataType.INT32);

        // Step 2: segment reduce over gathered rows
        switch (mode) {
            case MODE_SUM:
                return Nd4j.exec(new UnsortedSegmentSum(gathered, segIds32, numBags))[0];
            case MODE_MEAN:
                return Nd4j.exec(new UnsortedSegmentMean(gathered, segIds32, numBags))[0];
            case MODE_MAX:
                return Nd4j.exec(new UnsortedSegmentMax(gathered, segIds32, numBags))[0];
            default:
                throw new IllegalArgumentException("mode must be MODE_SUM(0), MODE_MEAN(1), or MODE_MAX(2), got " + mode);
        }
    }

    // -----------------------------------------------------------------------
    // SameDiff (differentiable graph) API
    // -----------------------------------------------------------------------

    /**
     * Wire an embedding lookup into a {@link SameDiff} graph.
     *
     * <p>The returned variable has shape [N, dim].  Because this delegates to
     * {@code sd.gather(table, indices, 0)}, the existing {@link Gather#doDiff}
     * implementation provides the gradient w.r.t. {@code table} via
     * {@code UnsortedSegmentSum}: duplicate indices accumulate correctly so rows
     * looked up multiple times receive the summed upstream gradient.  Rows not
     * present in {@code indices} receive zero gradient.
     *
     * @param sd      the SameDiff graph
     * @param table   SDVariable [vocabSize, dim] — must be a {@code sd.var}
     * @param indices SDVariable [N] — integer type, treated as a constant
     *                (zerosLike gradient returned for it)
     * @return        SDVariable [N, dim] — differentiable w.r.t. {@code table}
     */
    public static SDVariable sdLookup(SameDiff sd, SDVariable table, SDVariable indices) {
        Preconditions.checkNotNull(sd,      "sd must not be null");
        Preconditions.checkNotNull(table,   "table must not be null");
        Preconditions.checkNotNull(indices, "indices must not be null");
        // Gather axis=0: select rows of the embedding table
        return sd.gather(table, indices, 0);
    }

    /**
     * Wire a differentiable embedding-bag into a {@link SameDiff} graph.
     *
     * <p>The graph is: {@code gathered = gather(table, indices, 0)} followed by
     * {@code unsorted_segment_{sum|mean|max}(gathered, segmentIds, numBags)}.
     * Both ops have registered {@code doDiff} implementations, so the full
     * gradient chain back to {@code table} is available without any custom op.
     *
     * @param sd         the SameDiff graph
     * @param table      SDVariable [vocabSize, dim]
     * @param indices    SDVariable [nnz] — integer type
     * @param segmentIds SDVariable [nnz] — integer type, bag assignments in [0, numBags)
     * @param numBags    number of output bags
     * @param mode       {@link #MODE_SUM}, {@link #MODE_MEAN}, or {@link #MODE_MAX}
     * @return           SDVariable [numBags, dim]
     */
    public static SDVariable sdEmbeddingBag(SameDiff sd, SDVariable table,
                                             SDVariable indices, SDVariable segmentIds,
                                             int numBags, int mode) {
        Preconditions.checkNotNull(sd,         "sd must not be null");
        Preconditions.checkNotNull(table,      "table must not be null");
        Preconditions.checkNotNull(indices,    "indices must not be null");
        Preconditions.checkNotNull(segmentIds, "segmentIds must not be null");
        Preconditions.checkArgument(numBags > 0, "numBags must be positive, got %d", numBags);
        validateMode(mode);

        // Step 1: gather rows [nnz, dim] — differentiable via Gather.doDiff → UnsortedSegmentSum
        SDVariable gathered = sd.gather(table, indices, 0);

        // Step 2: segment reduce — each has its own Bp op for doDiff
        switch (mode) {
            case MODE_SUM:
                return new UnsortedSegmentSum(sd, gathered, segmentIds, numBags).outputVariable();
            case MODE_MEAN:
                return new UnsortedSegmentMean(sd, gathered, segmentIds, numBags).outputVariable();
            case MODE_MAX:
                return new UnsortedSegmentMax(sd, gathered, segmentIds, numBags).outputVariable();
            default:
                throw new IllegalArgumentException("mode must be MODE_SUM(0), MODE_MEAN(1), or MODE_MAX(2), got " + mode);
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    private static void validateMode(int mode) {
        if (mode < MODE_SUM || mode > MODE_MAX) {
            throw new IllegalArgumentException(
                    "mode must be MODE_SUM(0), MODE_MEAN(1), or MODE_MAX(2), got " + mode);
        }
    }
}
