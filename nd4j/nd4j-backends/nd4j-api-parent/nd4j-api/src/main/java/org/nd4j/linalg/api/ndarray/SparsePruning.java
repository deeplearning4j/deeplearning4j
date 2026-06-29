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

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.sparse.DenseToCsr;
import org.nd4j.linalg.api.ops.impl.sparse.Sddmm;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Utility class for magnitude-based weight pruning and a structured-sparse linear layer.
 *
 * <h3>Design constraints</h3>
 * <ul>
 *   <li>All compute runs on the active ND4J backend via existing ops — no host element
 *       loops over weight data, no manual syncToHost/syncToDevice, no silent fallback.</li>
 *   <li>Threshold computation uses the {@code top_k} op (via {@code DynamicCustomOp.builder}
 *       so that B_ARG(0)=sorted and INT_ARG(0)=k match the native op contract);
 *       sparsification uses {@code dense_to_csr} with the scalar threshold T_ARG —
 *       both ops run natively on whatever backend (CPU or CUDA) is active.</li>
 *   <li>Only one scalar is read from the backend to set the threshold
 *       ({@code topKValues.getDouble(k-1)}) — this is unavoidable and is not
 *       a "host loop over weight data."</li>
 * </ul>
 *
 * <h3>Threshold semantics</h3>
 * {@code dense_to_csr} keeps entries where {@code |x| > threshold} (strict inequality).
 * We set {@code threshold = nextDown(minKept)} computed in the weight's native precision
 * (float {@code nextDown} for FLOAT/HALF inputs, double for DOUBLE) so the k-th largest
 * value passes even after T_ARG is cast to the weight's dtype by the native op.
 * In the presence of exact ties at the boundary the actual nnz may differ slightly from
 * the requested count, which is the standard behaviour of magnitude pruning.
 */
public final class SparsePruning {

    private SparsePruning() {}

    // -----------------------------------------------------------------------
    // 1. magnitudePrune
    // -----------------------------------------------------------------------

    /**
     * Magnitude prune a dense weight matrix to a CSR {@link SparseNDArray}.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Compute k = round((1 − sparsityFraction) × m × n) — number of entries to keep.</li>
     *   <li>Run {@code top_k} on the flattened absolute values to find the k largest
     *       magnitudes on the active backend (no host loop).</li>
     *   <li>Set threshold = one ULP below the minimum of those k magnitudes.</li>
     *   <li>Run {@code dense_to_csr(weight, threshold)} which keeps {@code |w| > threshold}.</li>
     * </ol>
     *
     * @param weight           2D weight matrix [m, n]
     * @param sparsityFraction fraction of weights to zero out, in [0, 1)
     * @return CSR {@link SparseNDArray} with the top-(1−sparsity) fraction of entries
     */
    public static SparseNDArray magnitudePrune(INDArray weight, double sparsityFraction) {
        Preconditions.checkArgument(weight.rank() == 2,
                "weight must be 2D, got rank %d", weight.rank());
        Preconditions.checkArgument(sparsityFraction >= 0.0 && sparsityFraction < 1.0,
                "sparsityFraction must be in [0, 1), got %f", sparsityFraction);

        long m = weight.shape()[0];
        long n = weight.shape()[1];
        long total = m * n;
        long k = Math.max(1L, Math.round((1.0 - sparsityFraction) * total));

        double threshold;
        if (k >= total) {
            // Keep everything: threshold = 0.0 (dense_to_csr keeps all non-zeros)
            threshold = 0.0;
        } else {
            // Flatten and compute absolute values — no host loop, all on active backend
            INDArray mag = Transforms.abs(weight.reshape(total), true);

            // top_k: B_ARG(0)=sorted, INT_ARG(0)=k (native op contract).
            // The TopK(INDArray, double, boolean) constructor puts sorted_as_int at
            // INT_ARG(0) and k at INT_ARG(1), swapping them — use the builder instead.
            INDArray[] topKOut = Nd4j.exec(DynamicCustomOp.builder("top_k")
                    .addInputs(mag)
                    .addBooleanArguments(true)          // B_ARG(0) = sorted = true
                    .addIntegerArguments((int) k)       // INT_ARG(0) = k
                    .build());
            INDArray topKValues = topKOut[0]; // [k], descending

            // Minimum of the kept values = topKValues[k-1].
            // dense_to_csr converts T_ARG to the weight's native dtype before comparing.
            // Math.nextDown(double) is too fine a step for FLOAT: nextDown(16.0d) rounds
            // back to 16.0f, so |16.0f| > 16.0f = false and the entry is dropped.
            // Use nextDown in the native precision so the threshold is truly < minKept
            // when cast to FLOAT (or HALF).
            double minKept = topKValues.getDouble(k - 1);
            if (minKept > 0.0) {
                DataType dt = weight.dataType();
                if (dt == DataType.FLOAT || dt == DataType.HALF || dt == DataType.BFLOAT16) {
                    threshold = (double) Math.nextDown((float) minKept);
                } else {
                    threshold = Math.nextDown(minKept);
                }
            } else {
                threshold = 0.0;
            }
        }

        // dense_to_csr keeps |w| > threshold — runs on the active backend
        INDArray[] csrOut = Nd4j.exec(new DenseToCsr(weight, threshold));
        // csrOut[0] = values[nnz], csrOut[1] = colIdx[nnz], csrOut[2] = rowPtr[m+1]
        return new SparseNDArray(csrOut[0], csrOut[1], csrOut[2],
                new long[]{m, n}, SparseFormat.CSR);
    }

    // -----------------------------------------------------------------------
    // 2. SparseLinear
    // -----------------------------------------------------------------------

    /**
     * A pruned linear layer holding a sparse CSR weight matrix.
     *
     * <h3>Forward pass</h3>
     * {@code C = W · x} computed via {@code csr_spmm} — the existing differentiable
     * SpMM op.  No dense materialisation of W is needed.
     *
     * <h3>Structured weight gradient</h3>
     * {@code dL/dW} sampled only at W's kept (non-zero) positions via {@code sddmm}:
     * <pre>
     *   dW_values[k] = sum_b  gradOut[i(k), b] · x[j(k), b]
     * </pre>
     * where (i(k), j(k)) is the (row, col) of the k-th non-zero in W.
     * Pruned-away positions receive no gradient update — exactly the behaviour
     * needed for sparse-aware training of pruned networks.
     */
    public static final class SparseLinear {

        private final SparseNDArray W;

        /**
         * Construct a SparseLinear layer from a pre-built CSR weight matrix.
         *
         * @param W CSR sparse weight [outFeatures, inFeatures]
         * @throws IllegalArgumentException if W is not in CSR format
         */
        public SparseLinear(SparseNDArray W) {
            Preconditions.checkArgument(W.getFormat() == SparseFormat.CSR,
                    "SparseLinear requires a CSR weight matrix; got %s. Call magnitudePrune() or toCsr().",
                    W.getFormat());
            this.W = W;
        }

        /** Returns the number of output features (rows of W). */
        public long outFeatures() { return W.rows(); }

        /** Returns the number of input features (columns of W). */
        public long inFeatures() { return W.cols(); }

        /** Returns the underlying sparse weight matrix. */
        public SparseNDArray weights() { return W; }

        /**
         * Forward pass: {@code C = W · x}.
         *
         * <p>Executes {@code csr_spmm} on the active backend.
         *
         * @param x input matrix [inFeatures, batchSize]
         * @return output matrix [outFeatures, batchSize]
         */
        public INDArray forward(INDArray x) {
            Preconditions.checkArgument(x.rank() == 2,
                    "x must be 2D [inFeatures, batchSize], got rank %d", x.rank());
            Preconditions.checkArgument(x.shape()[0] == W.cols(),
                    "x.shape[0]=%d must equal inFeatures=%d", x.shape()[0], W.cols());
            // SparseNDArray.mmul delegates to CsrSpmm: W[out,in] · x[in,batch] = C[out,batch]
            return W.mmul(x);
        }

        /**
         * Structured weight-gradient: {@code dL/dW} sampled at W's sparsity pattern.
         *
         * <p>Uses {@code sddmm(rowPtr, colIdx, gradOut, x, outFeatures, inFeatures)} to
         * compute the outer-product gradient only at the kept (non-zero) positions.
         * Pruned positions receive zero gradient, which is what sparse-aware SGD requires.
         *
         * @param gradOut upstream gradient {@code dL/dC}, shape [outFeatures, batchSize]
         * @param x       the same input that was passed to {@link #forward}, shape [inFeatures, batchSize]
         * @return 1D INDArray [nnz] — gradient w.r.t. each kept weight value, in CSR order
         */
        public INDArray weightGradient(INDArray gradOut, INDArray x) {
            Preconditions.checkArgument(gradOut.rank() == 2,
                    "gradOut must be 2D [outFeatures, batchSize], got rank %d", gradOut.rank());
            Preconditions.checkArgument(gradOut.shape()[0] == W.rows(),
                    "gradOut.shape[0]=%d must equal outFeatures=%d", gradOut.shape()[0], W.rows());
            Preconditions.checkArgument(x.shape()[0] == W.cols(),
                    "x.shape[0]=%d must equal inFeatures=%d", x.shape()[0], W.cols());

            // sddmm(rowPtr, colIdx, D1, D2, rows, cols)
            // D1 = gradOut [out, batch], D2 = x [in, batch]
            // out[k] = sum_b D1[i(k), b] * D2[j(k), b]  =  sum_b gradOut[i,b] * x[j,b]  =  dL/dW_{ij}
            Sddmm op = new Sddmm(W.getRowPtr(), W.getColIdx(), gradOut, x, W.rows(), W.cols());
            return Nd4j.exec(op)[0];
        }
    }

    // -----------------------------------------------------------------------
    // 3. structuredPrune (neuron/channel pruning by row or column L2 norm)
    // -----------------------------------------------------------------------

    /**
     * Structured pruning: zero out entire rows or columns of a weight matrix based on
     * their L2 norm, then return the result as a CSR {@link SparseNDArray}.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Compute L2 norm per row (axis=0) or per column (axis=1).</li>
     *   <li>Find the k-th largest norm via {@code top_k} (k = round((1−fraction) × count)).</li>
     *   <li>Build a binary keep mask (1 for kept rows/cols, 0 for pruned).</li>
     *   <li>Zero out pruned rows/cols by elementwise multiply with the broadcast mask.</li>
     *   <li>Convert the result to CSR with threshold=0.0.</li>
     * </ol>
     *
     * <p>No host element loop over weight data is performed.
     *
     * @param weight   2D weight matrix [m, n]
     * @param fraction fraction of rows (axis=0) or columns (axis=1) to remove, in [0, 1)
     * @param axis     0 = prune whole rows (output neurons), 1 = prune whole columns (input features)
     * @return CSR {@link SparseNDArray} with pruned rows/columns zeroed out and compacted
     */
    public static SparseNDArray structuredPrune(INDArray weight, double fraction, int axis) {
        Preconditions.checkArgument(weight.rank() == 2,
                "weight must be 2D, got rank %d", weight.rank());
        Preconditions.checkArgument(fraction >= 0.0 && fraction < 1.0,
                "fraction must be in [0, 1), got %f", fraction);
        Preconditions.checkArgument(axis == 0 || axis == 1,
                "axis must be 0 (prune rows) or 1 (prune cols), got %d", axis);

        long m = weight.shape()[0];
        long n = weight.shape()[1];
        long numVectors = (axis == 0) ? m : n;
        long k = Math.max(1L, Math.round((1.0 - fraction) * numVectors));

        // norm2 along the non-pruning axis:
        //   axis=0 (prune rows)  → norm per row  → norm2(1) → shape [m]
        //   axis=1 (prune cols)  → norm per col  → norm2(0) → shape [n]
        int normAxis = (axis == 0) ? 1 : 0;
        INDArray norms = weight.norm2(normAxis);  // shape [m] or [n]

        double maskThreshold;
        if (k >= numVectors) {
            maskThreshold = 0.0;   // keep all rows/cols
        } else {
            // Same builder pattern as magnitudePrune: B_ARG(0)=sorted, INT_ARG(0)=k
            INDArray[] topKOut = Nd4j.exec(DynamicCustomOp.builder("top_k")
                    .addInputs(norms)
                    .addBooleanArguments(true)          // B_ARG(0) = sorted = true
                    .addIntegerArguments((int) k)       // INT_ARG(0) = k
                    .build());
            double minKept = topKOut[0].getDouble(k - 1);
            // norms.gt(maskThreshold) compares in the norms' native dtype; use
            // native-precision nextDown so the threshold is truly < minKept at FLOAT.
            if (minKept > 0.0) {
                DataType dt = norms.dataType();
                if (dt == DataType.FLOAT || dt == DataType.HALF || dt == DataType.BFLOAT16) {
                    maskThreshold = (double) Math.nextDown((float) minKept);
                } else {
                    maskThreshold = Math.nextDown(minKept);
                }
            } else {
                maskThreshold = 0.0;
            }
        }

        // keepMask[i] = 1.0 where norms[i] > maskThreshold, else 0.0
        INDArray keepMask = norms.gt(maskThreshold).castTo(weight.dataType());  // shape [m] or [n]

        // Zero out pruned rows/cols by broadcasting the mask
        INDArray pruned;
        if (axis == 0) {
            // keepMask [m]: scale row i by keepMask[i]  →  mulColumnVector broadcasts along cols
            pruned = weight.mulColumnVector(keepMask);   // [m,n] × col[m] → [m,n]
        } else {
            // keepMask [n]: scale col j by keepMask[j]  →  mulRowVector broadcasts along rows
            pruned = weight.mulRowVector(keepMask);      // [m,n] × row[n] → [m,n]
        }

        // Compactify: threshold=0.0 keeps all remaining non-zeros
        INDArray[] csrOut = Nd4j.exec(new DenseToCsr(pruned, 0.0));
        return new SparseNDArray(csrOut[0], csrOut[1], csrOut[2],
                new long[]{m, n}, SparseFormat.CSR);
    }
}
