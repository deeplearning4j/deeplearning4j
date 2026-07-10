/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.peft;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.LoftQConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Svd;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * LoftQ (LoRA-Fine-Tuning-aware Quantization) initialization.
 * <p>
 * Implements the alternating optimization algorithm from arXiv:2310.08659.
 * Instead of initializing LoRA matrices randomly or with Kaiming uniform,
 * LoftQ derives A and B from the SVD of the quantization residual so that
 * the initial approximation satisfies:
 * <pre>
 *     W ≈ Q + B @ A
 * </pre>
 * where Q is the quantized weight and B @ A is the low-rank correction.
 * <p>
 * Algorithm for one iteration:
 * <ol>
 *   <li>Quantize W → Q (block-wise NF4 or FP4)</li>
 *   <li>Residual R = W - Q</li>
 *   <li>Economy SVD: R = U @ diag(S) @ Vt  (keeping only rank-r components)</li>
 *   <li>B = U[:, :r] @ diag(sqrt(S[:r])),  A = diag(sqrt(S[:r])) @ Vt[:r, :]</li>
 * </ol>
 * For numIterations &gt; 1, the residual is recomputed as R = W − Q(W − B@A) each round.
 *
 * @author Adam Gibson
 * @see LoftQConfig
 * @see <a href="https://arxiv.org/abs/2310.08659">LoftQ Paper</a>
 */
@Slf4j
public class LoftQInitializer {

    private LoftQInitializer() {
        // Utility class — no instantiation
    }

    /**
     * Initialize LoRA A and B matrices using the LoftQ algorithm.
     *
     * @param weight pretrained weight matrix [outDim, inDim] in FLOAT32 (or FLOAT64)
     * @param rank   LoRA rank r; must satisfy r &lt;= min(outDim, inDim)
     * @param config LoftQ configuration (quantType, quantBits, blockSize, numIterations)
     * @return Pair of (B [outDim, rank], A [rank, inDim])
     */
    public static LoftQResult initializeFull(INDArray weight, int rank, LoftQConfig config) {
        Preconditions.checkArgument(weight != null, "weight must not be null");
        // P1-F: guard against feeding raw GGML-packed INT8 bytes to SVD.
        // A packed weight looks rank-1 and has INT8 dtype — both are invalid for LoftQ.
        if (weight.dataType() == DataType.INT8 || weight.dataType() == DataType.UINT8) {
            throw new IllegalArgumentException(
                "LoftQInitializer.initialize() received a packed/quantized INT8 weight. " +
                "GGUF-packed weights must be dequantized to FP32 before LoftQ initialisation. " +
                "Use GgmlQMatMul's dequantize helper or skip LoftQ for quantized bases.");
        }
        Preconditions.checkArgument(weight.rank() == 2, "weight must be a 2D matrix, got rank %s", weight.rank());
        Preconditions.checkArgument(rank > 0, "rank must be > 0, got %s", rank);

        int outDim = (int) weight.size(0);
        int inDim  = (int) weight.size(1);
        int diagSize = Math.min(outDim, inDim);
        Preconditions.checkArgument(rank <= diagSize,
            "rank (%s) must be <= min(outDim=%s, inDim=%s) = %s", rank, outDim, inDim, diagSize);

        // Work in FLOAT32; promote if needed but avoid modifying caller's array
        INDArray W = weight.dataType() == DataType.FLOAT ? weight.dup() : weight.castTo(DataType.FLOAT);

        // B and A will be updated each iteration
        INDArray B = Nd4j.zeros(DataType.FLOAT, outDim, rank);
        INDArray A = Nd4j.zeros(DataType.FLOAT, rank, inDim);
        // Q_n: the quantized weight from the LAST iteration, retained as the LoftQ base.
        INDArray quantizedBase = null;

        int numIterations = config.getNumIterations();
        String quantType  = config.getQuantType();
        int quantBits     = config.getQuantBits();
        int blockSize     = config.getBlockSize();

        for (int iter = 0; iter < numIterations; iter++) {
            // LoftQ Algorithm (arXiv:2310.08659 Algorithm 1):
            //   Each iteration updates Q and (B, A) jointly:
            //     1. Q_n = quantize(W - B@A)          ← update quantized weights based on current residual
            //     2. R_n = W - Q_n                    ← full residual of W vs quantized approximation
            //     3. B, A = top-rank SVD(R_n)          ← low-rank correction for quantization error
            //   Reconstruction: W ≈ Q_n + B@A
            //   With more iterations, Q_n adapts to the current B@A, giving better joint approximation.

            // Step 1: update Q = quantize(W - B@A)
            INDArray ba;
            if (iter == 0) {
                // First iteration: B=0, A=0, so W - B@A = W
                ba = null;
            } else {
                INDArray bDup = B.dup();
                INDArray aDup = A.dup();
                ba = bDup.mmul(aDup);  // [outDim, inDim]
                bDup.close();
                aDup.close();
            }
            INDArray target = (ba == null) ? W.dup() : W.sub(ba);
            if (ba != null) ba.close();

            INDArray Q;
            if ("nf4".equals(quantType) || quantBits == 4) {
                Q = quantizeNF4(target, blockSize);
            } else {
                Q = quantizeFP4(target, blockSize);
            }
            target.close();

            // Step 2: residual R = W - Q (full residual of original W vs quantized approximation)
            INDArray R = W.sub(Q);
            // Retain the latest quantized weight as the LoftQ base Q_n and release the prior
            // one. The adapters (B, A) below are fit to (W - Q_n), so a correct reconstruction
            // is Q_n + B@A ≈ W: the caller MUST freeze Q_n (not quantize(W)) as the base, or
            // multi-iteration LoftQ ends up worse than a single iteration.
            if (quantizedBase != null) quantizedBase.close();
            quantizedBase = Q;

            // Step 3: Economy SVD of R (only rank-r singular values needed)
            //   Returns: [S, U, Vt] when calcUV=true, fullUV=false
            INDArray[] svdResult = svd(R, rank);
            INDArray S  = svdResult[0];  // [min(outDim,inDim)] singular values
            INDArray U  = svdResult[1];  // [outDim, diagSize]  left singular vectors
            INDArray Vt = svdResult[2];  // [diagSize, inDim]   right singular vectors (transposed)
            R.close();

            // Step 4: build B and A from top-r components
            //   sqrtS_r[i] = sqrt(S[i]) for i in [0, rank)
            //   S has shape [diagSize]; take first `rank` elements via interval
            INDArray S_r   = S.get(NDArrayIndex.interval(0, rank));  // view [rank]
            INDArray sqrtS = Transforms.sqrt(S_r.dup(), false);        // [rank]
            S.close();

            // U_r = U[:, :r]  → [outDim, rank]  (dup to own the memory)
            INDArray U_r  = U.get(NDArrayIndex.all(), NDArrayIndex.interval(0, rank)).dup();
            U.close();

            // Vt_r = Vt[:r, :] → [rank, inDim]
            INDArray Vt_r = Vt.get(NDArrayIndex.interval(0, rank), NDArrayIndex.all()).dup();
            Vt.close();

            // B = U_r * sqrtS  (scale each column i by sqrtS[i])
            //   broadcast: U_r [outDim, rank] * sqrtS [rank] → mulRowVector broadcasts along cols
            //   dup() ensures B is a standalone array not linked to any InferenceSession lifecycle
            INDArray BNew = U_r.mulRowVector(sqrtS).dup();
            U_r.close();

            // A = sqrtS * Vt_r (scale each row i by sqrtS[i])
            //   broadcast: sqrtS [rank] reshaped to [rank,1] then mulColumnVector broadcasts along rows
            //   dup() ensures A is a standalone array not linked to any InferenceSession lifecycle
            INDArray ANew = Vt_r.mulColumnVector(sqrtS.reshape(rank, 1)).dup();
            Vt_r.close();
            sqrtS.close();

            // Close previous B and A (they are no longer needed after this iteration)
            if (iter > 0) {
                B.close();
                A.close();
            }
            B = BNew;
            A = ANew;

            log.debug("LoftQ iter {}/{}: B={}, A={}", iter + 1, numIterations, B.shapeInfoToString(), A.shapeInfoToString());
        }

        return new LoftQResult(quantizedBase, B, A);
    }

    /**
     * Legacy entry point returning only the LoRA adapters (B, A). Prefer
     * {@link #initializeFull} which also returns the iterated quantized base Q_n —
     * required for a correct reconstruction W ≈ Q_n + B@A when numIterations &gt; 1.
     */
    public static Pair<INDArray, INDArray> initialize(INDArray weight, int rank, LoftQConfig config) {
        LoftQResult r = initializeFull(weight, rank, config);
        if (r.quantizedBase != null) r.quantizedBase.close();
        return Pair.of(r.loraB, r.loraA);
    }

    /**
     * Result of LoftQ initialization: the iterated quantized base weight {@code Q_n} plus
     * the low-rank adapters. Freeze {@code quantizedBase} as the base weight so that
     * {@code quantizedBase + loraB @ loraA ≈ W} holds for any number of iterations.
     */
    public static final class LoftQResult {
        public final INDArray quantizedBase; // Q_n [outDim, inDim]
        public final INDArray loraB;         // [outDim, rank]
        public final INDArray loraA;         // [rank, inDim]
        public LoftQResult(INDArray quantizedBase, INDArray loraB, INDArray loraA) {
            this.quantizedBase = quantizedBase;
            this.loraB = loraB;
            this.loraA = loraA;
        }
    }

    /**
     * Quantize a weight matrix using block-wise NF4 (Normal Float 4-bit) quantization.
     * <p>
     * NF4 uses a lookup table derived from the normal distribution quantile function,
     * making it optimal for normally-distributed weights. Each block of
     * {@code blockSize} elements is independently scaled to the range [-1, 1] before
     * mapping to the nearest NF4 code, then dequantized back.
     *
     * @param weight    Weight matrix to quantize [rows, cols]
     * @param blockSize Number of elements per quantization block
     * @return Dequantized approximation of weight
     */
    public static INDArray quantizeNF4(INDArray weight, int blockSize) {
        // NF4 lookup table: 16 values derived from normal distribution quantiles
        // These are the standard NF4 codebook values from the QLoRA paper
        final float[] NF4_CODEBOOK = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
             0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
             0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
        };

        // Flatten to 1D for block processing
        long totalElements = weight.length();
        INDArray flat = weight.reshape(totalElements);
        INDArray result = Nd4j.zeros(DataType.FLOAT, totalElements);

        int numBlocks = (int) Math.ceil((double) totalElements / blockSize);
        for (int b = 0; b < numBlocks; b++) {
            int start = b * blockSize;
            int end   = (int) Math.min(start + blockSize, totalElements);

            INDArray block = flat.get(NDArrayIndex.interval(start, end)).dup();

            // Scale: find absmax in block
            float absMax = (float) block.amaxNumber().doubleValue();
            if (absMax == 0.0f) {
                result.put(new INDArrayIndex[]{NDArrayIndex.interval(start, end)},
                    Nd4j.zeros(DataType.FLOAT, end - start));
                block.close();
                continue;
            }

            // Normalize to [-1, 1]
            float scale = absMax;
            INDArray normalized = block.div(scale);
            block.close();

            // Quantize: find nearest NF4 code for each element
            float[] normData = normalized.toFloatVector();
            normalized.close();
            float[] quantData = new float[normData.length];
            for (int i = 0; i < normData.length; i++) {
                quantData[i] = nearestNF4(normData[i], NF4_CODEBOOK) * scale;
            }

            INDArray quantBlock = Nd4j.createFromArray(quantData);
            result.put(new INDArrayIndex[]{NDArrayIndex.interval(start, end)}, quantBlock);
            quantBlock.close();
        }
        // NOTE: flat is a VIEW sharing the same DataBuffer as weight — do NOT close it here.
        // Closing flat would mark weight's DataBuffer as closed, invalidating the caller's array.

        return result.reshape(weight.shape());
    }

    /**
     * Find the nearest NF4 code for a value in [-1, 1].
     */
    private static float nearestNF4(float value, float[] codebook) {
        // Clamp to valid range
        if (value <= codebook[0]) return codebook[0];
        if (value >= codebook[codebook.length - 1]) return codebook[codebook.length - 1];

        float best = codebook[0];
        float bestDist = Math.abs(value - best);
        for (int i = 1; i < codebook.length; i++) {
            float dist = Math.abs(value - codebook[i]);
            if (dist < bestDist) {
                bestDist = dist;
                best = codebook[i];
            }
        }
        return best;
    }

    /**
     * Quantize a weight matrix using block-wise FP4 (Float 4-bit) quantization.
     * <p>
     * FP4 represents values as a 4-bit floating-point number (1 sign, 2 exponent, 1 mantissa).
     * Each block of {@code blockSize} elements shares a scale factor.
     *
     * @param weight    Weight matrix to quantize [rows, cols]
     * @param blockSize Number of elements per quantization block
     * @return Dequantized approximation of weight
     */
    public static INDArray quantizeFP4(INDArray weight, int blockSize) {
        // FP4 codebook: 16 values covering sign + 2-bit exp + 1-bit mantissa
        final float[] FP4_CODEBOOK = {
            0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.4375f,
            0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 1.0f
        };

        long totalElements = weight.length();
        INDArray flat   = weight.reshape(totalElements);
        INDArray result = Nd4j.zeros(DataType.FLOAT, totalElements);

        int numBlocks = (int) Math.ceil((double) totalElements / blockSize);
        for (int b = 0; b < numBlocks; b++) {
            int start = b * blockSize;
            int end   = (int) Math.min(start + blockSize, totalElements);

            INDArray block = flat.get(NDArrayIndex.interval(start, end)).dup();
            float absMax = (float) block.amaxNumber().doubleValue();
            if (absMax == 0.0f) {
                result.put(new INDArrayIndex[]{NDArrayIndex.interval(start, end)},
                    Nd4j.zeros(DataType.FLOAT, end - start));
                block.close();
                continue;
            }

            float scale = absMax;
            float[] blockData = block.toFloatVector();
            block.close();

            float[] quantData = new float[blockData.length];
            for (int i = 0; i < blockData.length; i++) {
                float normalized = blockData[i] / scale;
                float sign = normalized >= 0 ? 1.0f : -1.0f;
                float absVal = Math.abs(normalized);
                quantData[i] = sign * nearestFP4(absVal, FP4_CODEBOOK) * scale;
            }

            INDArray quantBlock = Nd4j.createFromArray(quantData);
            result.put(new INDArrayIndex[]{NDArrayIndex.interval(start, end)}, quantBlock);
            quantBlock.close();
        }
        // NOTE: flat is a VIEW sharing the same DataBuffer as weight — do NOT close it here.
        // Closing flat would mark weight's DataBuffer as closed, invalidating the caller's array.

        return result.reshape(weight.shape());
    }

    /**
     * Find the nearest FP4 code value for |value| in [0, 1].
     */
    private static float nearestFP4(float value, float[] codebook) {
        if (value <= 0.0f) return 0.0f;
        if (value >= codebook[codebook.length - 1]) return codebook[codebook.length - 1];

        float best = codebook[0];
        float bestDist = Math.abs(value - best);
        for (int i = 1; i < codebook.length; i++) {
            float dist = Math.abs(value - codebook[i]);
            if (dist < bestDist) {
                bestDist = dist;
                best = codebook[i];
            }
        }
        return best;
    }

    /**
     * Compute the economy (thin) SVD of a matrix keeping only the top-{@code rank} singular
     * values. Returns [S, U, Vt] where:
     * <ul>
     *   <li>S  — singular values [min(m,n)] in descending order</li>
     *   <li>U  — left singular vectors [m, min(m,n)]</li>
     *   <li>Vt — right singular vectors transposed [min(m,n), n]</li>
     * </ul>
     *
     * @param matrix input matrix [m, n]
     * @param rank   number of singular values to use (only top-rank components will be accessed)
     * @return [S, U, Vt]
     */
    static INDArray[] svd(INDArray matrix, int rank) {
        // Use the high-level Svd op (fullUV=false → economy/thin SVD, computeUv=true).
        // Constructor: Svd(INDArray input, boolean fullUV, boolean computeUV, int switchNum)
        // Returns outputs in order: [S, U, V] where S=[k], U=[m,k], V=[n,k], k=min(m,n).
        // We need Vt = V.transpose() to get [k, n] as expected by callers.
        INDArray A = matrix.dataType() == DataType.FLOAT ? matrix.dup('c') : matrix.castTo(DataType.FLOAT).dup('c');
        INDArray[] svdOuts = Nd4j.exec(new Svd(A, false, true, Svd.DEFAULT_SWITCHNUM));
        A.close();
        // svdOuts[0] = S [k], svdOuts[1] = U [m, k], svdOuts[2] = V [n, k]
        // Transpose V to Vt = [k, n] as expected by the caller
        INDArray Vt = svdOuts[2].transpose().dup();
        svdOuts[2].close();
        return new INDArray[]{svdOuts[0], svdOuts[1], Vt};
    }

}
