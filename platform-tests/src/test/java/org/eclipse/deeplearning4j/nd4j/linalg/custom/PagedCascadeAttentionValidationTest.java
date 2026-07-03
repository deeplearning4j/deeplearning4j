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

package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Functional correctness validation for the paged_attention_forward and
 * cascade_attention custom ops against straightforward double-precision
 * reference implementations. These ops previously had no functional test
 * coverage on any backend (only op-trait enumeration).
 *
 * Semantics mirrored from the helpers (CPU/CUDA implement the same contract):
 *  - paged_attention_forward: NON-causal attention of a single-position query
 *    over the first contextLen positions gathered through the page table.
 *    GQA mapping: kvHead = h * numKvHeads / numHeads (integer division).
 *    Auto scale (tArg 0 = 0): 1/sqrt(headDim).
 *  - cascade_attention: NON-causal full attention computed in chunkSize chunks
 *    merged via log-sum-exp; result must match plain softmax attention.
 */
@NativeTag
@Tag(TagNames.FULL_CI)
public class PagedCascadeAttentionValidationTest extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    // ─── paged_attention_forward ────────────────────────────────────────────

    /**
     * Decode-shaped case: batch=1 (parallelism must come from heads), GQA
     * (numKvHeads < numHeads), context spanning multiple blocks with a partial
     * last block, and a scattered (non-identity) page table.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPagedAttentionDecodeBatch1Gqa(Nd4jBackend backend) {
        int batch = 1, numHeads = 8, numKvHeads = 4, headDim = 16;
        int blockSize = 4, numBlocks = 8, maxBlocksPerSeq = 4;
        int ctxLen = 11; // 3 blocks: 4 + 4 + 3 (partial)

        Nd4j.getRandom().setSeed(42);
        INDArray query = Nd4j.rand(DataType.FLOAT, batch, 1, numHeads, headDim).subi(0.5);
        INDArray keyPool = Nd4j.rand(DataType.FLOAT, numBlocks, blockSize, numKvHeads, headDim).subi(0.5);
        INDArray valuePool = Nd4j.rand(DataType.FLOAT, numBlocks, blockSize, numKvHeads, headDim).subi(0.5);
        // Scattered physical blocks; -1 padding beyond the 3 used entries.
        INDArray pageTables = Nd4j.createFromArray(new int[][]{{5, 2, 7, -1}}).castTo(DataType.INT32);
        INDArray contextLens = Nd4j.createFromArray(new int[]{ctxLen}).castTo(DataType.INT32);

        INDArray out = execPaged(query, keyPool, valuePool, pageTables, contextLens,
                blockSize, numHeads, numKvHeads, headDim);

        assertPagedMatchesReference(query, keyPool, valuePool,
                new int[][]{{5, 2, 7, -1}}, new int[]{ctxLen},
                blockSize, numHeads, numKvHeads, headDim, out);
    }

    /** Batched case with varied context lengths (all > 0) and MHA (numKvHeads == numHeads). */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testPagedAttentionBatchVariedContext(Nd4jBackend backend) {
        int batch = 3, numHeads = 4, numKvHeads = 4, headDim = 8;
        int blockSize = 4, numBlocks = 12, maxBlocksPerSeq = 3;
        int[] ctxLens = {5, 1, 8};
        int[][] pages = {{0, 3, -1}, {6, -1, -1}, {9, 4, -1}};

        Nd4j.getRandom().setSeed(1234);
        INDArray query = Nd4j.rand(DataType.FLOAT, batch, 1, numHeads, headDim).subi(0.5);
        INDArray keyPool = Nd4j.rand(DataType.FLOAT, numBlocks, blockSize, numKvHeads, headDim).subi(0.5);
        INDArray valuePool = Nd4j.rand(DataType.FLOAT, numBlocks, blockSize, numKvHeads, headDim).subi(0.5);
        INDArray pageTables = Nd4j.createFromArray(pages).castTo(DataType.INT32);
        INDArray contextLens = Nd4j.createFromArray(ctxLens).castTo(DataType.INT32);

        INDArray out = execPaged(query, keyPool, valuePool, pageTables, contextLens,
                blockSize, numHeads, numKvHeads, headDim);

        assertPagedMatchesReference(query, keyPool, valuePool, pages, ctxLens,
                blockSize, numHeads, numKvHeads, headDim, out);
    }

    private static INDArray execPaged(INDArray query, INDArray keyPool, INDArray valuePool,
                                      INDArray pageTables, INDArray contextLens,
                                      int blockSize, int numHeads, int numKvHeads, int headDim) {
        long batch = query.size(0);
        INDArray out = Nd4j.create(DataType.FLOAT, batch, 1, numHeads, headDim);
        DynamicCustomOp op = DynamicCustomOp.builder("paged_attention_forward")
                .addInputs(query, keyPool, valuePool, pageTables, contextLens)
                .addOutputs(out)
                .addIntegerArguments(blockSize, numHeads, numKvHeads, headDim)
                .addFloatingPointArguments(0.0) // 0 = auto scale (1/sqrt(headDim))
                .build();
        Nd4j.exec(op);
        return out;
    }

    private static void assertPagedMatchesReference(INDArray query, INDArray keyPool, INDArray valuePool,
                                                    int[][] pages, int[] ctxLens,
                                                    int blockSize, int numHeads, int numKvHeads, int headDim,
                                                    INDArray out) {
        int batch = ctxLens.length;
        double scale = 1.0 / Math.sqrt(headDim);
        double maxDiff = 0.0;

        for (int b = 0; b < batch; b++) {
            int ctxLen = ctxLens[b];
            for (int h = 0; h < numHeads; h++) {
                int kvHead = (numKvHeads > 0 && numKvHeads < numHeads) ? (h * numKvHeads / numHeads) : h;

                double[] scores = new double[ctxLen];
                double maxScore = Double.NEGATIVE_INFINITY;
                for (int pos = 0; pos < ctxLen; pos++) {
                    int physicalBlock = pages[b][pos / blockSize];
                    int off = pos % blockSize;
                    double dot = 0;
                    for (int d = 0; d < headDim; d++) {
                        dot += query.getDouble(b, 0, h, d) * keyPool.getDouble(physicalBlock, off, kvHead, d);
                    }
                    scores[pos] = dot * scale;
                    if (scores[pos] > maxScore) maxScore = scores[pos];
                }
                double sumExp = 0;
                for (int pos = 0; pos < ctxLen; pos++) {
                    scores[pos] = Math.exp(scores[pos] - maxScore);
                    sumExp += scores[pos];
                }
                for (int d = 0; d < headDim; d++) {
                    double acc = 0;
                    for (int pos = 0; pos < ctxLen; pos++) {
                        int physicalBlock = pages[b][pos / blockSize];
                        int off = pos % blockSize;
                        acc += (scores[pos] / sumExp) * valuePool.getDouble(physicalBlock, off, kvHead, d);
                    }
                    double got = out.getDouble(b, 0, h, d);
                    maxDiff = Math.max(maxDiff, Math.abs(got - acc));
                }
            }
        }
        assertTrue(maxDiff < TOL, "paged_attention_forward deviates from reference: maxAbsDiff=" + maxDiff);
    }

    // ─── cascade_attention ──────────────────────────────────────────────────

    /**
     * Chunked log-sum-exp attention must equal plain softmax attention.
     * Covers: kvLen not a multiple of chunkSize (37 = 4*8+5), kvLen smaller
     * than chunkSize (single-chunk path), and multi-query reuse of the
     * per-thread scratch buffers.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCascadeAttentionMatchesPlainReference(Nd4jBackend backend) {
        // kvLen spans multiple chunks with a partial tail
        runCascadeCase(2, 3, 4, 37, 8, 8, 7);
        // kvLen < chunkSize: single-chunk (first-chunk init path only)
        runCascadeCase(1, 2, 3, 5, 4, 16, 99);
        // kvLen an exact multiple of chunkSize
        runCascadeCase(1, 2, 2, 16, 8, 8, 1001);
    }

    private static void runCascadeCase(int batch, int heads, int queryLen, int kvLen, int headDim,
                                       int chunkSize, long seed) {
        Nd4j.getRandom().setSeed(seed);
        INDArray q = Nd4j.rand(DataType.FLOAT, batch, heads, queryLen, headDim).subi(0.5);
        INDArray k = Nd4j.rand(DataType.FLOAT, batch, heads, kvLen, headDim).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, batch, heads, kvLen, headDim).subi(0.5);
        INDArray out = Nd4j.create(DataType.FLOAT, batch, heads, queryLen, headDim);

        DynamicCustomOp op = DynamicCustomOp.builder("cascade_attention")
                .addInputs(q, k, v)
                .addOutputs(out)
                .addIntegerArguments(chunkSize)
                .build();
        Nd4j.exec(op);

        double scale = 1.0 / Math.sqrt(headDim);
        double maxDiff = 0.0;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int qi = 0; qi < queryLen; qi++) {
                    double[] scores = new double[kvLen];
                    double maxScore = Double.NEGATIVE_INFINITY;
                    for (int ki = 0; ki < kvLen; ki++) {
                        double dot = 0;
                        for (int d = 0; d < headDim; d++) {
                            dot += q.getDouble(b, h, qi, d) * k.getDouble(b, h, ki, d);
                        }
                        scores[ki] = dot * scale;
                        if (scores[ki] > maxScore) maxScore = scores[ki];
                    }
                    double sumExp = 0;
                    for (int ki = 0; ki < kvLen; ki++) {
                        scores[ki] = Math.exp(scores[ki] - maxScore);
                        sumExp += scores[ki];
                    }
                    for (int d = 0; d < headDim; d++) {
                        double acc = 0;
                        for (int ki = 0; ki < kvLen; ki++) {
                            acc += (scores[ki] / sumExp) * v.getDouble(b, h, ki, d);
                        }
                        double got = out.getDouble(b, h, qi, d);
                        maxDiff = Math.max(maxDiff, Math.abs(got - acc));
                    }
                }
            }
        }
        assertTrue(maxDiff < TOL, "cascade_attention deviates from plain-attention reference: maxAbsDiff="
                + maxDiff + " (b=" + batch + ",h=" + heads + ",q=" + queryLen + ",kv=" + kvLen
                + ",d=" + headDim + ",chunk=" + chunkSize + ")");
    }
}
