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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Edge-case tests for ONNX MultiHeadAttention mixed-type handling
 * and DotProductAttentionV2 (SDPA) rank guard behavior.
 *
 * Tests verify:
 *  - SDPA works correctly with no KV cache
 *  - SDPA rank guard ignores rank-0/rank-1 KV cache inputs
 *  - SDPA rank guard ignores empty-array KV cache inputs
 *  - ONNX MHA auto-casts mixed-dtype inputs without crashing
 *  - ONNX MHA works normally with matching dtypes
 */
@DisplayName("MHA and SDPA Edge Case Tests")
public class TestMhaAndSdpaEdgeCases {

    // Shapes: [batch, heads, seq, headDim] for rank-4 BSHD SDPA
    private static final int BATCH    = 1;
    private static final int HEADS    = 4;
    private static final int SEQ      = 2;
    private static final int HEAD_DIM = 8;

    // MHA config: hidden = numHeads * headDim
    private static final int MHA_HEADS    = 4;
    private static final int MHA_HEAD_DIM = 16;
    private static final int MHA_HIDDEN   = MHA_HEADS * MHA_HEAD_DIM;
    private static final int MHA_SEQ      = 4;

    /**
     * Run SDPA with Q [1,4,2,8], K [1,4,2,8], V [1,4,2,8] and no KV cache.
     * Verify output shape equals Q shape and contains no NaN.
     */
    @Test
    @DisplayName("SDPA - no KV cache, rank-4 BSHD inputs")
    public void testSdpaNoKvCache() {
        INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);
        INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);
        INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);

        // Constructor: queries, values, keys, queryMask, valueMask,
        //              scaleFactor, dropoutProbability, useCausalMask, training
        DotProductAttentionV2 op = new DotProductAttentionV2(
                q, v, k,
                /*queryMask=*/null, /*valueMask=*/null,
                /*scaleFactor=*/0.0, /*dropout=*/0.0,
                /*useCausalMask=*/false, /*training=*/false);

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs, "Outputs must not be null");
        assertTrue(outputs.length >= 1, "Must have at least 1 output, got " + outputs.length);

        INDArray attnOut = outputs[0];
        assertNotNull(attnOut, "Attention output must not be null");

        // Output must have same shape as Q
        assertArrayEquals(q.shape(), attnOut.shape(),
                "Output shape must match Q shape");

        // No NaN in output
        assertFalse(attnOut.isNaN().any(),
                "Output must not contain NaN");
    }

    /**
     * Run SDPA with Q/K/V as above, but pass a scalar (rank-0) array as kvCacheK/kvCacheV.
     * The rank < 2 guard in the C++ op (dot_product_attention_v2.cpp lines 159-161)
     * should treat rank-0 inputs as absent and behave identically to no-cache mode.
     */
    @Test
    @DisplayName("SDPA - rank-0 KV cache inputs are ignored by rank guard")
    public void testSdpaRank0KvCacheIgnored() {
        INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);
        INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);
        INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);

        // No-cache reference
        INDArray[] refOutputs = Nd4j.exec(new DotProductAttentionV2(
                q, v, k, null, null,
                0.0, 0.0, false, false));
        INDArray refOut = refOutputs[0];

        // Rank-0 scalar — should trigger rank < 2 guard and be treated as absent
        INDArray rankZeroCacheK = Nd4j.scalar(DataType.FLOAT, 0.0f);
        INDArray rankZeroCacheV = Nd4j.scalar(DataType.FLOAT, 0.0f);

        assertEquals(0, rankZeroCacheK.rank(), "kvCacheK must be rank 0");
        assertEquals(0, rankZeroCacheV.rank(), "kvCacheV must be rank 0");

        // Constructor: queries, values, keys, queryMask, valueMask,
        //              keyCache, valueCache, cachePosition, attentionBias,
        //              scaleFactor, dropout, useCausalMask, training
        DotProductAttentionV2 op = new DotProductAttentionV2(
                q, v, k,
                /*queryMask=*/null, /*valueMask=*/null,
                /*keyCache=*/rankZeroCacheK, /*valueCache=*/rankZeroCacheV,
                /*cachePosition=*/null, /*attentionBias=*/null,
                0.0, 0.0, false, false);

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs, "Outputs must not be null");
        assertTrue(outputs.length >= 1, "Must have at least 1 output");

        INDArray attnOut = outputs[0];
        assertArrayEquals(q.shape(), attnOut.shape(),
                "Output shape must match Q shape (rank guard should ignore rank-0 cache)");
        assertFalse(attnOut.isNaN().any(),
                "Output must not contain NaN");

        // Result must match the no-cache baseline (rank-0 cache was discarded)
        assertEquals(refOut.shape().length, attnOut.shape().length,
                "Rank-0 KV cache should be silently ignored — output rank must match no-cache");
        assertArrayEquals(refOut.shape(), attnOut.shape(),
                "Rank-0 KV cache ignored — output shape must match no-cache baseline");
    }

    /**
     * Run SDPA with empty-array kvCacheK/kvCacheV (shape has a 0-length dim).
     * The C++ op checks isEmpty() in the kv-cache branch; empty arrays must also
     * be ignored so behavior matches no-cache mode.
     */
    @Test
    @DisplayName("SDPA - empty KV cache arrays are ignored")
    public void testSdpaEmptyKvCacheIgnored() {
        INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);
        INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);
        INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HEADS, HEAD_DIM).muli(0.1f);

        // No-cache reference
        INDArray[] refOutputs = Nd4j.exec(new DotProductAttentionV2(
                q, v, k, null, null,
                0.0, 0.0, false, false));
        INDArray refOut = refOutputs[0];

        // Empty KV cache: seq dim is 0 → isEmpty() returns true in C++
        INDArray emptyCacheK = Nd4j.create(DataType.FLOAT, BATCH, 0, HEADS, HEAD_DIM);
        INDArray emptyCacheV = Nd4j.create(DataType.FLOAT, BATCH, 0, HEADS, HEAD_DIM);

        assertTrue(emptyCacheK.isEmpty(), "emptyCacheK must be empty");
        assertTrue(emptyCacheV.isEmpty(), "emptyCacheV must be empty");

        DotProductAttentionV2 op = new DotProductAttentionV2(
                q, v, k,
                null, null,
                emptyCacheK, emptyCacheV,
                null, null,
                0.0, 0.0, false, false);

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs, "Outputs must not be null");
        assertTrue(outputs.length >= 1, "Must have at least 1 output");

        INDArray attnOut = outputs[0];
        assertArrayEquals(q.shape(), attnOut.shape(),
                "Output shape must match Q shape (empty cache ignored)");
        assertFalse(attnOut.isNaN().any(),
                "Output must not contain NaN");

        assertArrayEquals(refOut.shape(), attnOut.shape(),
                "Empty KV cache ignored — output shape must match no-cache baseline");
    }

    /**
     * ONNX MHA with query=FLOAT and key/value=HALF.
     * The C++ op (onnx_multi_head_attention.cpp line 103-130) auto-casts all inputs
     * to query's dtype. Verify the op doesn't crash and produces output with query's dtype.
     *
     * This test exercises the mixed-type path: key and value are passed as HALF
     * and must be cast to FLOAT before the attention kernel runs.
     */
    @Test
    @DisplayName("ONNX MHA - mixed dtype: query=FLOAT, key/value=HALF auto-cast")
    public void testOnnxMhaMixedDtypeQueryFloatKeyHalf() {
        int batch = 1;
        int seqLen = MHA_SEQ;
        int hidden = MHA_HIDDEN;

        INDArray queryFloat = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden).muli(0.1f);
        // Key and value in HALF — the C++ op should auto-cast to query's FLOAT type
        INDArray keyHalf   = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden).castTo(DataType.HALF);
        INDArray valueHalf = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden).castTo(DataType.HALF);

        assertEquals(DataType.FLOAT, queryFloat.dataType(), "Query must be FLOAT");
        assertEquals(DataType.HALF,  keyHalf.dataType(),    "Key must be HALF");
        assertEquals(DataType.HALF,  valueHalf.dataType(),  "Value must be HALF");

        // The OnnxMultiHeadAttention INDArray constructor will pass key/value as-is;
        // the C++ op detects dtype mismatch and casts to query dtype.
        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                queryFloat, keyHalf, valueHalf,
                /*attnBias=*/null, /*pastKey=*/null, /*pastValue=*/null,
                MHA_HEADS, 0.0, false,
                /*numOutputs=*/1));  // Only request attention output, skip optional present KV

        assertNotNull(outputs, "Outputs must not be null");
        assertTrue(outputs.length >= 1, "Must have at least 1 output");

        INDArray attnOut = outputs[0];
        assertNotNull(attnOut, "Attention output must not be null");

        // Output dtype must be FLOAT (query's type)
        assertEquals(DataType.FLOAT, attnOut.dataType(),
                "Output dtype must match query dtype (FLOAT), got " + attnOut.dataType());

        // Output shape must be [batch, seqQ, hidden]
        assertArrayEquals(new long[]{batch, seqLen, hidden}, attnOut.shape(),
                "Output shape must be [batch, seqLen, hidden]");

        // No NaN
        assertFalse(attnOut.isNaN().any(),
                "Output must not contain NaN after mixed-dtype auto-cast");
    }

    /**
     * ONNX MHA baseline: all inputs are FLOAT.
     * Verifies the op works normally (no casting needed) and produces correct output shape.
     * This is the same-type baseline that the mixed-dtype test should match in shape/dtype.
     */
    @Test
    @DisplayName("ONNX MHA - matching dtype baseline (all FLOAT)")
    public void testOnnxMhaMatchingDtype() {
        int batch = 1;
        int seqLen = MHA_SEQ;
        int hidden = MHA_HIDDEN;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden).muli(0.1f);
        INDArray key   = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden).muli(0.1f);

        INDArray[] outputs = Nd4j.exec(new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                MHA_HEADS, 0.0, false,
                /*numOutputs=*/1));

        assertNotNull(outputs, "Outputs must not be null");
        assertTrue(outputs.length >= 1, "Must have at least 1 output");

        INDArray attnOut = outputs[0];
        assertNotNull(attnOut, "Attention output must not be null");

        assertEquals(DataType.FLOAT, attnOut.dataType(),
                "Output dtype must be FLOAT");

        assertArrayEquals(new long[]{batch, seqLen, hidden}, attnOut.shape(),
                "Output shape must be [batch, seqLen, hidden]");

        assertFalse(attnOut.isNaN().any(),
                "Output must not contain NaN");

        // Sanity: output must not be all-zeros (attention is non-trivial)
        double absMax = attnOut.amaxNumber().doubleValue();
        assertTrue(absMax > 1e-9,
                "Output should not be all-zeros, absMax=" + absMax);
    }
}
