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
package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the ONNX MultiHeadAttention operation.
 * 
 * This tests the native onnx_multi_head_attention op which handles pre-projected
 * Q, K, V tensors directly (as ONNX provides them) rather than requiring weight matrices.
 */
@Tag(TagNames.ONNX)
public class OnnxMultiHeadAttentionTest {

    /**
     * Test basic forward pass with pre-projected Q, K, V tensors.
     * ONNX MultiHeadAttention expects inputs in [batch, seq, hidden] format.
     */
    @Test
    public void testBasicMultiHeadAttention() {
        int batch = 2;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;  // 16

        // Create Q, K, V with shape [batch, seq, hidden]
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // Pre-allocate output arrays with expected shapes
        INDArray attnOutput = Nd4j.create(DataType.FLOAT, batch, seqLen, hidden);
        INDArray presentKey = Nd4j.create(DataType.FLOAT, batch, numHeads, seqLen, headDim);  // BHSD format
        INDArray presentValue = Nd4j.create(DataType.FLOAT, batch, numHeads, seqLen, headDim);  // BHSD format

        // Execute the op using DynamicCustomOp.builder with only required inputs (no empty arrays)
        // The op signature is: query, key, value, [attn_bias], [past_key], [past_value]
        // For basic case, we only pass query, key, value
        DynamicCustomOp op = DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value)
                .addOutputs(attnOutput, presentKey, presentValue)
                .addIntegerArguments(numHeads, 0)  // numHeads, useCausalMask=false
                .addFloatingPointArguments(0.0)    // scale=0 (auto)
                .build();

        Nd4j.exec(op);

        // Verify we got outputs
        assertNotNull(attnOutput);
        assertNotNull(presentKey);
        assertNotNull(presentValue);

        // Attention output should be [batch, seq, hidden]
        assertArrayEquals(new long[]{batch, seqLen, hidden}, attnOutput.shape(),
                "Attention output shape should be [batch, seq, hidden]");
        
        // Present key/value should be [batch, numHeads, seq, headDim] (BHSD)
        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, presentKey.shape(),
                "Present key shape should be [batch, numHeads, seq, headDim]");
        assertArrayEquals(new long[]{batch, numHeads, seqLen, headDim}, presentValue.shape(),
                "Present value shape should be [batch, numHeads, seq, headDim]");
        
        // Verify output is not all zeros (indicates computation happened)
        double absSum = attnOutput.ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "Attention output should not be all zeros");
    }

    /**
     * Test multi-head attention with causal masking (unidirectional).
     * Causal mask prevents attending to future tokens.
     */
    @Test
    public void testCausalMasking() {
        int batch = 1;
        int seqLen = 8;
        int numHeads = 4;
        int headDim = 16;
        int hidden = numHeads * headDim;  // 64

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // With causal mask
        OnnxMultiHeadAttention opCausal = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, true  // causal = true
        );
        INDArray[] outputsCausal = Nd4j.exec(opCausal);

        // Without causal mask
        OnnxMultiHeadAttention opNonCausal = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, false  // causal = false
        );
        INDArray[] outputsNonCausal = Nd4j.exec(opNonCausal);

        // Both should produce valid outputs
        assertNotNull(outputsCausal[0]);
        assertNotNull(outputsNonCausal[0]);
        assertArrayEquals(outputsCausal[0].shape(), outputsNonCausal[0].shape());

        // The outputs should be different due to causal masking
        double diff = outputsCausal[0].sub(outputsNonCausal[0]).ameanNumber().doubleValue();
        assertTrue(diff > 1e-6, "Causal and non-causal outputs should differ");
    }

    /**
     * Test multi-head attention with past key/value (KV cache for autoregressive generation).
     * Past KV is in ONNX format: [batch, numHeads, pastSeq, headDim]
     */
    @Test
    public void testWithPastKeyValue() {
        int batch = 1;
        int seqQ = 1;  // Single new token
        int pastSeq = 10;  // Past context
        int numHeads = 4;
        int headDim = 16;
        int hidden = numHeads * headDim;

        // New query for single token
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden);
        // New key/value for single token
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden);

        // Past key/value in ONNX format [batch, numHeads, pastSeq, headDim]
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value,
                null, pastKey, pastValue,
                numHeads, 0.0, true  // causal for autoregressive
        );

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertTrue(outputs.length >= 3, "Should have 3 outputs: attn, presentKey, presentValue");

        // Attention output for single token
        INDArray attnOutput = outputs[0];
        assertArrayEquals(new long[]{batch, seqQ, hidden}, attnOutput.shape());

        // Present key should include past + new: [batch, numHeads, pastSeq + seqQ, headDim]
        INDArray presentKey = outputs[1];
        assertArrayEquals(new long[]{batch, numHeads, pastSeq + seqQ, headDim}, presentKey.shape(),
                "Present key should concatenate past and new");

        // Present value should include past + new
        INDArray presentValue = outputs[2];
        assertArrayEquals(new long[]{batch, numHeads, pastSeq + seqQ, headDim}, presentValue.shape(),
                "Present value should concatenate past and new");
    }

    /**
     * Incremental decode correctness check:
     * output for token t with (past_key, past_value, new_qkv) should match
     * full causal attention output at token t from a single full-sequence call.
     */
    @Test
    public void testIncrementalMatchesFullCausal() {
        int batch = 1;
        int pastSeq = 6;
        int newSeq = 1;
        int totalSeq = pastSeq + newSeq;
        int numHeads = 4;
        int headDim = 16;
        int hidden = numHeads * headDim;

        INDArray queryFull = Nd4j.randn(DataType.FLOAT, batch, totalSeq, hidden);
        INDArray keyFull = Nd4j.randn(DataType.FLOAT, batch, totalSeq, hidden);
        INDArray valueFull = Nd4j.randn(DataType.FLOAT, batch, totalSeq, hidden);

        // Full-sequence causal output
        INDArray[] fullOut = Nd4j.exec(new OnnxMultiHeadAttention(
                queryFull, keyFull, valueFull,
                null, null, null,
                numHeads, 0.0, true));
        INDArray fullLast = fullOut[0].get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(totalSeq - 1),
                NDArrayIndex.all()).dup();

        // Incremental decode inputs: last token query/key/value + past key/value cache
        INDArray queryNew = queryFull.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(pastSeq, totalSeq),
                NDArrayIndex.all()).dup();
        INDArray keyNew = keyFull.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(pastSeq, totalSeq),
                NDArrayIndex.all()).dup();
        INDArray valueNew = valueFull.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(pastSeq, totalSeq),
                NDArrayIndex.all()).dup();

        INDArray keyPast = keyFull.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq),
                NDArrayIndex.all()).dup()
                .reshape(batch, pastSeq, numHeads, headDim)
                .permute(0, 2, 1, 3).dup();  // BSHD -> BHSD
        INDArray valuePast = valueFull.get(
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq),
                NDArrayIndex.all()).dup()
                .reshape(batch, pastSeq, numHeads, headDim)
                .permute(0, 2, 1, 3).dup();  // BSHD -> BHSD

        INDArray[] incOut = Nd4j.exec(new OnnxMultiHeadAttention(
                queryNew, keyNew, valueNew,
                null, keyPast, valuePast,
                numHeads, 0.0, true));
        INDArray incLast = incOut[0].get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.all()).dup();

        double maxDiff = fullLast.sub(incLast).normmaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Incremental causal output should match full causal output for last token. maxDiff=" + maxDiff);
    }

    /**
     * Test multi-head attention with attention bias (relative position bias or custom mask).
     * Attention bias shape: [batch, numHeads, seqQ, seqKV]
     */
    @Test
    public void testWithAttentionBias() {
        int batch = 2;
        int seqLen = 6;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // Attention bias [batch, numHeads, seqQ, seqKV]
        INDArray attnBias = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, seqLen);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value,
                attnBias, null, null,
                numHeads, 0.0, false
        );

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertNotNull(outputs[0]);
        assertArrayEquals(new long[]{batch, seqLen, hidden}, outputs[0].shape());
    }

    /**
     * Test with custom scale factor.
     */
    @Test
    public void testCustomScale() {
        int batch = 1;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;
        double customScale = 0.5;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // With custom scale
        OnnxMultiHeadAttention opCustom = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, customScale, false
        );
        INDArray[] outputsCustom = Nd4j.exec(opCustom);

        // With auto scale (0.0 means auto = 1/sqrt(headDim))
        OnnxMultiHeadAttention opAuto = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, false
        );
        INDArray[] outputsAuto = Nd4j.exec(opAuto);

        assertNotNull(outputsCustom[0]);
        assertNotNull(outputsAuto[0]);

        // Different scales should produce different outputs
        double diff = outputsCustom[0].sub(outputsAuto[0]).ameanNumber().doubleValue();
        assertTrue(diff > 1e-6, "Custom and auto scale should produce different outputs");
    }

    /**
     * Test SameDiff integration - the op can be used in a SameDiff graph.
     */
    @Test
    public void testSameDiffIntegration() {
        int batch = 2;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;

        SameDiff sd = SameDiff.create();

        SDVariable query = sd.placeHolder("query", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, batch, seqLen, hidden);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, batch, seqLen, hidden);

        // Create the op via SameDiff
        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                sd, query, key, value,
                null, null, null,
                numHeads, 0.0, false
        );

        SDVariable[] outputs = op.outputVariables();
        assertNotNull(outputs);
        assertTrue(outputs.length >= 1);

        // Execute the graph
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("query", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));
        inputs.put("key", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));
        inputs.put("value", Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden));

        Map<String, INDArray> results = sd.output(inputs, outputs[0].name());

        INDArray attnOutput = results.get(outputs[0].name());
        assertNotNull(attnOutput);
        assertArrayEquals(new long[]{batch, seqLen, hidden}, attnOutput.shape());
    }

    /**
     * Test cross-attention (different Q and K/V sequence lengths).
     * This is common in encoder-decoder models.
     */
    @Test
    public void testCrossAttention() {
        int batch = 2;
        int seqQ = 4;   // Query sequence length
        int seqKV = 8;  // Key/Value sequence length (different!)
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, hidden);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, false  // no causal mask for cross attention
        );

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        // Output should match query sequence length
        assertArrayEquals(new long[]{batch, seqQ, hidden}, outputs[0].shape(),
                "Cross attention output should have query's sequence length");
    }

    /**
     * Test with FP16 (half precision) inputs for mixed-precision training.
     */
    @Test
    public void testFloat16() {
        int batch = 1;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT16, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT16, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT16, batch, seqLen, hidden);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, false
        );

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertNotNull(outputs[0]);
        assertEquals(DataType.FLOAT16, outputs[0].dataType(),
                "Output should preserve FP16 data type");
        assertArrayEquals(new long[]{batch, seqLen, hidden}, outputs[0].shape());
    }

    /**
     * Test various head configurations.
     */
    @Test
    public void testVariousHeadConfigurations() {
        int batch = 1;
        int seqLen = 4;

        // Test different head configurations: 1, 2, 4, 8 heads
        int[] headCounts = {1, 2, 4, 8};
        int headDim = 8;

        for (int numHeads : headCounts) {
            int hidden = numHeads * headDim;

            INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
            INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
            INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

            OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                    query, key, value,
                    null, null, null,
                    numHeads, 0.0, false
            );

            INDArray[] outputs = Nd4j.exec(op);

            assertNotNull(outputs, "Failed for numHeads=" + numHeads);
            assertArrayEquals(new long[]{batch, seqLen, hidden}, outputs[0].shape(),
                    "Shape mismatch for numHeads=" + numHeads);
        }
    }

    /**
     * Test Grouped Query Attention (GQA) where K/V have fewer heads than Q.
     * This is the pattern used by SmolDocling: Q has 9 heads, K/V have 3 heads.
     * Q hidden=576 (9*64), KV hidden=192 (3*64), headDim=64.
     */
    @Test
    public void testGroupedQueryAttention() {
        int batch = 1;
        int seqLen = 4;
        int numQHeads = 9;
        int numKvHeads = 3;
        int headDim = 64;
        int qHidden = numQHeads * headDim;   // 576
        int kvHidden = numKvHeads * headDim;  // 192

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, qHidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, kvHidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, kvHidden);

        DynamicCustomOp op = DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value)
                .addIntegerArguments((long) numQHeads, 0L)
                .addFloatingPointArguments(0.0)
                .build();

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertTrue(outputs.length >= 1);
        // Output should have Q's hidden dim
        assertArrayEquals(new long[]{batch, seqLen, qHidden}, outputs[0].shape(),
                "GQA output shape should match Q hidden");

        if (outputs.length >= 2) {
            // Present key should use numKvHeads, not numQHeads
            assertArrayEquals(new long[]{batch, numKvHeads, seqLen, headDim}, outputs[1].shape(),
                    "Present key should use numKvHeads");
        }
        if (outputs.length >= 3) {
            assertArrayEquals(new long[]{batch, numKvHeads, seqLen, headDim}, outputs[2].shape(),
                    "Present value should use numKvHeads");
        }

        double absSum = outputs[0].ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "GQA output should not be all zeros");
    }

    /**
     * Test GQA with past key/value (KV cache) — the exact SmolDocling decode pattern.
     * Q=[1,1,576] (9 heads), K=[1,1,192] (3 heads), past_key=[1,3,10,64].
     */
    @Test
    public void testGQAWithPastKeyValue() {
        int batch = 1;
        int seqQ = 1;
        int pastSeq = 10;
        int numQHeads = 9;
        int numKvHeads = 3;
        int headDim = 64;
        int qHidden = numQHeads * headDim;   // 576
        int kvHidden = numKvHeads * headDim;  // 192

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, qHidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqQ, kvHidden);

        // Past KV in BHSD: [batch, numKvHeads, pastSeq, headDim]
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numKvHeads, pastSeq, headDim);

        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value,
                null, pastKey, pastValue,
                numQHeads, 0.0, true
        );

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertTrue(outputs.length >= 3);

        // Attention output
        assertArrayEquals(new long[]{batch, seqQ, qHidden}, outputs[0].shape(),
                "GQA+KV output shape should match Q hidden");

        // Present key/value: [batch, numKvHeads, pastSeq+seqQ, headDim]
        int totalSeq = pastSeq + seqQ;
        assertArrayEquals(new long[]{batch, numKvHeads, totalSeq, headDim}, outputs[1].shape(),
                "Present key should have numKvHeads and totalSeq");
        assertArrayEquals(new long[]{batch, numKvHeads, totalSeq, headDim}, outputs[2].shape(),
                "Present value should have numKvHeads and totalSeq");

        double absSum = outputs[0].ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "GQA+KV output should not be all zeros");
    }

    /**
     * Test multiple GQA head configurations:
     * 9Q/3KV, 12Q/4KV, 8Q/2KV, 6Q/1KV (MQA)
     */
    @Test
    public void testMultipleGQAConfigurations() {
        int batch = 1;
        int seqLen = 4;
        int headDim = 32;

        int[][] configs = {
            {9, 3},   // SmolDocling-like
            {12, 4},  // 3:1 ratio
            {8, 2},   // 4:1 ratio
            {6, 1},   // MQA (Multi-Query Attention)
            {4, 4},   // Standard MHA (no grouping)
        };

        for (int[] cfg : configs) {
            int numQHeads = cfg[0];
            int numKvHeads = cfg[1];
            int qHidden = numQHeads * headDim;
            int kvHidden = numKvHeads * headDim;

            INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, qHidden);
            INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, kvHidden);
            INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, kvHidden);

            DynamicCustomOp op = DynamicCustomOp.builder("onnx_multi_head_attention")
                    .addInputs(query, key, value)
                    .addIntegerArguments((long) numQHeads, 0L)
                    .addFloatingPointArguments(0.0)
                    .build();

            INDArray[] outputs = Nd4j.exec(op);

            assertNotNull(outputs, "Failed for Q=" + numQHeads + " KV=" + numKvHeads);
            assertArrayEquals(new long[]{batch, seqLen, qHidden}, outputs[0].shape(),
                    "Output shape mismatch for Q=" + numQHeads + " KV=" + numKvHeads);

            if (outputs.length >= 2) {
                assertArrayEquals(new long[]{batch, numKvHeads, seqLen, headDim}, outputs[1].shape(),
                        "Present key shape mismatch for Q=" + numQHeads + " KV=" + numKvHeads);
            }
        }
    }
}
