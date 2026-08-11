/*
 *  ******************************************************************************
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
package org.nd4j.autodiff.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone tests for DSP segment behavior with value-key ops and in-place fusion.
 *
 * These test SPECIFIC scenarios that have caused production crashes:
 * 1. Value-key ops (range/create) in segments with changing input values
 * 2. In-place fusion corrupting buffers read by value-dependent ops (reshape_no_copy)
 * 3. Multi-consumer output slots and in-place fusion interaction
 */
@Slf4j
@Tag("samediff")
@DisplayName("DSP Value-Key Segment Correctness")
public class DspValueKeySegmentTest {

    /**
     * Test 1: Value-key ops in segments with changing position_id.
     *
     * Simulates the decode pattern: shape_of → concat → reshape_no_copy
     * where concat's inputs change each step (position_id increments).
     * The segment containing these ops must produce correct shapes on EVERY step,
     * not just the first (capture) step.
     */
    @Test
    @DisplayName("reshape_no_copy with changing concat shape — 10 decode steps")
    public void testReshapeWithChangingConcatShape() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Simulate decode: input is [1, 1, hidden] (batch=1, seq=1, hidden=64)
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 1, 64);
        // position_id changes each step — this is the value-key input
        SDVariable positionId = sd.placeHolder("position_id", DataType.INT64, 1);

        // Shape chain: shape_of → gather → concat → reshape_no_copy
        // This pattern appears in every transformer decode step
        SDVariable inputShape = sd.shape(input);  // [1, 1, 64]
        SDVariable batchSize = sd.gather(inputShape, sd.constant(Nd4j.scalar(0)), 0);  // scalar: 1
        SDVariable hiddenSize = sd.gather(inputShape, sd.constant(Nd4j.scalar(2)), 0); // scalar: 64

        // concat([batchSize, hiddenSize]) = [1, 64] — this is the new shape
        SDVariable newShape = sd.concat(0, batchSize, hiddenSize);

        // reshape using the concat output as shape
        SDVariable squeezed = sd.squeeze(input, 1);  // [1, 64]
        SDVariable reshaped = sd.reshape("output", squeezed, newShape);

        // Also use position_id in a computation to make it a real dependency
        SDVariable posFloat = positionId.castTo(DataType.FLOAT);
        SDVariable scaled = sd.math.add("scaled_output", reshaped, posFloat);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        // Run 10 decode steps with changing position_id
        for (int step = 0; step < 10; step++) {
            INDArray posArr = Nd4j.scalar(DataType.INT64, step + 1);

            Map<String, INDArray> out = sd.output(
                    Map.of("input", inputArr, "position_id", posArr),
                    "scaled_output");

            INDArray result = out.get("scaled_output");
            assertNotNull(result, "Step " + step + ": output is null");
            assertArrayEquals(new long[]{1, 64}, result.shape(),
                    "Step " + step + ": wrong output shape");

            // Verify values are correct: reshaped + position_id
            float expectedOffset = step + 1;
            float actualFirst = result.getFloat(0);
            float inputFirst = inputArr.getFloat(0, 0, 0);
            assertEquals(inputFirst + expectedOffset, actualFirst, 1e-4f,
                    "Step " + step + ": incorrect value — reshape or concat corrupted");
        }
    }

    /**
     * Test 2: Multi-consumer output slot with in-place fusion.
     *
     * Tests the specific crash scenario: an elementwise op's output feeds BOTH
     * a value-dependent op (reshape_no_copy) AND another elementwise consumer.
     * In-place fusion on the second consumer must NOT corrupt the buffer that
     * reshape_no_copy reads.
     */
    @Test
    @DisplayName("in-place fusion must not corrupt multi-consumer shape buffers")
    public void testInPlaceFusionMultiConsumer() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Input: [1, 1, 64]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 1, 64);

        // Create a shape tensor that feeds BOTH reshape AND another elementwise op
        SDVariable inputShape = sd.shape(input);  // INT64 [3]: [1, 1, 64]
        SDVariable batchDim = sd.gather(inputShape, sd.constant(Nd4j.scalar(0)), 0);   // INT64 scalar: 1
        SDVariable hiddenDim = sd.gather(inputShape, sd.constant(Nd4j.scalar(2)), 0);  // INT64 scalar: 64

        // concat produces [1, 64] — THIS is the multi-consumer buffer
        SDVariable targetShape = sd.concat(0, batchDim, hiddenDim);  // INT64 [2]

        // Consumer 1: reshape_no_copy uses targetShape as shape parameter
        SDVariable squeezed = sd.squeeze(input, 1);
        SDVariable reshaped = sd.reshape("reshaped", squeezed, targetShape);

        // Consumer 2: cast + elementwise on the shape (forces the shape buffer to be
        // read by multiple ops — in-place fusion on this chain must not corrupt Consumer 1)
        SDVariable shapeFloat = targetShape.castTo(DataType.FLOAT);
        SDVariable shapeScaled = sd.math.mul("shape_scaled", shapeFloat, sd.constant(Nd4j.scalar(2.0f)));
        // Sum the shape-scaled vector to a scalar so it broadcasts with reshaped [1,64]
        SDVariable shapeScalar = sd.math.sum("shape_scalar", shapeScaled);

        // Final output combines both paths: [1,64] + scalar
        SDVariable sum = sd.math.add("output", reshaped, shapeScalar);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, 64);

        // Run multiple times to trigger frozen constant detection + in-place fusion
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> out = sd.output(
                    Map.of("input", inputArr),
                    "output");

            INDArray result = out.get("output");
            assertNotNull(result, "Step " + step + ": output is null");
            assertArrayEquals(new long[]{1, 64}, result.shape(),
                    "Step " + step + ": wrong output shape — reshape_no_copy got corrupted shape");

            // Verify: output = reshaped + sum([2.0, 128.0]) = reshaped + 130.0
            // First element should be input[0,0,0] + 130.0
            float inputFirst = inputArr.getFloat(0, 0, 0);
            float expected = inputFirst + 130.0f;  // sum(1*2 + 64*2) = 2+128 = 130
            assertEquals(expected, result.getFloat(0), 1e-4f,
                    "Step " + step + ": value corruption detected");
        }
    }

    /**
     * Test 3: Graph invalidation on value-key mismatch.
     *
     * Verifies that when a segment's value-key changes (position_id increments),
     * the segment properly invalidates and re-captures WITHOUT crashing.
     * This tests the graph invalidation path directly.
     */
    @Test
    @DisplayName("segment invalidation on value-key change must not crash")
    public void testSegmentInvalidationOnValueKeyChange() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Simulate a mini transformer layer that uses position_id for RoPE-like computation
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 4, 16);  // [batch, heads, dim]
        SDVariable positionId = sd.placeHolder("position_id", DataType.INT64, 1);

        // range(0, position_id, 1) — produces different-length output each step
        // In decode, position_id is constant (1 element), so range always produces same shape
        // but different VALUES
        SDVariable start = sd.constant(Nd4j.scalar(DataType.INT64, 0));
        SDVariable step = sd.constant(Nd4j.scalar(DataType.INT64, 1));

        // Use position_id in computation — value changes each decode step
        SDVariable posFloat = positionId.castTo(DataType.FLOAT);
        SDVariable posScaled = sd.math.mul(posFloat, sd.constant(Nd4j.scalar(0.01f)));

        // Matmul-like computation that follows the position-dependent value
        SDVariable shifted = sd.math.add("output", input, posScaled);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 4, 16);

        // Run 20 steps with incrementing position_id
        // This forces repeated value-key changes → graph invalidation on each step
        for (int i = 0; i < 20; i++) {
            INDArray posArr = Nd4j.scalar(DataType.INT64, i + 1);

            Map<String, INDArray> out = sd.output(
                    Map.of("input", inputArr, "position_id", posArr),
                    "output");

            INDArray result = out.get("output");
            assertNotNull(result, "Step " + i + ": output is null");
            assertArrayEquals(new long[]{1, 4, 16}, result.shape(),
                    "Step " + i + ": wrong shape after invalidation");

            // Verify: output = input + (position_id * 0.01)
            float expectedOffset = (i + 1) * 0.01f;
            float inputVal = inputArr.getFloat(0, 0, 0);
            assertEquals(inputVal + expectedOffset, result.getFloat(0), 1e-4f,
                    "Step " + i + ": incorrect value after graph invalidation/re-capture");
        }
    }

    /**
     * Test 4: The exact VLM crash pattern — concat feeding reshape_no_copy.
     *
     * Reproduces the slot 482 crash: shape_of → add → concat → reshape_no_copy
     * where the concat output (INT64 shape tensor) gets corrupted by in-place
     * fusion from a FLOAT32 elementwise chain sharing the same buffer.
     */
    @Test
    @DisplayName("VLM decode pattern: shape_of → concat → reshape_no_copy × 5 layers")
    public void testVlmDecodeShapeChainMultiLayer() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int numLayers = 5;
        int hiddenSize = 64;
        int numHeads = 4;
        int headDim = hiddenSize / numHeads;

        // Shared input for all layers
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 1, hiddenSize);

        // Build multiple layers that all use the same shape chain pattern
        SDVariable layerInput = input;
        for (int layer = 0; layer < numLayers; layer++) {
            // Shape chain (same pattern as VLM decode)
            SDVariable shape = sd.shape(layerInput);
            SDVariable batch = sd.gather(shape, sd.constant(Nd4j.scalar(0)), 0);
            SDVariable seq = sd.gather(shape, sd.constant(Nd4j.scalar(1)), 0);

            // concat([batch, seq, numHeads, headDim]) for attention reshape
            SDVariable heads = sd.constant("heads_" + layer, Nd4j.scalar(DataType.INT64, numHeads));
            SDVariable dim = sd.constant("dim_" + layer, Nd4j.scalar(DataType.INT64, headDim));
            SDVariable attnShape = sd.concat(0, batch, seq, heads, dim);

            // reshape for multi-head attention: [1,1,64] -> [1,1,4,16]
            SDVariable attnInput = sd.reshape("attn_reshape_" + layer, layerInput, attnShape);

            // Simulate attention output — elementwise (preserves shape, no reduction)
            SDVariable attnOut = sd.nn.relu("attn_act_" + layer, attnInput, 0);
            // Reshape back: [1,1,4,16] -> [1,1,64] (element count preserved: 64=64)
            SDVariable outShape = sd.concat(0, batch, seq,
                    sd.constant("hidden_" + layer, Nd4j.scalar(DataType.INT64, hiddenSize)));
            layerInput = sd.reshape("layer_out_" + layer, attnOut, outShape);

            // Add elementwise ops that could trigger in-place fusion
            SDVariable weight = sd.constant("w_" + layer, Nd4j.randn(DataType.FLOAT, 1, 1, hiddenSize));
            layerInput = sd.math.add("add_" + layer, layerInput, weight);
            layerInput = sd.nn.relu("relu_" + layer, layerInput, 0);
        }

        sd.setOutputs("relu_" + (numLayers - 1));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 1, hiddenSize);

        // Run 10 steps — frozen constants kick in after step 2
        for (int step = 0; step < 10; step++) {
            Map<String, INDArray> out = sd.output(
                    Map.of("input", inputArr),
                    "relu_" + (numLayers - 1));

            INDArray result = out.get("relu_" + (numLayers - 1));
            assertNotNull(result, "Step " + step + ": output is null — likely reshape crash");
            assertArrayEquals(new long[]{1, 1, hiddenSize}, result.shape(),
                    "Step " + step + ": wrong shape — concat→reshape chain corrupted");

            // Verify no NaN/Inf
            assertFalse(Double.isNaN(result.sumNumber().doubleValue()),
                    "Step " + step + ": NaN detected in output");
            assertFalse(Double.isInfinite(result.sumNumber().doubleValue()),
                    "Step " + step + ": Inf detected in output");
        }
    }

    /**
     * Reproduces the LFM2 last-real-token path while alternating between the
     * prefill and decode plans. The slice begin/size vectors are plan-internal
     * GPU-produced controls; replay must refresh their host values before shape
     * inference instead of reading the previous plan's stale primary buffer.
     */
    @Test
    @DisplayName("LFM2 sizeAt → stack → slice controls survive prefill/decode plan reuse")
    public void testInternalSliceControlsAcrossPlanSwitches() {
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int hiddenSize = 16;
        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, -1, -1, hiddenSize);
        SDVariable actualLength = sd.placeHolder("actual_sequence_length", DataType.INT64);
        SDVariable transformed = hidden.mul(2.0).add(1.0);

        SDVariable batchSize = sd.sizeAt(transformed, 0);
        SDVariable hiddenDim = sd.sizeAt(transformed, 2);
        SDVariable zero = sd.constant(Nd4j.scalar(DataType.INT64, 0L));
        SDVariable one = sd.constant(Nd4j.scalar(DataType.INT64, 1L));
        SDVariable actualLast = actualLength.sub(one);
        SDVariable begin = sd.stack("last_begin", 0, zero, actualLast, zero);
        SDVariable size = sd.stack("last_size", 0, batchSize, one, hiddenDim);
        SDVariable hiddenLast = sd.slice("hidden_last", transformed, begin, size);
        SDVariable output = hiddenLast.mul("output", 3.0);

        int[] sequenceLengths = {8, 1, 8, 1, 8, 1, 8};
        int[] actualLengths = {5, 1, 6, 1, 7, 1, 4};
        for (int execution = 0; execution < sequenceLengths.length; execution++) {
            int sequenceLength = sequenceLengths[execution];
            int realLength = actualLengths[execution];
            INDArray hiddenArray = Nd4j.arange(
                            1L * sequenceLength * hiddenSize)
                    .castTo(DataType.FLOAT)
                    .reshape(1, sequenceLength, hiddenSize);

            INDArray result = sd.outputSingle(
                    Map.of(
                            "hidden", hiddenArray,
                            "actual_sequence_length", Nd4j.scalar(DataType.INT64, realLength)),
                    "output");

            assertArrayEquals(new long[]{1, 1, hiddenSize}, result.shape(),
                    "Execution " + execution + ": slice size control was corrupted");
            INDArray expected = hiddenArray.get(
                            NDArrayIndex.all(),
                            NDArrayIndex.interval(realLength - 1, realLength),
                            NDArrayIndex.all())
                    .mul(2.0).add(1.0).mul(3.0);
            assertEquals(expected, result,
                    "Execution " + execution + ": slice begin control selected the wrong token");
        }
    }
}
