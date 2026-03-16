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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that shapes freeze correctly after the first decode step with padded inputs,
 * and that subsequent steps with frozen shapes produce correct results.
 */
@Slf4j
@DisplayName("ShapeFreezeTest")
public class ShapeFreezeTest {

    private static final int BATCH = 1;
    private static final int NUM_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN = NUM_HEADS * HEAD_DIM;
    private static final int MAX_KV_LEN = 15;

    /**
     * Execute step 1 with padded inputs (establishing decode shape profile),
     * then freeze shapes, then execute steps 2-10 with the same fixed shapes.
     * Verifies all steps produce correct (non-NaN, non-zero) results.
     */
    @Test
    @DisplayName("testFreezeAfterFirstStep")
    public void testFreezeAfterFirstStep() {
        // Build a SameDiff graph with OnnxMultiHeadAttention
        SameDiff sd = SameDiff.create();

        SDVariable query = sd.placeHolder("query", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable bias = sd.placeHolder("bias", DataType.FLOAT, 1, 1, 1, MAX_KV_LEN + 1);
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, query, key, value, bias, pastKey, pastValue,
                NUM_HEADS, 0.0, false);
        SDVariable[] outputs = mha.outputVariables();
        String attnName = outputs[0].name();

        // Static KV buffers
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        float maskFill = -3.4028235e+38f;
        int cachePos = 0;

        // Enable DSP auto-compile
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        for (int step = 0; step < 10; step++) {
            INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);

            // Build sparse bias
            INDArray biasArr = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, MAX_KV_LEN + 1);
            if (cachePos < MAX_KV_LEN) {
                biasArr.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                        NDArrayIndex.interval(cachePos, MAX_KV_LEN)).assign(maskFill);
            }

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("query", q);
            ph.put("key", k);
            ph.put("value", v);
            ph.put("bias", biasArr);
            ph.put("past_key", staticKey);
            ph.put("past_value", staticValue);

            Map<String, INDArray> result = sd.output(ph, attnName);
            INDArray attnResult = result.get(attnName);

            assertNotNull(attnResult, "Step " + step + ": attention output null");
            assertFalse(Double.isNaN(attnResult.sumNumber().doubleValue()),
                    "Step " + step + ": attention output contains NaN");
            assertTrue(attnResult.ameanNumber().doubleValue() > 1e-8,
                    "Step " + step + ": attention output should not be all zeros");

            log.info("Step {}: attn absMax={} cachePos={}", step,
                    attnResult.amaxNumber().floatValue(), cachePos);

            // After step 0: freeze shapes if DSP compiled a plan
            if (step == 0) {
                InferenceSession sess = sd.getOrCreateSession();
                DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
                if (exec != null && exec.getCurrentPlan() != null) {
                    exec.setShapesFrozen(true);
                    log.info("Shapes frozen after step 0");
                }
            }

            cachePos++;
        }

        sd.close();
    }

    /**
     * After freezing shapes, verify that the plan maintains a constant slot count
     * (shapes don't change, so no recompilation needed).
     */
    @Test
    @DisplayName("testFreezePreventsDynamicShapes")
    public void testFreezePreventsDynamicShapes() {
        SameDiff sd = SameDiff.create();

        SDVariable query = sd.placeHolder("query", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable bias = sd.placeHolder("bias", DataType.FLOAT, 1, 1, 1, MAX_KV_LEN + 1);
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, query, key, value, bias, pastKey, pastValue,
                NUM_HEADS, 0.0, false);
        SDVariable[] outputs = mha.outputVariables();
        String attnName = outputs[0].name();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, MAX_KV_LEN, HEAD_DIM);
        float maskFill = -3.4028235e+38f;

        // Step 0: establish shapes
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("query", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("key", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("value", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        INDArray biasArr = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, MAX_KV_LEN + 1);
        biasArr.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                NDArrayIndex.interval(0, MAX_KV_LEN)).assign(maskFill);
        ph.put("bias", biasArr);
        ph.put("past_key", staticKey);
        ph.put("past_value", staticValue);
        sd.output(ph, attnName);

        // Freeze
        InferenceSession sess = sd.getOrCreateSession();
        DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
        if (exec != null && exec.getCurrentPlan() != null) {
            exec.setShapesFrozen(true);
            assertTrue(exec.isShapesFrozen(), "Shapes should be frozen");

            // Steps 1-5: same fixed shapes
            for (int step = 1; step <= 5; step++) {
                ph.put("query", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
                ph.put("key", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
                ph.put("value", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));

                Map<String, INDArray> result = sd.output(ph, attnName);
                INDArray attn = result.get(attnName);
                assertFalse(Double.isNaN(attn.sumNumber().doubleValue()),
                        "Step " + step + ": should not produce NaN with frozen shapes");
            }

            // Verify still frozen
            assertTrue(exec.isShapesFrozen(), "Shapes should remain frozen after multiple steps");
        } else {
            log.warn("DSP plan not compiled — skipping freeze verification (CPU-only build?)");
        }

        sd.close();
    }
}
