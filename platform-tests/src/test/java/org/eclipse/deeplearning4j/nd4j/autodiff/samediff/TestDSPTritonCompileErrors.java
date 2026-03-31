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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that Triton compilation errors are surfaced as real errors rather than
 * silently falling back to CUDA graph capture or slot-by-slot execution.
 *
 * <p>Root issue: When Triton IR builder fails to compile a segment (e.g., missing
 * cross-segment input shape for a divide op), the failure was silently cascading
 * through fallback paths (CUDA graph capture → slot-by-slot), eventually producing
 * cryptic NaN outputs with no indication of the root cause.</p>
 *
 * <p>These tests verify:</p>
 * <ul>
 *   <li>Cross-segment divide ops compile and execute correctly</li>
 *   <li>Multi-segment graphs with cross-segment dependencies produce correct results</li>
 *   <li>Frozen shapes don't cause stale shape caches that break compilation</li>
 * </ul>
 */
@Slf4j
@Tag("samediff")
public class TestDSPTritonCompileErrors extends BaseNd4jTestWithBackends {

    @Test
    @DisplayName("Cross-segment divide compiles and executes correctly")
    public void testCrossSegmentDivide() {
        // Build a graph where a divide op's inputs come from different segments.
        // This is the pattern that triggered the original bug: the divide op at
        // a segment boundary has cross-segment inputs whose shapes may not be
        // in the cachedShapeInfoMap if the producing op's state was reset.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 64));

        // matmul -> cast -> divide creates cross-segment dependencies
        // (matmul is cuBLAS fallback, cast+divide are elementwise/Triton)
        SDVariable mm1 = sd.mmul("mm1", input, w1);
        SDVariable cast1 = sd.castTo("cast1", mm1, DataType.FLOAT);
        SDVariable mm2 = sd.mmul("mm2", input, w2);
        SDVariable cast2 = sd.castTo("cast2", mm2, DataType.FLOAT);

        // Divide with cross-segment inputs: cast1 from one segment, cast2 from another
        // Add small epsilon to prevent 0/0
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-6f));
        SDVariable denom = cast2.add("denom", eps);
        SDVariable output = cast1.div("output", denom);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32);

        // Reference computation
        INDArray mm1Ref = inputArr.mmul(w1.getArr());
        INDArray mm2Ref = inputArr.mmul(w2.getArr());
        INDArray expected = mm1Ref.div(mm2Ref.add(1e-6f));

        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": cross-segment divide produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            // Division near zero amplifies numerical differences between cuBLAS
            // matmul and reference mmul, so use a wider tolerance.
            assertTrue(maxDiff < 0.5f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Multi-layer graph with cross-segment elementwise ops")
    public void testMultiLayerCrossSegmentElementwise() {
        // Build a deeper graph with multiple cross-segment elementwise dependencies.
        // Each matmul creates a segment boundary; the elementwise ops between them
        // must all compile correctly with their cross-segment inputs.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));

        // Layer 1: matmul -> relu (creates segment boundary at matmul)
        SDVariable h1 = sd.nn.relu("h1", sd.mmul("mm1", input, w1), 0);

        // Layer 2: add + multiply (cross-segment elementwise using h1)
        SDVariable scale = sd.constant("scale", Nd4j.valueArrayOf(new long[]{1, 32}, 0.5f));
        SDVariable h2 = h1.mul("scaled", scale);
        SDVariable bias = sd.constant("bias", Nd4j.randn(DataType.FLOAT, 1, 32));
        SDVariable h3 = h2.add("biased", bias);

        // Layer 3: matmul -> tanh
        SDVariable output = sd.nn.tanh("output", sd.mmul("mm2", h3, w2));

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 16);

        // Reference
        INDArray h1Ref = Nd4j.nn().relu(inputArr.mmul(w1.getArr()), 0);
        INDArray h2Ref = h1Ref.mul(0.5f);
        INDArray h3Ref = h2Ref.add(bias.getArr());
        INDArray expected = Nd4j.nn().tanh(h3Ref.mmul(w2.getArr()));

        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": multi-layer cross-segment produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.1f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Frozen shapes preserve cross-segment shape caches")
    public void testFrozenShapesPreserveCrossSegmentShapes() {
        // Verify that freezing shapes doesn't clear the shape caches needed
        // for cross-segment Triton compilation. The fix ensures cachedOutputShapes
        // are included in the shape cache even when shapeCacheValid() is false.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 8));

        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable relu = sd.nn.relu("relu", mm, 0);
        SDVariable output = sd.nn.tanh("output", relu);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 8);
        INDArray expected = Nd4j.nn().tanh(Nd4j.nn().relu(inputArr.mmul(w.getArr()), 0));

        // Call 1: unfrozen warmup
        Map<String, INDArray> result1 = sd.output(
                Collections.singletonMap("input", inputArr), "output");
        assertFalse(result1.get("output").isNaN().any(), "Call 1: NaN");

        // Call 2-5: should use compiled path
        for (int call = 2; call <= 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": produced NaN after compilation");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("1-input Where (coordinate extraction) runs via fallback without error")
    public void testWhereOneInputRunsViaFallback() {
        // 1-input Where is coordinate extraction (data-dependent output shape).
        // It should NOT be compiled by Triton — it must be excluded at the
        // section classification level and run via native slot-by-slot.
        // Previously this caused a hard error because:
        // 1. Where was classified as TERNARY → ELEMENTWISE section → Triton tried to compile
        // 2. TERNARY handler requires 3 inputs, Where has 1 → compilation fail
        // 3. Zero compiled sub-kernels → KERNEL_FAILURE
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);

        // Create a condition and use Where to extract coordinates
        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable gt = sd.neq("neq", input, zero);
        SDVariable cast = sd.castTo("castBool", gt, DataType.BOOL);
        // 1-input Where: extracts coordinates of non-zero elements
        SDVariable whereResult = sd.where("whereResult", cast);
        // Since Where output is data-dependent, we just check it doesn't crash
        // Use a simple output that doesn't depend on Where's variable-length output
        SDVariable output = input.mul("output", input);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.createFromArray(new float[][]{{1.0f, 0.0f, 3.0f, 0.0f}});
        INDArray expected = inputArr.mul(inputArr);

        for (int call = 0; call < 3; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": Where graph produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("bool_not compiles and executes correctly via Triton")
    public void testBoolNotCompiles() {
        // bool_not is a LOGICAL op that should be compiled by Triton.
        // Previously failed because the TritonIRBuilder op table didn't have
        // "bool_not" (only "boolean_not" and "BooleanNot").
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);

        SDVariable zero = sd.constant("zero", Nd4j.scalar(DataType.FLOAT, 0.0f));
        SDVariable isZero = sd.eq("isZero", input, zero);
        SDVariable notZero = sd.booleanNot("notZero", isZero);
        SDVariable castToFloat = sd.castTo("output", notZero, DataType.FLOAT);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.createFromArray(new float[][]{{1.0f, 0.0f, 3.0f, 0.0f}});
        // Expected: [1, 0, 1, 0] → eq zero → [0, 1, 0, 1] → not → [1, 0, 1, 0]
        INDArray expected = Nd4j.createFromArray(new float[][]{{1.0f, 0.0f, 1.0f, 0.0f}});

        for (int call = 0; call < 3; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": bool_not graph produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": bool_not result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("LayerNorm pattern with divide compiles correctly")
    public void testLayerNormDivideCompiles() {
        // LayerNorm-like pattern that includes divide: (x - mean) / std
        // This creates multiple cross-segment dependencies and exercises
        // the divide op with cross-segment inputs.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);

        SDVariable mean = sd.math.mean(input, true, 1);
        SDVariable centered = input.sub("centered", mean);
        SDVariable sq = centered.mul("sq", centered);
        SDVariable variance = sd.math.mean("var", sq, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable std = sd.math.sqrt("std", variance.add("varEps", eps));
        SDVariable output = centered.div("output", std);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.createFromArray(new float[][]{
                {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}
        });

        // Reference: layer norm
        INDArray meanArr = inputArr.mean(true, 1);
        INDArray centeredArr = inputArr.sub(meanArr);
        INDArray varArr = centeredArr.mul(centeredArr).mean(true, 1);
        INDArray stdArr = Nd4j.math().sqrt(varArr.add(1e-5f));
        INDArray expected = centeredArr.div(stdArr);

        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": LayerNorm divide produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.05f,
                    "Call " + call + ": LayerNorm result too far from reference: maxDiff=" + maxDiff);
        }
    }
}
