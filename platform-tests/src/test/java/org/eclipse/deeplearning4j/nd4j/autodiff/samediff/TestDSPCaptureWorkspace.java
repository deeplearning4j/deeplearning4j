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
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that CUDA graph capture workspace exhaustion is handled gracefully.
 *
 * <p>When a segment's gap ops (concat, split, etc.) need more temporary memory than the
 * pre-allocated capture workspace, the allocator must NOT fall through to cudaMallocAsync
 * (which corrupts capture with error 901). Instead, it should return nullptr, causing the
 * segment to abort capture and fall back to slot-by-slot execution.</p>
 *
 * <p>This test creates graphs with large intermediate tensors to stress the workspace,
 * verifying that results are correct regardless of whether capture succeeds or falls back.</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPCaptureWorkspace extends BaseNd4jTestWithBackends {

    @Test
    @DisplayName("Large concat should not corrupt capture")
    public void testLargeConcatDoesNotCorruptCapture() {
        // Build a graph with multiple matmuls feeding a concat — the concat's temporary
        // buffer may exceed the capture workspace, forcing graceful fallback.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 256);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 256, 512));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 256, 512));
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 256, 512));

        SDVariable m1 = sd.mmul("m1", input, w1);
        SDVariable m2 = sd.mmul("m2", input, w2);
        SDVariable m3 = sd.mmul("m3", input, w3);

        // Concat 3 x [batch, 512] -> [batch, 1536]
        SDVariable concat = sd.concat("concat", 1, m1, m2, m3);
        SDVariable output = sd.nn.relu("output", concat, 0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 256);

        // Reference
        INDArray m1Ref = inputArr.mmul(w1.getArr());
        INDArray m2Ref = inputArr.mmul(w2.getArr());
        INDArray m3Ref = inputArr.mmul(w3.getArr());
        INDArray concatRef = Nd4j.concat(1, m1Ref, m2Ref, m3Ref);
        INDArray expectedOutput = Nd4j.nn().relu(concatRef, 0);

        // Execute 3 times (warmup + capture attempt + replay/fallback)
        for (int call = 0; call < 3; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": concat path produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expectedOutput)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.1f,
                    "Call " + call + ": concat result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Multi-call stability with workspace-heavy segments")
    public void testMultiCallStabilityWithHeavySegments() {
        // Verify that repeated execution is stable even when some segments
        // may fail capture and fall back to slot-by-slot.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 128);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 128, 256));
        SDVariable mm = sd.mmul("mm", input, w);
        SDVariable relu = sd.nn.relu("relu", mm, 0);
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 256, 64));
        SDVariable mm2 = sd.mmul("mm2", relu, w2);
        SDVariable output = sd.nn.tanh("output", mm2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 128);

        // Reference
        INDArray expected = Nd4j.nn().tanh(
                Nd4j.nn().relu(inputArr.mmul(w.getArr()), 0).mmul(w2.getArr()));

        // Execute 5 times
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }
}
