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
 * Tests that frozen-shapes execution does not produce NaN from stale slotArrayCache_ entries.
 *
 * <p>Root cause: Between calls, cleanup nulls outputSlots_ entries but slotArrayCache_ retains
 * stale pointers. When the capture path is entered (call 3+), POST-WARMUP restore code restores
 * null outputSlots_ from slotArrayCache_ without validating the DataBuffer is still alive.
 * Downstream ops then read freed memory (garbage/NaN).</p>
 *
 * <p>The fix adds DataBuffer validity checking to both the POST-WARMUP restore and the
 * cache-fallback input gathering code.</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPFrozenPathRestore extends BaseNd4jTestWithBackends {

    @Test
    @DisplayName("Frozen cross-segment restore should not produce NaN")
    public void testFrozenCrossSegmentRestore() {
        // Build a multi-segment graph: matmul -> relu -> matmul -> output
        // This creates cross-segment dependencies that exercise the cache restore path.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 64, 128));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 128, 32));

        SDVariable mm1 = sd.mmul("mm1", input, w1);
        SDVariable relu = sd.nn.relu("relu", mm1, 0);
        SDVariable mm2 = sd.mmul("mm2", relu, w2);
        SDVariable output = sd.nn.tanh("output", mm2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 64);

        // Reference
        INDArray expected = Nd4j.nn().tanh(
                Nd4j.nn().relu(inputArr.mmul(w1.getArr()), 0).mmul(w2.getArr()));

        // Execute 5 times — call 3+ enters capture path where the bug manifests
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": frozen cross-segment produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Frozen multi-call stability with frozen shapes")
    public void testFrozenMultiCallStability() {
        // Longer pipeline to stress cache restore across many segments
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 64, 16));

        SDVariable h1 = sd.nn.relu("h1", sd.mmul("mm1", input, w1), 0);
        SDVariable h2 = sd.nn.relu("h2", sd.mmul("mm2", h1, w2), 0);
        SDVariable output = sd.nn.tanh("output", sd.mmul("mm3", h2, w3));

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 4, 32);

        // Reference
        INDArray h1Ref = Nd4j.nn().relu(inputArr.mmul(w1.getArr()), 0);
        INDArray h2Ref = Nd4j.nn().relu(h1Ref.mmul(w2.getArr()), 0);
        INDArray expected = Nd4j.nn().tanh(h2Ref.mmul(w3.getArr()));

        // Execute 10 times to ensure stability through capture path transitions
        for (int call = 0; call < 10; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": multi-layer frozen produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Frozen cleanup does not clobber capture output")
    public void testFrozenCleanupDoesNotClobberCaptureOutput() {
        // Graph with intermediate op whose output feeds downstream segment.
        // Tests that between-call cleanup + cache restore doesn't corrupt data.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);

        // LayerNorm-like pattern: mean -> subtract -> pow -> mean -> sqrt -> div
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
                    "Call " + call + ": LayerNorm frozen produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.05f,
                    "Call " + call + ": LayerNorm result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Capture workspace gap op output does not corrupt downstream warmup")
    public void testCaptureWorkspaceGapOpOutput() {
        // This test reproduces the specific bug: when a gap op (softmax) inside a
        // captured segment allocates its output from the capture workspace, and that
        // output feeds a downstream segment that hasn't been captured yet, the
        // downstream segment's warmup reads from the capture workspace instead of
        // the warmup data, producing NaN.
        //
        // Graph structure creates multiple segments:
        //   seg1: matmul -> softmax (gap op, output feeds seg2)
        //   seg2: matmul (reads softmax output from seg1)
        //
        // The softmax output is a gap op output in seg1. During seg1's capture,
        // softmax allocates from capture workspace. Without the fix, seg2's warmup
        // reads the workspace pointer (stale data) instead of the warmup pointer.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 64, 16));

        // matmul -> softmax -> matmul
        // softmax is a gap op that creates a segment boundary
        SDVariable mm1 = sd.mmul("mm1", input, w1);
        SDVariable soft = sd.nn.softmax("soft", mm1, -1);
        SDVariable output = sd.mmul("output", soft, w2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32);

        // Reference: matmul -> softmax -> matmul
        INDArray mm1Ref = inputArr.mmul(w1.getArr());
        INDArray softRef = Nd4j.nn().softmax(mm1Ref, -1);
        INDArray expected = softRef.mmul(w2.getArr());

        // Execute 5 times — bug manifests at call 3+ when capture kicks in
        for (int call = 0; call < 5; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": capture workspace gap op produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.1f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Warmup arrays survive flushPendingClose during capture")
    public void testWarmupArraysSurviveFlushPendingClose() {
        // This test targets the specific bug where warmup arrays get pushed to
        // pendingClose_ during capture execution (when capture workspace arrays
        // replace warmup arrays in slotArrayCache_), then flushPendingClose closes
        // them, leaving outputSlots_ with dead pointers for downstream segments.
        //
        // The graph creates multiple operations that produce intermediate results
        // consumed by later segments. During capture of early segments, those
        // intermediate warmup arrays must survive flushPendingClose so downstream
        // segments can read them during their own warmup/capture.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);

        // Deep pipeline with many intermediate outputs
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 32, 64));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 64, 64));
        SDVariable w3 = sd.constant("w3", Nd4j.randn(DataType.FLOAT, 64, 32));
        SDVariable w4 = sd.constant("w4", Nd4j.randn(DataType.FLOAT, 32, 16));

        // Layer 1: matmul -> relu
        SDVariable h1 = sd.nn.relu("h1", sd.mmul("mm1", input, w1), 0);
        // Layer 2: matmul -> sigmoid (gap op - creates segment boundary)
        SDVariable h2 = sd.nn.sigmoid("h2", sd.mmul("mm2", h1, w2));
        // Layer 3: matmul -> relu
        SDVariable h3 = sd.nn.relu("h3", sd.mmul("mm3", h2, w3), 0);
        // Layer 4: matmul -> tanh
        SDVariable output = sd.nn.tanh("output", sd.mmul("mm4", h3, w4));

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 32);

        // Compute reference result
        INDArray h1Ref = Nd4j.nn().relu(inputArr.mmul(w1.getArr()), 0);
        INDArray h2Ref = Nd4j.nn().sigmoid(h1Ref.mmul(w2.getArr()));
        INDArray h3Ref = Nd4j.nn().relu(h2Ref.mmul(w3.getArr()), 0);
        INDArray expected = Nd4j.nn().tanh(h3Ref.mmul(w4.getArr()));

        // Execute many times. The warmup-array-closed-by-pendingClose bug
        // manifests at call 3+ when capture kicks in and flushPendingClose
        // runs after each segment's capture.
        for (int call = 0; call < 8; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": deep pipeline frozen produced NaN "
                    + "(warmup array likely closed by flushPendingClose)");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.05f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);
        }
    }

    @Test
    @DisplayName("Frozen replay handle persistence across calls")
    public void testFrozenReplayHandlePersistence() {
        // Simple graph — verify output is consistent across many frozen calls
        // (replay handles should persist and produce identical results)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 8));
        SDVariable output = sd.nn.relu("output", sd.mmul("mm", input, w), 0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 8);
        INDArray expected = Nd4j.nn().relu(inputArr.mmul(w.getArr()), 0);

        INDArray firstResult = null;
        for (int call = 0; call < 8; call++) {
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": simple frozen produced NaN");

            float maxDiff = Transforms.abs(actual.sub(expected)).maxNumber().floatValue();
            assertTrue(maxDiff < 0.01f,
                    "Call " + call + ": result too far from reference: maxDiff=" + maxDiff);

            if (firstResult == null) {
                firstResult = actual.dup();
            } else {
                // All frozen calls should produce identical output
                float callDiff = Transforms.abs(actual.sub(firstResult)).maxNumber().floatValue();
                assertTrue(callDiff < 1e-6f,
                        "Call " + call + ": output differs from call 0: maxDiff=" + callDiff);
            }
        }
    }
}
