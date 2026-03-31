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
import org.junit.jupiter.api.Timeout;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Slf4j
public class TestDSPReplayDeadlockDetection extends BaseNd4jTestWithBackends {

    private static final int BATCH = 1;
    private static final int HEADS = 12;
    private static final int HEAD_DIM = 64;
    private static final double STABILITY_TOLERANCE = 1e-2;

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @Timeout(45)
    @DisplayName("DSP introspection exposes the 4-op attention replay segment")
    public void testCrossSegmentAttentionSegmentStructure() {
        assumeTrue(isCudaBackend(), "Replay deadlock regression requires CUDA backend");
        NativeOps nativeOps = Nd4j.getNativeOps();
        assumeTrue(nativeOps.isTritonAvailable(), "Replay deadlock regression requires Triton");

        SameDiff sd = buildCrossSegmentAttentionGraph(128);
        Environment env = Nd4j.getEnvironment();
        boolean prevGraphCapture = env.tritonGraphCapture();
        int prevCaptureMinExec = env.tritonCaptureMinExec();

        try {
            configureFrozenTritonReplay(sd);
            env.setTritonGraphCapture(true);
            env.setTritonCaptureMinExec(1);

            Map<String, INDArray> feed = makeInputFeed(128);

            sd.outputDirect(feed, "output");
            sd.outputDirect(feed, "output");

            GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                    List.of("output"), GraphExecutionMode.TRITON, true);
            assumeTrue(mode == GraphExecutionMode.TRITON,
                    "Expected TRITON execution mode but got " + mode);

            for (int step = 0; step < 3; step++) {
                Map<String, INDArray> result = sd.outputDirect(feed, "output");
                assertNotNull(result.get("output"), "Replay step " + step + " output is null");
            }

            PlanIntrospection.SegmentInfo attentionSegment = getAttentionSegment(sd);
            assertNotNull(attentionSegment, "Attention replay segment was not found");
            int attentionStart = findAttentionSubsequenceStart(attentionSegment.getOpNames());
            assertTrue(attentionStart >= 0,
                    "Attention replay segment should contain matmul -> mul -> softmax -> matmul: "
                            + attentionSegment.getOpNames());
            assertTrue(attentionSegment.isCapturable(), "Attention replay segment should be capturable");
            assertFalse(attentionSegment.isCaptureFailed(),
                    "Attention replay segment should not be marked failed: "
                            + attentionSegment.getStatisticsJson());
            assertEquals(List.of("matmul", "mul", "softmax", "matmul"),
                    attentionSegment.getOpNames().subList(attentionStart, attentionStart + 4),
                    "Attention replay segment should expose the expected attention op chain");
            assertNotEquals("NO_HANDLE", attentionSegment.getReplayStateName(),
                    "Attention replay segment should have replay metadata");
            assertFalse(attentionSegment.getReplayBackendName().isEmpty(),
                    "Attention replay segment should expose replay backend");
        } finally {
            env.setTritonCaptureMinExec(prevCaptureMinExec);
            env.setTritonGraphCapture(prevGraphCapture);
            sd.close();
        }
    }

    @Test
    @Timeout(60)
    @DisplayName("Frozen cross-segment attention replay stays live across repeated executions")
    public void testCrossSegmentAttentionReplayDoesNotDeadlock() {
        assumeTrue(isCudaBackend(), "Replay deadlock regression requires CUDA backend");
        NativeOps nativeOps = Nd4j.getNativeOps();
        assumeTrue(nativeOps.isTritonAvailable(), "Replay deadlock regression requires Triton");

        SameDiff sd = buildCrossSegmentAttentionGraph(1024);
        Environment env = Nd4j.getEnvironment();
        boolean prevGraphCapture = env.tritonGraphCapture();
        int prevCaptureMinExec = env.tritonCaptureMinExec();

        try {
            configureFrozenTritonReplay(sd);
            env.setTritonGraphCapture(true);
            env.setTritonCaptureMinExec(1);

            Map<String, INDArray> feed = makeInputFeed(1024);

            sd.outputDirect(feed, "output");
            sd.outputDirect(feed, "output");

            GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                    List.of("output"), GraphExecutionMode.TRITON, true);
            assumeTrue(mode == GraphExecutionMode.TRITON,
                    "Expected TRITON execution mode but got " + mode);

            INDArray firstOutput = null;
            for (int step = 0; step < 5; step++) {
                log.info("Cross-segment attention replay step {}", step);
                Map<String, INDArray> result = sd.outputDirect(feed, "output");
                INDArray actual = result.get("output");
                assertNotNull(actual, "Replay step " + step + " output is null");
                assertFalse(actual.isNaN().any(), "Replay step " + step + " produced NaN values");
                assertFalse(actual.isInfinite().any(), "Replay step " + step + " produced Inf values");

                if (firstOutput == null) {
                    firstOutput = actual.dup();
                } else {
                    double maxDiff = firstOutput.dup().sub(actual).amaxNumber().doubleValue();
                    assertTrue(maxDiff < STABILITY_TOLERANCE,
                            "Replay step " + step + " diverged from the first frozen replay, maxDiff="
                                    + maxDiff);
                }
            }

            PlanIntrospection.SegmentInfo attentionSegment = getAttentionSegment(sd);
            assertNotNull(attentionSegment, "Attention replay segment was not found");
            assertFalse(attentionSegment.isCaptureFailed(),
                    "Attention replay segment should remain replayable after repeated frozen executions: "
                            + attentionSegment.getStatisticsJson());
            assertTrue(attentionSegment.getReplayCount() > 0,
                    "Attention replay segment should record at least one replay");
        } finally {
            env.setTritonCaptureMinExec(prevCaptureMinExec);
            env.setTritonGraphCapture(prevGraphCapture);
            sd.close();
        }
    }

    private SameDiff buildCrossSegmentAttentionGraph(int seqLen) {
        SameDiff sd = SameDiff.create();
        int hidden = HEADS * HEAD_DIM;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, BATCH, seqLen, hidden);

        SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, hidden).mul(0.02));
        SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, hidden).mul(0.02));
        SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, hidden).mul(0.02));

        SDVariable qFlat = input.mul("q_proj", wq);
        SDVariable kFlat = input.mul("k_proj", wk);
        SDVariable vFlat = input.mul("v_proj", wv);

        SDVariable qReshaped = sd.reshape("q_reshape", qFlat, BATCH, seqLen, HEADS, HEAD_DIM);
        SDVariable qMultiHead = sd.permute("q_mh", qReshaped, 0, 2, 1, 3);

        SDVariable kReshaped = sd.reshape("k_reshape", kFlat, BATCH, seqLen, HEADS, HEAD_DIM);
        SDVariable kMultiHead = sd.permute("k_mh", kReshaped, 0, 2, 3, 1);

        SDVariable vReshaped = sd.reshape("v_reshape", vFlat, BATCH, seqLen, HEADS, HEAD_DIM);
        SDVariable vMultiHead = sd.permute("v_mh", vReshaped, 0, 2, 1, 3);

        SDVariable qk = sd.linalg().matmul("qk", qMultiHead, kMultiHead);
        SDVariable scaled = qk.mul("scaled", 1.0 / Math.sqrt(HEAD_DIM));
        SDVariable attn = sd.nn().softmax("attn", scaled, -1);
        sd.linalg().matmul("output", attn, vMultiHead);

        return sd;
    }

    private void configureFrozenTritonReplay(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.TRITON);
    }

    private Map<String, INDArray> makeInputFeed(int seqLen) {
        int hidden = HEADS * HEAD_DIM;
        return Collections.singletonMap("input",
                Nd4j.randn(DataType.FLOAT, BATCH, seqLen, hidden));
    }

    private PlanIntrospection.SegmentInfo getAttentionSegment(SameDiff sd) {
        DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        assertNotNull(executor, "DynamicShapePlanExecutor should be available");
        assertNotNull(executor.getCurrentPlan(), "Current plan should be available");
        assertNotNull(executor.getNativePlanHandle(), "Native plan handle should be available");

        List<PlanIntrospection.SegmentInfo> segments = PlanIntrospection.getSegmentsWithReplayState(
                executor.getCurrentPlan(), executor.getNativePlanHandle());
        Optional<PlanIntrospection.SegmentInfo> attentionSegment = segments.stream()
                .filter(seg -> findAttentionSubsequenceStart(seg.getOpNames()) >= 0)
                .findFirst();

        if (attentionSegment.isEmpty()) {
            String summary = PlanIntrospection.formatReplaySummary(executor.getNativePlanHandle());
            fail("Attention replay segment was not found. Replay summary:\n" + summary);
        }

        return attentionSegment.get();
    }

    private int findAttentionSubsequenceStart(List<String> opNames) {
        List<String> attentionOps = List.of("matmul", "mul", "softmax", "matmul");
        if (opNames.size() < attentionOps.size()) {
            return -1;
        }

        for (int i = 0; i <= opNames.size() - attentionOps.size(); i++) {
            if (opNames.subList(i, i + attentionOps.size()).equals(attentionOps)) {
                return i;
            }
        }

        return -1;
    }

    private boolean isCudaBackend() {
        return Nd4j.getBackend().getClass().getSimpleName().contains("Cuda");
    }
}
