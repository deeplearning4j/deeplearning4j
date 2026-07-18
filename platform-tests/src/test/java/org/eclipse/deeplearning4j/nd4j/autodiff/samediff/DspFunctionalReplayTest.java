/*
 * ******************************************************************************
 * *
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * *  See the NOTICE file distributed with this work for additional
 * *  information regarding copyright ownership.
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * * License for the specific language governing permissions and limitations
 * * under the License.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.ExecutionPhase;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("dsp")
class DspFunctionalReplayTest {

    @BeforeEach
    void enableDynamicShapePlan() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    void commitPendingWork() {
        Nd4j.getExecutioner().commit();
    }

    @Test
    void executesPublishedProgramAndRefreshesIdentityAliases() {
        SameDiff sameDiff = SameDiff.create();
        try {
            SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 2, 2);
            SDVariable forwarded = sameDiff.identity("forwarded", input);
            SDVariable scaled = forwarded.mul("scaled", 3.0);
            scaled.add("out", 1.0);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);

            for (int step = 0; step < 6; step++) {
                INDArray currentInput = Nd4j.ones(DataType.FLOAT, 2, 2).muli(step + 1.0);
                Map<String, INDArray> outputs =
                        sameDiff.output(Collections.singletonMap("input", currentInput), "out");
                INDArray actual = outputs.get("out");
                INDArray expected = currentInput.mul(3.0).add(1.0);

                assertNotNull(actual, "step " + step + " produced no output");
                assertTrue(expected.equalsWithEps(actual, 1e-6),
                        "step " + step + " used stale identity input: expected="
                                + expected + ", actual=" + actual);
            }

            DspPlanAssertions.assertCapturedGraphSegmentsAtLeast(
                    sameDiff, 1, "EMULATED_REPLAY must publish a functional program");
            DspPlanAssertions.assertAllCapturableSegmentsReachedPhase(
                    sameDiff, ExecutionPhase.REPLAYING,
                    "EMULATED_REPLAY functional program");
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) > 0,
                    "functional program was captured but never replayed");
            DspPlanAssertions.assertNoCaptureFailures(
                    sameDiff, "EMULATED_REPLAY functional program");
        } finally {
            sameDiff.close();
        }
    }

    @Test
    void portableReplayExecutesWithChangingBindings() {
        SameDiff sameDiff = SameDiff.create();
        try {
            SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 2, 2);
            input.mul(2.0).add("out", 5.0);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.PORTABLE_REPLAY);

            for (int step = 0; step < 6; step++) {
                INDArray currentInput = Nd4j.ones(DataType.FLOAT, 2, 2).muli(step + 1.0);
                INDArray actual = sameDiff.outputSingle(
                        Collections.singletonMap("input", currentInput), "out");
                INDArray expected = currentInput.mul(2.0).add(5.0);

                assertNotNull(actual, "step " + step + " produced no output");
                assertTrue(expected.equalsWithEps(actual, 1e-6),
                        "step " + step + " used a stale portable replay binding: expected="
                                + expected + ", actual=" + actual);
            }
        } finally {
            sameDiff.close();
        }
    }

    @Test
    void portableReplayFallsBackToFunctionalProgramForUnfusedSegment() {
        SameDiff sameDiff = SameDiff.create();
        try {
            SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 2, 2);
            SDVariable forwarded = sameDiff.identity("forwarded", input);
            sameDiff.identity("out", forwarded);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.PORTABLE_REPLAY);

            for (int step = 0; step < 6; step++) {
                INDArray currentInput = Nd4j.ones(DataType.FLOAT, 2, 2).muli(step + 1.0);
                INDArray actual = sameDiff.outputSingle(
                        Collections.singletonMap("input", currentInput), "out");

                assertNotNull(actual, "step " + step + " produced no output");
                assertTrue(currentInput.equalsWithEps(actual, 1e-6),
                        "step " + step + " used a stale portable identity binding");
            }

            assertEquals(1, sameDiff.dsp().numSegments(),
                    "the identity chain must remain one materialized segment");
            DspPlanAssertions.assertSegmentBackend(
                    sameDiff, 0, "FunctionalReplay",
                    "the unfused identity segment must own the functional program");
            DspPlanAssertions.assertCapturedGraphSegmentsAtLeast(
                    sameDiff, 1,
                    "PORTABLE_REPLAY must record segments rejected by CPU graph backends");
            DspPlanAssertions.assertAllCapturableSegmentsReachedPhase(
                    sameDiff, ExecutionPhase.REPLAYING,
                    "PORTABLE_REPLAY functional fallback");
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) > 0,
                    "portable functional fallback was captured but never replayed");
            DspPlanAssertions.assertNoCaptureFailures(
                    sameDiff, "PORTABLE_REPLAY functional fallback");
        } finally {
            sameDiff.close();
        }
    }

    @Test
    void replaysAcrossViewOffsetsAndMultipleOutputBindings() {
        SameDiff sameDiff = SameDiff.create();
        try {
            SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 2, 2);
            input.mul("doubled", 2.0);
            input.mul("tripled", 3.0);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.EMULATED_REPLAY);

            INDArray backing = Nd4j.create(DataType.FLOAT, 4, 2);
            for (int step = 0; step < 6; step++) {
                int rowStart = step % 3;
                INDArray currentInput = backing.get(
                        NDArrayIndex.interval(rowStart, rowStart + 2),
                        NDArrayIndex.all());
                currentInput.assign(step + 1.0);

                assertTrue(currentInput.isView(), "test input must exercise a view binding");
                assertEquals(rowStart * 2L, currentInput.offset(),
                        "test input must change its view offset");

                Map<String, INDArray> outputs = sameDiff.output(
                        Collections.singletonMap("input", currentInput),
                        "doubled", "tripled");
                INDArray actualDoubled = outputs.get("doubled");
                INDArray actualTripled = outputs.get("tripled");

                assertNotNull(actualDoubled, "step " + step + " produced no doubled output");
                assertNotNull(actualTripled, "step " + step + " produced no tripled output");
                assertTrue(currentInput.mul(2.0).equalsWithEps(actualDoubled, 1e-6),
                        "step " + step + " used a stale doubled binding");
                assertTrue(currentInput.mul(3.0).equalsWithEps(actualTripled, 1e-6),
                        "step " + step + " used a stale tripled binding");
            }

            DspPlanAssertions.assertCapturedGraphSegmentsAtLeast(
                    sameDiff, 1, "view-offset program must publish a functional program");
            DspPlanAssertions.assertAllCapturableSegmentsReachedPhase(
                    sameDiff, ExecutionPhase.REPLAYING,
                    "EMULATED_REPLAY view-offset functional program");
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) > 0,
                    "view-offset functional program was captured but never replayed");
            DspPlanAssertions.assertNoCaptureFailures(
                    sameDiff, "EMULATED_REPLAY view-offset functional program");
        } finally {
            sameDiff.close();
        }
    }
}
