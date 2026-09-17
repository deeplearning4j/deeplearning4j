/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Bounded native AUTO replay gate: arithmetic alone cannot prove alias publication. */
class DspTritonAliasPublicationTest {
    private static final Pattern ALIASED_CONSOLIDATION = Pattern.compile(
            "TRITON_CONSOLIDATED_SUBMISSION:[^\"\\n]*aliases=[1-9][0-9]*");

    @Test
    void requestedViewDeliveryDoesNotReplaceCapturedProducer() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA with Triton");
        var environment = Nd4j.getEnvironment();
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean capture = environment.tritonGraphCapture();
        boolean compileAll = environment.tritonCompileAll();
        List<INDArray> delivered = new ArrayList<>();
        List<INDArray> callerCopies = new ArrayList<>();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            environment.setTritonGraphCapture(true);
            environment.setTritonCompileAll(true);
            try (SameDiff sd = SameDiff.create();
                 INDArray input = Nd4j.create(DataType.FLOAT, 4, 16)) {
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
                // A requested transpose is also consumed inside the replay plan.
                // Delivery must gather its strides without replacing that binding.
                SDVariable view = sd.permute("requestedView", x.add(1.0), 1, 0);
                view.mul("out", 2.0);
                long replayPlan = 0;
                for (int step = 0; step < 32; step++) {
                    for (int row = 0; row < 4; row++) {
                        for (int col = 0; col < 16; col++) {
                            input.putScalar(new long[]{row, col}, row * 16 + col + step);
                        }
                    }
                    Map<String, INDArray> outputs = sd.output(
                            Map.of("x", input), "requestedView", "out");
                    // Retain the actual caller result, not another defensive dup.
                    delivered.add(outputs.get("requestedView"));
                    try (INDArray out = outputs.get("out")) {
                        for (int row = 0; row < 16; row++) {
                            for (int col = 0; col < 4; col++) {
                                assertEquals(2.0 * (col * 16 + row + step + 1),
                                        out.getDouble(row, col), 0.0);
                            }
                        }
                    }
                    // Closing the sibling output and executing the plan must not
                    // change the borrowed input or any earlier delivered view.
                    for (int row = 0; row < 4; row++) {
                        for (int col = 0; col < 16; col++) {
                            assertEquals(row * 16 + col + step, input.getDouble(row, col), 0.0,
                                    "input changed at step=" + step);
                        }
                    }
                    assertRequestedViewValues(delivered);
                    if (step == 15) {
                        DspPlanAssertions.assertFullyReplaying(sd, "requested view must reach replay");
                        replayPlan = DspPlanAssertions.getPlanHandleForQuery(sd).address();
                    }
                }
                DspPlanAssertions.assertTotalGraphReplaysAtLeast(sd, 1, "requested-view replay required");
                DspPlanAssertions.assertFullyReplaying(sd, "delivery must not invalidate capture");
                assertEquals(replayPlan, DspPlanAssertions.getPlanHandleForQuery(sd).address());
                // These are independent delivery buffers, not exclusive caller ownership:
                // InferenceSession.output retains the same SDValues in latestRequestedOutputs;
                // closePooledResources -> closeLatestRequestedOutputBuffers closes their
                // DataBuffers during SameDiff.close/resetSession. SameDiff.output only unwraps
                // them (no extra dup), despite the executor's "duped by caller" comment.
                // Check actual retained results before that documented cleanup boundary.
                assertRequestedViewValues(delivered);
                // Explicit caller-owned copies have a separate, longer lifetime contract.
                // Copy only after testing every actual result across all 32 deliveries.
                for (INDArray result : delivered) callerCopies.add(result.dup());
            }
            assertEquals(32, callerCopies.size());
            assertRequestedViewValues(callerCopies);
        } finally {
            for (INDArray result : delivered) result.close();
            for (INDArray result : callerCopies) result.close();
            environment.setTritonCompileAll(compileAll);
            environment.setTritonGraphCapture(capture);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    private static void assertRequestedViewValues(List<INDArray> results) {
        for (int step = 0; step < results.size(); step++) {
            for (int row = 0; row < 16; row++) {
                for (int col = 0; col < 4; col++) {
                    assertEquals(col * 16 + row + step + 1.0,
                            results.get(step).getDouble(row, col), 0.0,
                            "delivery=" + step + " row=" + row + " col=" + col);
                }
            }
        }
    }

    @Test
    void aliasConsolidationSurvivesStableReplayAndRebinding() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA with Triton");
        var environment = Nd4j.getEnvironment();
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean capture = environment.tritonGraphCapture();
        boolean compileAll = environment.tritonCompileAll();
        boolean consolidated = environment.tritonConsolidatedArgTable();
        int captureMinExec = environment.tritonCaptureMinExec();
        int mask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
        int level = Nd4j.getNativeOps().dspDiagGetLevel();
        List<INDArray> snapshots = new ArrayList<>();
        List<Double> expected = new ArrayList<>();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            environment.setTritonGraphCapture(true);
            environment.setTritonCompileAll(true);
            environment.setTritonConsolidatedArgTable(true);
            // Exercise compiled direct submission before AUTO naturally captures.
            // This finite window leaves ample iterations for capture and replay.
            environment.setTritonCaptureMinExec(4);
            DspDiagnostics.setCategories(DspDiagnostics.VERIFY);
            DspDiagnostics.setLevel(DspDiagnostics.LEVEL_FULL);
            DspDiagnostics.clear();
            try (SameDiff sd = SameDiff.create();
                 INDArray input = Nd4j.ones(DataType.FLOAT, 4, 16)) {
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
                SDVariable state = sd.var("state", Nd4j.ones(DataType.FLOAT, 4, 16));
                SDVariable stateAlias = sd.identity("stateAlias", state);
                // identity publishes its input wrapper in native warmup. Request
                // stateAlias as an output so Triton must publish that actual alias,
                // not merely inline the identity's value into its consumer.
                // Keep this segment entirely Triton-lowerable: native matmul/view
                // gaps require ordered capture immediately and bypass the direct
                // consolidated H2D branch, even with a capture warmup window.
                SDVariable first = sd.nn.relu("firstRelu",
                        x.add(stateAlias).mul(16.0).add(1.0), 0);
                SDVariable second = first.mul(8.0).add(1.0);
                sd.nn.relu("secondRelu", second, 0).mul("out", 2.0);
                assertEquals(GraphExecutionMode.AUTO, sd.getGraphExecutionMode());
                boolean aliasPrepared = false;
                boolean aliasConsolidated = false;
                boolean capturedSubmission = false;
                long planAddress = 0;
                for (int phase = 0; phase < 2; phase++) {
                    double biasValue = phase == 0 ? 1.0 : 3.0;
                    if (phase == 1) {
                        // New storage, same shape; do not clear the plan or force a mode.
                        sd.associateArrayWithVariable(
                                Nd4j.valueArrayOf(new long[]{4, 16}, biasValue, DataType.FLOAT), state);
                    }
                    for (int step = 0; step < 16; step++) {
                        input.assign(step + 1.0);
                        Map<String, INDArray> outputs = sd.output(Map.of("x", input), "out", "stateAlias");
                        snapshots.add(outputs.get("out").dup());
                        expected.add(256.0 * (step + 1 + biasValue) + 18.0);
                        snapshots.add(outputs.get("stateAlias").dup());
                        expected.add(biasValue);
                        String report = DspDiagnostics.getJsonReport();
                        aliasPrepared |= report.contains("TRITON_ALIAS_PREPARED:");
                        aliasConsolidated |= ALIASED_CONSOLIDATION.matcher(report).find();
                        capturedSubmission |= report.contains("TRITON_CAPTURED_ARG_SUBMISSION:");
                    }
                    // Read only after the bounded submission batch, not between calls.
                    for (int i = 0; i < snapshots.size(); i++) {
                        INDArray result = snapshots.get(i);
                        for (int j = 0; j < result.length(); j++) {
                            assertEquals(expected.get(i), result.getDouble(j), 0.0,
                                    "phase=" + phase + " submission=" + i + " element=" + j);
                        }
                    }
                    for (INDArray result : snapshots) result.close();
                    snapshots.clear();
                    expected.clear();
                    DspPlanAssertions.assertTotalGraphReplaysAtLeast(sd, 1, "natural AUTO replay required");
                    DspPlanAssertions.assertFullyReplaying(sd, "must settle after binding publication");
                    long currentPlan = DspPlanAssertions.getPlanHandleForQuery(sd).address();
                    if (phase == 0) planAddress = currentPlan;
                    else assertEquals(planAddress, currentPlan, "same-shape rebind must reuse the plan");
                }
                assertTrue(aliasPrepared, "fixture must exercise a real input/output alias");
                assertTrue(aliasConsolidated, "an aliased scratch binding must be submitted via consolidation");
                assertTrue(capturedSubmission, "replay owner must record captured H2D source consumption");
                assertEquals(GraphExecutionMode.AUTO, sd.getGraphExecutionMode());
            }
        } finally {
            for (INDArray snapshot : snapshots) snapshot.close();
            DspDiagnostics.clear();
            DspDiagnostics.setCategories(mask);
            DspDiagnostics.setLevel(level);
            environment.setTritonCaptureMinExec(captureMinExec);
            environment.setTritonConsolidatedArgTable(consolidated);
            environment.setTritonCompileAll(compileAll);
            environment.setTritonGraphCapture(capture);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }
}
