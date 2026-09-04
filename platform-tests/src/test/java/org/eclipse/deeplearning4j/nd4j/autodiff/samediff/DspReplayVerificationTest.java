/*
 *  ******************************************************************************
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises the isolated reference used by monolithic CUDA-graph replay
 * verification. Triton capture is disabled deliberately so the test reaches
 * replayMonolithicGraph rather than the async Triton path, which cannot perform
 * a blocking output comparison.
 */
@Tag("dsp")
class DspReplayVerificationTest {

    @BeforeEach
    void enableDynamicShapePlan() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        org.nd4j.autodiff.samediff.internal.InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    void restoreEnvironment() {
        Nd4j.getEnvironment().setTritonVerifyKernels(false);
        Nd4j.getEnvironment().setTritonGraphCapture(true);
        Nd4j.getExecutioner().commit();
    }

    @Test
    void cudaGraphReplayVerificationUsesIsolatedReferencePlan() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        assumeTrue(backendName.contains("cuda") || backendName.contains("jcublas"),
                "isolated CUDA-graph verification requires the CUDA backend");

        SameDiff sameDiff = SameDiff.create();
        try {
            SDVariable input = sameDiff.placeHolder("input", DataType.FLOAT, 2, 2);
            input.mul("scaled", 2.0).add("out", 1.0);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);
            sameDiff.setDspAutoCompileEnabled(true);
            sameDiff.setDspNativeAutoCompileEnabled(true);

            // CUDA_GRAPHS is recorder-only; disabling Triton capture guarantees
            // that replayMonolithicGraph owns the captured segment.
            Nd4j.getEnvironment().setTritonGraphCapture(false);
            Nd4j.getEnvironment().setTritonVerifyKernels(true);
            Nd4j.getEnvironment().setDspFreezeMergeSegments(false);

            INDArray currentInput = Nd4j.create(DataType.FLOAT, 2, 2);
            for (int step = 0; step < 8; step++) {
                currentInput.assign(step + 1.0);
                INDArray actual = sameDiff.outputSingle(
                        Collections.singletonMap("input", currentInput), "out");
                INDArray expected = currentInput.mul(2.0).add(1.0);
                assertTrue(expected.equalsWithEps(actual, 1e-6),
                        "step " + step + " diverged while replay verification was enabled");
            }

            assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) > 0,
                    "CUDA_GRAPHS did not launch a monolithic replay");
            assertTrue(DspPlanAssertions.getSegmentReplayCount(sameDiff, 0) > 0,
                    "segment replay counter did not record the monolithic replay");
        } finally {
            sameDiff.close();
        }
    }
}
