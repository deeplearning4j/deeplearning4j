/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.NativeExecutionBinding;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The native plan entries (executeDynamicShapePlan, executeSteadyStatePlan) return Status codes,
 * so callers can name a failure: a null handle or wrongly bound inputs is BAD_INPUT, a context
 * binding the wrong number of outputs is BAD_ARGUMENTS. The entries used to return their own
 * numbers 1-7, which collided with Status (a stale input buffer read as BAD_OUTPUT).
 */
class DspPlanEntryStatusTest {

    @Test
    void nullPlanHandleIsABadInput() {
        NativeOps ops = Nd4j.getNativeOps();
        try {
            assertStatus(OpStatus.ND4J_STATUS_BAD_INPUT, ops.executeDynamicShapePlan(null, null, null),
                    ops, "null plan handle");
            ops.clearLastError();
            assertStatus(OpStatus.ND4J_STATUS_BAD_INPUT, ops.executeSteadyStatePlan(null, null, null),
                    ops, "null plan handle");
        } finally {
            ops.clearLastError();
        }
    }

    @Test
    void wronglyBoundContextIsRejectedWithItsStatus() throws Exception {
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(true);
        try (SameDiff sd = SameDiff.create();
             INDArray x = Nd4j.rand(DataType.FLOAT, 2, 3)) {
            SDVariable in = sd.placeHolder("x", DataType.FLOAT, 2, 3);
            sd.math.exp(in).add("out", 1.0);
            sd.output(Map.of("x", x), "out").get("out").close();

            DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor, "the graph must run through a dynamic shape plan");
            try (NativeExecutionBinding binding = executor.captureNativeExecutionBinding();
                 OpContext context = Nd4j.getExecutioner().buildContext()) {
                NativeOps ops = binding.getBackendOwner().nativeOps();
                INDArray[] surplusOutputs = new INDArray[binding.getOutputCount() + 1];
                for (int i = 0; i < surplusOutputs.length; i++) {
                    surplusOutputs[i] = Nd4j.create(DataType.FLOAT, 2, 3);
                }
                binding.beginNativeUse();
                try {
                    // no inputs bound
                    assertStatus(OpStatus.ND4J_STATUS_BAD_INPUT,
                            ops.executeDynamicShapePlan(binding.getPlanHandle(), context.contextPointer(), null),
                            ops, "input count mismatch");
                    ops.clearLastError();

                    // the plan's inputs, but one output more than it was asked for
                    context.setInputArrays(binding.getExternalInputsSnapshot());
                    context.setOutputArrays(surplusOutputs);
                    assertStatus(OpStatus.ND4J_STATUS_BAD_ARGUMENTS,
                            ops.executeSteadyStatePlan(binding.getPlanHandle(), context.contextPointer(), null),
                            ops, "output count mismatch");
                } finally {
                    ops.clearLastError();
                    binding.completeNativeUse();
                    for (INDArray output : surplusOutputs) {
                        output.close();
                    }
                }
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    private static void assertStatus(OpStatus expected, int status, NativeOps ops, String detail) {
        // Reading the message consumes the error code, so the code is read first.
        int errorCode = ops.lastErrorCode();
        String message = ops.lastErrorMessage();
        assertEquals(expected.name(), OpStatus.nameOf(status), message);
        assertEquals(status, errorCode, "the error reference carries the returned status");
        assertTrue(message != null && message.contains(detail), message);
    }
}
