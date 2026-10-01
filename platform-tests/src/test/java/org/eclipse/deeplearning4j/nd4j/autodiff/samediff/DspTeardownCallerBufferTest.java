/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Plan teardown runs between executes, so every input buffer it can reach belongs to the
 * caller: the caller's placeholder arrays, arrays a SameDiff variable holds, and in-place
 * attention state that a generation deletes as soon as it ends while its plan stays cached.
 * Teardown releases only the plan's own storage. It may move a caller buffer only when it
 * holds a frozen pin on it, because the pin registry is the only way to tell a live buffer
 * from a deleted one without reading it. A slot-by-slot plan never pins.
 */
class DspTeardownCallerBufferTest {
    private static final int ROWS = 4;
    private static final int COLS = 16;
    private static final int OUT_COLS = 8;
    private static final int STEPS = 4;
    private static final int FRESH_ARRAYS = 16;

    @Test
    void teardownLeavesTheCallersInputIntact() {
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            try (SameDiff sd = SameDiff.create(); INDArray x = input(0)) {
                // The reshape is a view the plan creates over the caller's buffer.
                sd.reshape("flat", sd.placeHolder("x", DataType.FLOAT, ROWS, COLS), ROWS * COLS)
                        .add(1.0).mul("out", 2.0);
                for (int step = 0; step < STEPS; step++) {
                    INDArray out = sd.output(Map.of("x", x), "out").get("out");
                    for (int i = 0; i < ROWS * COLS; i++) {
                        assertEquals((input(0, i / COLS, i % COLS) + 1.0f) * 2.0f, out.getFloat(i), 0.0f,
                                "step " + step + " [" + i + "]");
                    }
                }
                long storage = storage(x);
                sd.clearDynamicShapePlanCache();
                assertEquals(storage, storage(x), "teardown released the caller's input");
                assertArrayEquals(flatInput(0), x.data().asFloat(), 0.0f, "teardown changed the caller's input");
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    @Test
    void teardownLeavesUnpinnedCallerWeightsInPlace() {
        runSlotBySlotPlan((sd, weight) -> {
            long storage = storage(weight);
            sd.clearDynamicShapePlanCache();
            assertEquals(storage, storage(weight), "teardown moved a caller buffer it holds no pin on");
            assertArrayEquals(flatWeights(0), weight.data().asFloat(), 0.0f,
                    "teardown changed a caller buffer");
        });
    }

    @Test
    void teardownAfterTheCallerDeletedAProtectedInput() {
        runSlotBySlotPlan((sd, weight) -> {
            // The caller rebinds the variable and deletes the array the plan protected, as a
            // generation deletes its KV state while the plan stays cached. Arrays created next
            // can reuse that array's memory, so teardown must not read through the old pointer.
            sd.associateArrayWithVariable(Nd4j.createFromArray(weights()), "w");
            weight.close();
            List<INDArray> fresh = new ArrayList<>();
            try {
                for (int i = 0; i < FRESH_ARRAYS; i++) {
                    fresh.add(Nd4j.createFromArray(weights()).addi(i));
                }
                Nd4j.getExecutioner().commit();
                long[] storage = new long[FRESH_ARRAYS];
                for (int i = 0; i < FRESH_ARRAYS; i++) {
                    storage[i] = storage(fresh.get(i));
                }
                sd.clearDynamicShapePlanCache();
                for (int i = 0; i < FRESH_ARRAYS; i++) {
                    assertEquals(storage[i], storage(fresh.get(i)),
                            "teardown moved array " + i + ", created after the caller deleted the plan's input");
                    assertArrayEquals(flatWeights(i), fresh.get(i).data().asFloat(), 0.0f,
                            "teardown changed array " + i);
                }
            } finally {
                for (INDArray array : fresh) {
                    array.close();
                }
            }
        });
    }

    /** Runs after the plan executed; the weight is the array the plan protected. */
    @FunctionalInterface
    private interface Teardown {
        void run(SameDiff sd, INDArray weight);
    }

    /** Teardown moves weights only on CUDA, so these cases need it. */
    private static void runSlotBySlotPlan(Teardown teardown) {
        assumeTrue(Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA,
                "teardown weight migration only exists on CUDA");
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            try (SameDiff sd = SameDiff.create()) {
                sd.mmul("projected", sd.placeHolder("x", DataType.FLOAT, ROWS, COLS),
                        sd.var("w", Nd4j.createFromArray(weights()))).add("out", 1.0);
                sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                for (int step = 0; step < STEPS; step++) {
                    try (INDArray input = input(step)) {
                        assertProjection(step, sd.output(Map.of("x", input), "out").get("out"));
                    }
                }
                DspPlanAssertions.assertPhaseExact(sd, PlanPhase.SLOT_BY_SLOT,
                        "a slot-by-slot plan holds no pins");
                INDArray weight = sd.getArrForVarName("w");
                assertNotEquals(0L, storage(weight), "the weight must be on the device before teardown");
                teardown.run(sd, weight);
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    /** The device buffer on CUDA, the host buffer elsewhere. */
    private static long storage(INDArray array) {
        NativeOps ops = Nd4j.getNativeOps();
        OpaqueDataBuffer buffer = array.data().opaqueBuffer();
        Pointer pointer = Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA
                ? ops.dbSpecialBuffer(buffer)
                : ops.dbPrimaryBuffer(buffer);
        return pointer == null ? 0L : pointer.address();
    }

    /** Small integers keep the matmul exact under any accumulation order. */
    private static float[][] weights() {
        float[][] w = new float[COLS][OUT_COLS];
        for (int k = 0; k < COLS; k++) {
            for (int col = 0; col < OUT_COLS; col++) {
                w[k][col] = (k * 3 + col) % 5 - 2;
            }
        }
        return w;
    }

    private static float[] flatWeights(float offset) {
        float[][] w = weights();
        float[] flat = new float[COLS * OUT_COLS];
        for (int k = 0; k < COLS; k++) {
            for (int col = 0; col < OUT_COLS; col++) {
                flat[k * OUT_COLS + col] = w[k][col] + offset;
            }
        }
        return flat;
    }

    private static float input(int step, int row, int col) {
        return (row * COLS + col) * 0.25f + step;
    }

    private static float[] flatInput(int step) {
        float[] flat = new float[ROWS * COLS];
        for (int i = 0; i < flat.length; i++) {
            flat[i] = input(step, i / COLS, i % COLS);
        }
        return flat;
    }

    private static INDArray input(int step) {
        float[][] values = new float[ROWS][COLS];
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                values[row][col] = input(step, row, col);
            }
        }
        return Nd4j.createFromArray(values);
    }

    private static void assertProjection(int step, INDArray out) {
        float[][] w = weights();
        assertArrayEquals(new long[]{ROWS, OUT_COLS}, out.shape(), "step " + step);
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < OUT_COLS; col++) {
                float sum = 0;
                for (int k = 0; k < COLS; k++) {
                    sum += input(step, row, k) * w[k][col];
                }
                assertEquals(sum + 1.0f, out.getFloat(row, col), 0.0f,
                        "step " + step + " [" + row + "," + col + "]");
            }
        }
    }
}
