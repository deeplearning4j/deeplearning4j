/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A caller may delete its input as soon as output() returns while the plan stays cached. Plan
 * introspection, invalidation and teardown run between executes, so they may not read an
 * output slot that is the caller's input (an identity) or a plan view over it: such an output
 * is reported without being read, and the plan's own outputs are still validated. Run with
 * {@code -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full}: the diagnostics
 * dereference slots, so a read of the deleted input faults instead of passing silently.
 */
class DspCallerInputIntrospectionTest {
    private static final int ROWS = 4;
    private static final int COLS = 16;
    /** An identity of the input, a view of it, and an output the plan computes. */
    private static final String[] OUTPUTS = {"same", "flat", "computed"};

    @ParameterizedTest
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "SLOT_BY_SLOT"})
    void validatorsSkipOutputsBackedByTheCallersDeletedInput(GraphExecutionMode mode) {
        runWithDsp(() -> {
            try (SameDiff sd = graph(mode)) {
                INDArray x = input(0);
                sd.output(Map.of("x", x), OUTPUTS);
                x.close();

                DspHandle handle = sd.dsp();
                int[] flags = handle.validateOutputs();
                assertEquals(OUTPUTS.length, flags.length, "one flag per requested output");
                for (int flag : flags) {
                    assertEquals(0, flag, "every output is valid or the caller's: " + Arrays.toString(flags));
                }

                float[] norms = new float[OUTPUTS.length];
                boolean[] stale = new boolean[OUTPUTS.length];
                assertEquals(0, handle.detectStaleOutputs(norms, stale, 1e-10f), "nothing is stale on first use");
                long measured = 0;
                for (float norm : norms) {
                    if (norm != 0f) measured++;
                }
                assertTrue(measured >= 1, "the plan's own output is still measured: " + Arrays.toString(norms));
            }
        });
    }

    @ParameterizedTest
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "SLOT_BY_SLOT"})
    void invalidationAfterTheCallerDeletedItsInput(GraphExecutionMode mode) {
        runWithDsp(() -> {
            try (SameDiff sd = graph(mode)) {
                INDArray x = input(0);
                sd.output(Map.of("x", x), OUTPUTS);
                x.close();
                // Invalidation resets every segment's slots between executes, while the identity
                // slot still names the deleted input and the view slot wraps its storage.
                sd.dsp().invalidateBackendCaches("");
                assertNextExecuteCorrect(sd, 1);
            }
        });
    }

    @ParameterizedTest
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "SLOT_BY_SLOT"})
    void teardownAfterTheCallerDeletedARequestedViewsInput(GraphExecutionMode mode) {
        runWithDsp(() -> {
            try (SameDiff sd = graph(mode)) {
                INDArray x = input(0);
                sd.output(Map.of("x", x), OUTPUTS);
                x.close();
                // Teardown keeps requested outputs' wrappers and re-measures the plan after
                // clearing its record of the caller's inputs; the "flat" view's storage is gone.
                sd.clearDynamicShapePlanCache();
                assertNextExecuteCorrect(sd, 1);
            }
        });
    }

    private static void runWithDsp(Runnable body) {
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            body.run();
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    private static SameDiff graph(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, ROWS, COLS);
        sd.identity("same", x);
        sd.reshape("flat", x, ROWS * COLS);
        x.add("computed", 1.0);
        sd.setGraphExecutionMode(mode);
        return sd;
    }

    private static void assertNextExecuteCorrect(SameDiff sd, int step) {
        try (INDArray next = input(step)) {
            Map<String, INDArray> out = sd.output(Map.of("x", next), OUTPUTS);
            float[] expected = flatInput(step);
            assertArrayEquals(expected, out.get("same").dup().data().asFloat(), 0f, "same");
            assertArrayEquals(expected, out.get("flat").dup().data().asFloat(), 0f, "flat");
            for (int i = 0; i < expected.length; i++) {
                expected[i] += 1.0f;
            }
            assertArrayEquals(expected, out.get("computed").dup().data().asFloat(), 0f, "computed");
        }
    }

    private static float value(int step, int i) {
        return i * 0.25f + step;
    }

    private static float[] flatInput(int step) {
        float[] flat = new float[ROWS * COLS];
        for (int i = 0; i < flat.length; i++) {
            flat[i] = value(step, i);
        }
        return flat;
    }

    private static INDArray input(int step) {
        return Nd4j.createFromArray(flatInput(step)).reshape(ROWS, COLS);
    }
}
