/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.api;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that GraphExecutionMode Java enum values match the C++ GEM_* constants
 * in NativeDynamicShapePlan.h. These must stay in sync for JNI interop.
 */
public class GraphExecutionModeTest {

    @Test
    public void testTpuEnumValue() {
        assertEquals(13, GraphExecutionMode.TPU.getNativeCode(),
                "TPU native code must match C++ GEM_TPU = 13");
    }

    @Test
    public void testHexagonEnumValue() {
        assertEquals(14, GraphExecutionMode.HEXAGON.getNativeCode(),
                "HEXAGON native code must match C++ GEM_HEXAGON = 14");
    }

    @Test
    public void testAllEnumValuesUnique() {
        GraphExecutionMode[] modes = GraphExecutionMode.values();
        java.util.Set<Integer> codes = new java.util.HashSet<>();
        for (GraphExecutionMode mode : modes) {
            assertTrue(codes.add(mode.getNativeCode()),
                    "Duplicate native code " + mode.getNativeCode() + " for " + mode.name());
        }
    }

    @Test
    public void testFromNativeCodeRoundTrip() {
        for (GraphExecutionMode mode : GraphExecutionMode.values()) {
            GraphExecutionMode resolved = GraphExecutionMode.fromNativeCode(mode.getNativeCode());
            assertEquals(mode, resolved,
                    "fromNativeCode(" + mode.getNativeCode() + ") should return " + mode.name());
        }
    }

    @Test
    public void testPortableReplayContract() {
        assertEquals(19, GraphExecutionMode.PORTABLE_REPLAY.getNativeCode(),
                "PORTABLE_REPLAY native code must match C++ GEM_PORTABLE_REPLAY = 19");
        assertTrue(GraphExecutionMode.PORTABLE_REPLAY.requiresGraphBackend(),
                "Portable replay must enter the capture/replay lifecycle when native replay is available");
        assertFalse(GraphExecutionMode.PORTABLE_REPLAY.isSlotBySlot(),
                "Portable replay resolves its backend natively and is not intrinsically slot-by-slot");
    }

    @Test
    public void testEnumOrdering() {
        // Verify the new entries don't break ordering
        assertTrue(GraphExecutionMode.TPU.getNativeCode() > GraphExecutionMode.METAL.getNativeCode(),
                "TPU should have higher native code than METAL");
        assertTrue(GraphExecutionMode.HEXAGON.getNativeCode() > GraphExecutionMode.TPU.getNativeCode(),
                "HEXAGON should have higher native code than TPU");
    }
}
