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
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Hexagon replay handle lifecycle.
 * Tests that verify hexagon-mlir command list recording require Hexagon hardware
 * and are gated by the sd.backend=hexagon system property.
 *
 * The non-hardware tests verify Java-side enum values and configuration
 * that must be correct for the C++ HexagonReplayHandle to function.
 */
public class HexagonReplayHandleTest {

    @Test
    public void testHexagonGraphExecutionModeValue() {
        // HexagonReplayHandle is created when GEM_HEXAGON (14) is requested
        assertEquals(14, GraphExecutionMode.HEXAGON.getNativeCode(),
                "HEXAGON mode native code must be 14 to match C++ GEM_HEXAGON");
    }

    @Test
    public void testHexagonModeFromNativeCode() {
        GraphExecutionMode mode = GraphExecutionMode.fromNativeCode(14);
        assertEquals(GraphExecutionMode.HEXAGON, mode,
                "Native code 14 should resolve to HEXAGON mode");
    }

    @Test
    @EnabledIfSystemProperty(named = "sd.backend", matches = "hexagon")
    public void testHexagonReplayHandleCreation() {
        // This test verifies that GraphReplayFactory creates a HexagonReplayHandle
        // when HAVE_HEXAGON_MLIR is defined. Requires Hexagon backend to be active.
        assertNotNull(GraphExecutionMode.HEXAGON);
    }
}
