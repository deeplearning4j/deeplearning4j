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
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Hexagon NPU graph backend integration.
 * Most tests require actual Hexagon hardware and are gated by system property.
 */
public class HexagonGraphBackendTest {

    @Test
    public void testHexagonExecutionerTypeExists() {
        OpExecutioner.ExecutionerType hexType = OpExecutioner.ExecutionerType.HEXAGON;
        assertNotNull(hexType);
        assertEquals("HEXAGON", hexType.name());
    }

    @Test
    public void testGraphExecutionModeHexagonExists() {
        GraphExecutionMode hexMode = GraphExecutionMode.HEXAGON;
        assertNotNull(hexMode);
        assertEquals(14, hexMode.getNativeCode());
    }

    @Test
    @EnabledIfSystemProperty(named = "sd.backend", matches = "hexagon")
    public void testHexagonBackendLoaded() {
        OpExecutioner executioner = org.nd4j.linalg.factory.Nd4j.getExecutioner();
        assertEquals(OpExecutioner.ExecutionerType.HEXAGON, executioner.type(),
                "When sd.backend=hexagon, executioner should be HEXAGON type");
    }
}
