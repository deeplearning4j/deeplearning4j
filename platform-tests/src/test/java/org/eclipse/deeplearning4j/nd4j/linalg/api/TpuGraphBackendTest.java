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
 * Tests for TPU graph backend integration.
 * Most tests require actual TPU hardware and are gated by system property.
 */
public class TpuGraphBackendTest {

    @Test
    public void testTpuExecutionerTypeExists() {
        // Verify the TPU ExecutionerType enum value exists
        OpExecutioner.ExecutionerType tpuType = OpExecutioner.ExecutionerType.TPU;
        assertNotNull(tpuType);
        assertEquals("TPU", tpuType.name());
    }

    @Test
    public void testGraphExecutionModeTpuExists() {
        GraphExecutionMode tpuMode = GraphExecutionMode.TPU;
        assertNotNull(tpuMode);
        assertEquals(13, tpuMode.getNativeCode());
    }

    @Test
    @EnabledIfSystemProperty(named = "sd.backend", matches = "tpu")
    public void testTpuBackendLoaded() {
        // This test only runs when TPU backend is active
        OpExecutioner executioner = org.nd4j.linalg.factory.Nd4j.getExecutioner();
        assertEquals(OpExecutioner.ExecutionerType.TPU, executioner.type(),
                "When sd.backend=tpu, executioner should be TPU type");
    }
}
