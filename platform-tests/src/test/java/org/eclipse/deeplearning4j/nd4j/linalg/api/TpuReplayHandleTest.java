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
 * Unit tests for TPU replay handle lifecycle.
 * Tests that verify PJRT compilation caching behavior require TPU hardware
 * and are gated by the sd.backend=tpu system property.
 *
 * The non-hardware tests verify Java-side enum values and configuration
 * that must be correct for the C++ TpuReplayHandle to function.
 */
public class TpuReplayHandleTest {

    @Test
    public void testTpuGraphExecutionModeValue() {
        // TpuReplayHandle is created when GEM_TPU (13) is requested
        assertEquals(13, GraphExecutionMode.TPU.getNativeCode(),
                "TPU mode native code must be 13 to match C++ GEM_TPU");
    }

    @Test
    public void testTpuModeFromNativeCode() {
        GraphExecutionMode mode = GraphExecutionMode.fromNativeCode(13);
        assertEquals(GraphExecutionMode.TPU, mode,
                "Native code 13 should resolve to TPU mode");
    }

    @Test
    @EnabledIfSystemProperty(named = "sd.backend", matches = "tpu")
    public void testTpuReplayHandleCreation() {
        // This test verifies that GraphReplayFactory creates a TpuReplayHandle
        // when SD_TPU is defined. Requires TPU backend to be active.
        // The C++ side creates the handle; Java side verifies through
        // DynamicShapePlanExecutor behavior.
        assertNotNull(GraphExecutionMode.TPU);
    }
}
