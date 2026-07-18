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

package org.nd4j.autodiff.samediff.execution;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class DynamicShapePlanExecutorModeResolutionTest {

    @Test
    void javaResolutionLeavesBackendSelectionToNativePlan() {
        for (GraphExecutionMode mode : GraphExecutionMode.values()) {
            if (mode != GraphExecutionMode.TRITON) {
                assertSame(mode,
                        DynamicShapePlanExecutor.resolveEffectiveGraphExecutionMode(mode, false, true),
                        "Java must not infer native graph-backend availability for " + mode);
            }
        }
    }

    @Test
    void unavailableTritonUsesOnlyTheConfiguredFallbackPolicy() {
        assertSame(GraphExecutionMode.AUTO,
                DynamicShapePlanExecutor.resolveEffectiveGraphExecutionMode(
                        GraphExecutionMode.TRITON, false, true));
        assertSame(GraphExecutionMode.TRITON,
                DynamicShapePlanExecutor.resolveEffectiveGraphExecutionMode(
                        GraphExecutionMode.TRITON, false, false));
        assertSame(GraphExecutionMode.TRITON,
                DynamicShapePlanExecutor.resolveEffectiveGraphExecutionMode(
                        GraphExecutionMode.TRITON, true, true));
    }

    @Test
    void compilingAutoDoesNotRemapItInJava() {
        try (SameDiff sd = SameDiff.create()) {
            sd.placeHolder("input", DataType.FLOAT, 1, 4)
                    .add(1.0)
                    .rename("out");

            GraphExecutionMode effectiveMode = sd.compileNativeDynamicShapePlan(
                    Collections.singletonList("out"), GraphExecutionMode.AUTO, true);

            assertEquals(GraphExecutionMode.AUTO, effectiveMode,
                    "AUTO must reach NativeDynamicShapePlan so CPU graph backends remain eligible");
        }
    }
}
