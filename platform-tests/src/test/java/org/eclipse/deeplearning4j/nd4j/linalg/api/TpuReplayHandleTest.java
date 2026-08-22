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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.nativeblas.NativeOps;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Java contracts required by the native TPU replay lifecycle. */
@Tag(TagNames.TPU)
public class TpuReplayHandleTest {

    @Test
    public void testTpuModeRoundTrip() {
        assertEquals(13, GraphExecutionMode.TPU.getNativeCode());
        assertEquals(GraphExecutionMode.TPU, GraphExecutionMode.fromNativeCode(13));
    }

    @Test
    public void testTpuExecutionerUsesNativeControlPlane() throws Exception {
        Class<?> executioner = Class.forName("org.nd4j.linalg.jtpu.ops.TpuExecutioner");
        Class<?> context = Class.forName("org.nd4j.linalg.jtpu.ops.TpuOpContext");
        Class<?> binding = Class.forName("org.nd4j.linalg.jtpu.bindings.Nd4jTpu");

        assertTrue(hasSuperclass(executioner,
                "org.nd4j.linalg.cpu.nativecpu.ops.NativeOpExecutioner"));
        assertTrue(OpContext.class.isAssignableFrom(context));
        assertTrue(NativeOps.class.isAssignableFrom(binding));
    }

    private static boolean hasSuperclass(Class<?> type, String expectedName) {
        for (Class<?> current = type; current != null; current = current.getSuperclass()) {
            if (expectedName.equals(current.getName())) return true;
        }
        return false;
    }
}
