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

package org.nd4j.dsp.runtime;

import org.bytedeco.javacpp.BytePointer;
import org.junit.jupiter.api.Test;
import org.nd4j.dsp.runtime.bindings.SdxNative;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class SdxRuntimeJavaCppTest {

    @Test
    void loadsGeneratedTransportAndCreatesRuntime() {
        try (SdxRuntime runtime = SdxRuntime.create()) {
            assertEquals(1, runtime.abiVersion());
            assertFalse(runtime.lastError() == null);
        }
    }

    @Test
    void mobileOptionsAreStrictAndAotOnly() {
        SdxRuntime.ModelOptions options =
                SdxRuntime.ModelOptions.mobileVulkan();

        assertEquals(SdxRuntime.SDX_BACKEND_VULKAN, options.backend);
        assertEquals(1, options.strict_backend);
        assertEquals(0, options.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_VULKAN, options.gpu_target);

        SdxRuntime.ModelOptions hexagon =
                SdxRuntime.ModelOptions.mobileHexagon();
        assertEquals(SdxRuntime.SDX_BACKEND_HEXAGON, hexagon.backend);
        assertEquals(1, hexagon.strict_backend);
        assertEquals(0, hexagon.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, hexagon.gpu_target);

        String cacheDirectory = "/tmp/sdx-device-cache";
        SdxRuntime.ModelOptions nnapi = SdxRuntime.ModelOptions.mobileNnapiAccelerator()
                .deviceCompilationCacheDirectory(cacheDirectory);
        assertEquals(SdxRuntime.SDX_BACKEND_NNAPI, nnapi.backend);
        assertEquals(1, nnapi.strict_backend);
        assertEquals(0, nnapi.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, nnapi.gpu_target);
        assertEquals(cacheDirectory, nnapi.device_compilation_cache_directory);
        assertDoesNotThrow(() -> SdxNative.sdx_model_options_t.class.getDeclaredMethod(
                "device_compilation_cache_directory", BytePointer.class));

        SdxRuntime.RunOptions hexagonRun =
                SdxRuntime.RunOptions.mobileHexagon();
        assertEquals(SdxRuntime.SDX_BACKEND_HEXAGON, hexagonRun.backend);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, hexagonRun.gpu_target);

        SdxRuntime.RunOptions nnapiRun =
                SdxRuntime.RunOptions.mobileNnapiAccelerator();
        assertEquals(SdxRuntime.SDX_BACKEND_NNAPI, nnapiRun.backend);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, nnapiRun.gpu_target);

        SdxRuntime.ModelOptions openVino = SdxRuntime.ModelOptions.desktopOpenVino();
        assertEquals(SdxRuntime.SDX_BACKEND_OPENVINO, openVino.backend);
        assertEquals(1, openVino.strict_backend);
        assertEquals(1, openVino.allow_runtime_jit);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, openVino.gpu_target);

        SdxRuntime.RunOptions openVinoRun = SdxRuntime.RunOptions.desktopOpenVino();
        assertEquals(SdxRuntime.SDX_BACKEND_OPENVINO, openVinoRun.backend);
        assertEquals(SdxRuntime.SDX_GPU_TARGET_AUTO, openVinoRun.gpu_target);
    }
}
