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

package org.nd4j.dsp.runtime.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * JavaCPP transport for the stable, backend-neutral SDX runtime C ABI.
 *
 * <p>The generated JNI library contains no model, tensor, or generation logic.
 * It forwards directly into {@code dsp_runtime_c.h}, whose implementation lives
 * in the selected libnd4j backend. Android selects the native library through
 * the {@code sdx.native.library} Maven property so Vulkan, Hexagon/HTP, and
 * Google Tensor packages all reuse this exact generated transport.</p>
 */
@Properties(
        target = "org.nd4j.dsp.runtime.bindings.SdxNative",
        value = {
                @Platform(
                        include = "dsp/runtime/dsp_runtime_c.h",
                        includepath = {
                                "/org/nd4j/dsp/runtime/include/",
                                "../../../../../../libnd4j/include/"
                        },
                        compiler = {"cpp17", "nowarnings"},
                        library = "jnisdx"
                ),
                @Platform(
                        not = {"android", "ios"}
                ),
                @Platform(
                        value = "ios",
                        link = "nd4jmetal",
                        preload = "nd4jmetal"
                )
        }
)
public class SdxRuntimePresets implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
        infoMap
                .put(new Info("SDX_API").cppTypes().annotations())
                .put(new Info("dsp/runtime/dsp_runtime_c.h").objectify())
                .put(new Info("sdx_runtime", "sdx_runtime_t")
                        .pointerTypes("sdx_runtime_t"))
                .put(new Info("sdx_model", "sdx_model_t")
                        .pointerTypes("sdx_model_t"))
                .put(new Info("sdx_context", "sdx_context_t")
                        .pointerTypes("sdx_context_t"))
                .put(new Info("sdx_generation_session",
                        "sdx_generation_session_t")
                        .pointerTypes("sdx_generation_session_t"))
                .put(new Info("sdx_status_t", "sdx_backend_t",
                        "sdx_device_type_t", "sdx_gpu_target_t",
                        "sdx_generation_finish_reason_t")
                        .cast().valueTypes("int"))
                .put(new Info("size_t").cast().valueTypes("long"))
                .put(new Info("uint32_t").cast().valueTypes("int"))
                .put(new Info("int32_t").cast().valueTypes("int")
                        .pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("uint64_t").cast().valueTypes("long")
                        .pointerTypes("LongPointer", "LongBuffer", "long[]"))
                .put(new Info("int64_t").cast().valueTypes("long")
                        .pointerTypes("LongPointer", "LongBuffer", "long[]"));
    }
}
