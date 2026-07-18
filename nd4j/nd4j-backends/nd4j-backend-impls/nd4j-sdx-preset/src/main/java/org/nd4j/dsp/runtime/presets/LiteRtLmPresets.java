/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */

package org.nd4j.dsp.runtime.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 * JavaCPP transport for the stable LiteRT-LM C API.
 *
 * <p>This preset deliberately binds only the public {@code c/engine.h} ABI.
 * Model loading, tokenization, conversation state, sampling, streaming, and NPU
 * execution stay inside the pinned LiteRT-LM runtime. The Google Tensor package
 * selects the {@code npu} backend and a vendor dispatch directory in the Java
 * facade; JavaCPP is only the Java/Kotlin-to-C transport.</p>
 */
@Properties(
        target = "org.nd4j.dsp.runtime.litertlm.bindings.LiteRtLmNative",
        value = {
                @Platform(
                        include = "c/engine.h",
                        compiler = {"cpp17", "nowarnings"},
                        library = "jnilitertlm"
                ),
                @Platform(
                        value = "android",
                        link = "litert-lm",
                        preload = "litert-lm"
                )
        }
)
public class LiteRtLmPresets implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
        infoMap
                .put(new Info("LITERT_LM_C_API_EXPORT").cppTypes().annotations())
                .put(new Info("c/engine.h").objectify());

        String[] opaqueTypes = {
                "LiteRtLmEngine",
                "LiteRtLmSession",
                "LiteRtLmResponses",
                "LiteRtLmEngineSettings",
                "LiteRtLmBenchmarkInfo",
                "LiteRtLmConversation",
                "LiteRtLmConversationOptionalArgs",
                "LiteRtLmJsonResponse",
                "LiteRtLmDetokenizeResult",
                "LiteRtLmTokenizeResult",
                "LiteRtLmTokenUnion",
                "LiteRtLmTokenUnions",
                "LiteRtLmInputData",
                "LiteRtLmSessionConfig",
                "LiteRtLmConversationConfig",
                "LiteRtLmSamplerParams"
        };
        for (String type : opaqueTypes) {
            infoMap.put(new Info(type).pointerTypes(type));
        }

        infoMap
                .put(new Info(
                        "LiteRtLmTokenUnionType",
                        "LiteRtLmSamplerType",
                        "LiteRtLmInputDataType"
                ).cast().valueTypes("int"))
                .put(new Info("size_t").cast().valueTypes("long"))
                .put(new Info("int32_t").cast().valueTypes("int")
                        .pointerTypes("IntPointer", "IntBuffer", "int[]"));
    }
}
