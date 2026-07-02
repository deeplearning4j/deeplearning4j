/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.presets.metal;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;
import org.nd4j.presets.OpExclusionUtils;

/**
 * JavaCPP Presets for the ND4J Metal/MLX backend (Apple Silicon).
 *
 * <p>This class defines the native library bindings for Apple Silicon GPU operations
 * via the Metal framework (per-op MPS helpers + Metal ICB replay) and MLX
 * (lazy graph evaluation via mlx::core, dispatching to Metal shaders).</p>
 *
 * <h3>Native build requirements</h3>
 * <ul>
 *   <li>macOS arm64 (Apple Silicon M1+)</li>
 *   <li>Xcode Command Line Tools (clang with Objective-C++ / .mm support)</li>
 *   <li>Metal framework (system-provided on macOS 10.11+)</li>
 *   <li>MetalPerformanceShaders framework (system-provided on macOS 10.13+)</li>
 *   <li>MLX library: install from https://github.com/ml-explore/mlx or via
 *       {@code -DSD_MLX=ON} in the libnd4j CMake build which fetches + builds MLX</li>
 * </ul>
 *
 * <h3>Cmake build command (macOS arm64)</h3>
 * <pre>
 * /path/to/mvn -Pcpu -Dlibnd4j.chip=cpu -Dlibnd4j.helper=mps \
 *   -Dlibnd4j.platform=macosx-arm64 -Djavacpp.platform=macosx-arm64 \
 *   -DSD_MLX=ON -Dlibnd4j.triton=ON \
 *   -pl libnd4j,:nd4j-native clean install -DskipTests 2>&1 | tee metal-build.log
 * </pre>
 *
 * <p>This class is intentionally kept as a skeleton — the full header list mirrors
 * {@code Nd4jTpuPresets} and will be filled in when the macOS arm64 CI job
 * (build-deploy-mac-arm64.yml, helper=mps-compile) is wired to this preset.</p>
 */
@Properties(target = "org.nd4j.linalg.metal.bindings.Nd4jMetal",
        helper = "org.nd4j.presets.metal.Nd4jMetalHelper",
        value = {@Platform(
                define = {"SD_ALL_OPS", "SD_METAL", "HAVE_MPS", "HAVE_MLX"},
                // Compiled only on macOS arm64 — requires Metal + MLX headers
                // include list will mirror Nd4jCpu + add MPS/MLX-specific headers:
                //   "graph/metal/MetalReplayHandle.h",
                //   "graph/cpu/MlxGraphBackend.h",
                //   "ops/declarable/platform/mps/mpsUtils.h"
                include = {
                        "generated/include_ops.h",
                        "array/DataType.h",
                        "array/DataBuffer.h",
                        "array/PointerDeallocator.h",
                        "array/PointerWrapper.h",
                        "array/ConstantDescriptor.h",
                        "array/ConstantDataBuffer.h",
                        "array/ConstantShapeBuffer.h",
                        "array/ConstantOffsetsBuffer.h",
                        "array/TadPack.h",
                        "execution/ErrorReference.h",
                        "execution/Engine.h",
                        "execution/ExecutionMode.h",
                        "memory/MemoryType.h",
                        "system/Environment.h",
                        "types/utf8string.h",
                        "legacy/NativeOps.h",
                        "dsp/NativeOpsDsp.h",
                        "memory/ExternalWorkspace.h",
                        "memory/Workspace.h"
                        // TODO when Metal bindings land:
                        // "graph/metal/MetalReplayHandle.h",
                        // "graph/cpu/MlxGraphBackend.h"
                },
                compiler = {"default", "-fobjc-arc"},   // ARC required for .mm files
                link = {"nd4j_metal"},
                // macOS-only: link Metal + MPS + Accelerate frameworks
                // These are added by buildnativeoperations.sh when -Dlibnd4j.helper=mps
                // frameworkPath applies on macOS; link value is the .dylib name
                preload = {"nd4j_metal"}
        )})
public class Nd4jMetalPresets implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        // Restrict this preset to macOS arm64
        String platform = properties.getProperty("platform");
        if (platform != null && !platform.startsWith("macosx")) {
            // Clear the include list so no native compilation is attempted off-platform
            properties.get("platform.include").clear();
            properties.get("platform.link").clear();
        }
    }

    @Override
    public void map(InfoMap infoMap) {
        // Exclusions mirror Nd4jTpuPresets / Nd4jCpuPresets: skip ops with
        // CUDA-only signatures and all generated ops that require a full CPU build.
        // Filled in when native bindings are generated.
    }
}
