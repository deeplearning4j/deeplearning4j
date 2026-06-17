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

package org.nd4j.presets.hexagon;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;
import org.nd4j.presets.OpExclusionUtils;

/**
 * JavaCPP Presets for ND4J Hexagon NPU backend using hexagon-mlir.
 *
 * This class defines the native library bindings for Hexagon NPU operations
 * through Qualcomm's hexagon-mlir compiler runtime.
 */
@Properties(target = "org.nd4j.linalg.hexagon.bindings.Nd4jHexagon", helper = "org.nd4j.presets.hexagon.Nd4jHexagonHelper",
        value = {@Platform(define = {"SD_ALL_OPS", "SD_HEXAGON", "HAVE_HEXAGON_MLIR"}, include = {
                // Core headers - order matters
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
                "memory/Workspace.h",
                "indexing/NDIndex.h",
                "indexing/IndicesList.h",
                "graph/VariableType.h",
                "graph/ArgumentsList.h",
                "types/pair.h",
                "array/NDArray.h",
                "array/NDArrayList.h",
                "array/ResultSet.h",
                "graph/RandomGenerator.h",
                "graph/Variable.h",
                "graph/VariablesSet.h",
                "graph/Intervals.h",
                "graph/Stash.h",
                "graph/VariableSpace.h",
                "helpers/helper_generator.h",
                "graph/profiling/GraphProfile.h",
                "graph/profiling/NodeProfile.h",
                "graph/Context.h",
                "graph/ContextPrototype.h",
                "helpers/shape.h",
                "array/ShapeList.h",
                "system/op_boilerplate.h",
                "ops/InputType.h",
                "ops/declarable/OpDescriptor.h",
                "ops/declarable/PlatformHelper.h",
                "ops/declarable/BroadcastableOp.h",
                "ops/declarable/BroadcastableBoolOp.h",
                "helpers/OpArgsHolder.h",
                "ops/declarable/DeclarableOp.h",
                "ops/declarable/DeclarableListOp.h",
                "ops/declarable/DeclarableReductionOp.h",
                "ops/declarable/DeclarableCustomOp.h",
                "ops/declarable/BooleanOp.h",
                "ops/declarable/LogicOp.h",
                "ops/declarable/OpRegistrator.h",
                "execution/ContextBuffers.h",
                "execution/LaunchContext.h",
                "array/ShapeDescriptor.h",
                "array/TadDescriptor.h",
                "helpers/DebugInfo.h",
                "ops/declarable/CustomOperations.h",
                "build_info.h",
                // Hexagon graph backend headers
                "graph/hexagon/HexagonRuntimeManager.h",
                "graph/hexagon/HexagonReplayHandle.h",
                "graph/hexagon/HexagonIRBuilder.h",
                "graph/hexagon/HexagonGraphBackend.h",
        },
                exclude = {"ops/declarable/headers/activations.h",
                        "ops/declarable/headers/boolean.h",
                        "ops/declarable/headers/broadcastable.h",
                        "ops/declarable/headers/convo.h",
                        "ops/declarable/headers/list.h",
                        "ops/declarable/headers/recurrent.h",
                        "ops/declarable/headers/transforms.h",
                        "ops/declarable/headers/parity_ops.h",
                        "ops/declarable/headers/shape.h",
                        "ops/declarable/headers/random.h",
                        "ops/declarable/headers/nn.h",
                        "ops/declarable/headers/blas.h",
                        "ops/declarable/headers/bitwise.h",
                        "ops/declarable/headers/tests.h",
                        "ops/declarable/headers/loss.h",
                        "ops/declarable/headers/datatypes.h",
                        "ops/declarable/headers/third_party.h",
                        "cnpy/cnpy.h"
                },
                compiler = {"cpp11", "nowarnings"},
                library = "jnind4jhexagon", link = "nd4jhexagon", preload = "libnd4jhexagon"),
                @Platform(value = "linux", preload = "gomp@.1", preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/"}),
                @Platform(value = "linux-arm64", preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"})})
public class Nd4jHexagonPresets implements LoadEnabled, BuildEnabled, InfoMapper {
    private Logger logger;
    private java.util.Properties properties;
    private String encoding;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
    }

    @Override
    public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> preloads = properties.get("platform.preload");

        String[] defaultLibs = {"nd4jhexagon"};

        for (String lib : defaultLibs) {
            if (platform.startsWith("linux")) {
                lib = lib + (lib.indexOf('@') < 0 ? "" : lib.substring(lib.indexOf('@')));
                if (!preloads.contains(lib)) {
                    preloads.add(lib);
                }
            }
        }

        // Attempt to load hexagon-mlir runtime library
        if (platform.startsWith("linux")) {
            preloads.add(0, "hexagon_mlir_runtime");
        }
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("thread_local", "SD_LIB_EXPORT", "SD_LIB_HIDDEN", "SD_INLINE",
                        "SD_HOST", "SD_DEVICE", "SD_HOST_DEVICE", "SD_KERNEL",
                        "SD_ALL_OPS", "SD_HEXAGON", "HAVE_HEXAGON_MLIR")
                        .cppTypes().annotations())
               .put(new Info("openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h",
                        "lapack.h", "lapacke.h", "lapacke_utils.h").skip())
               .put(new Info("std::initializer_list", "cnpy::NpyArray", "cnpy::npz_t").skip())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")
                        .pointerTypes("BytePointer"))
               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("std::vector<std::vector<int> >").pointerTypes("IntVectorVector").define())
               .put(new Info("std::vector<std::vector<sd::LongType> >").pointerTypes("LongVectorVector").define())
               .put(new Info("std::vector<const sd::NDArray*>").pointerTypes("ConstNDArrayVector").define())
               .put(new Info("std::vector<sd::NDArray*>").pointerTypes("NDArrayVector").define())
               .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BooleanPointer", "boolean[]"));

        // Handle operations exclusion
        OpExclusionUtils.processExclusions(infoMap);
    }
}
