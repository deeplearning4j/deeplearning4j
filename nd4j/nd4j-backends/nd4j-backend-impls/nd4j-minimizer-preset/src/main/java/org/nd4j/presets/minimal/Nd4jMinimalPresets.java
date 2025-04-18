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

package org.nd4j.presets.minimal;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;
import org.nd4j.presets.OpExclusionUtils;

/**
 *
 * @author saudet
 */
@Properties(target = "org.nd4j.linalg.minimal.bindings.Nd4jMinimal", helper = "org.nd4j.presets.minimal.Nd4jMinimalHelper",
        value = {@Platform(define = {"SD_ALL_OPS"}, include = {
                //note, order matters here
                //this particular header file is either
                //going to be the source of ops, see also:
                //https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/blas/CMakeLists.txt#L76
                //https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/buildnativeoperations.sh#L517
                "generated/include_ops.h",
                "memory/MemoryType.h",
                "array/DataType.h",
                "array/DataBuffer.h",
                "array/PointerDeallocator.h",
                "array/PointerWrapper.h",
                "array/ConstantDataBuffer.h",
                "array/ConstantShapeBuffer.h",
                "array/ConstantOffsetsBuffer.h",
                "array/ConstantDescriptor.h",
                "array/TadPack.h",
                "execution/ErrorReference.h",
                "execution/Engine.h",
                "execution/ExecutionMode.h",
                "system/CudaLimitType.h",
                "system/Environment.h",
                "types/utf8string.h",
                "legacy/NativeOps.h",
                "build_info.h",
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
                "types/pair.h",
                "graph/RandomGenerator.h",
                "graph/Variable.h",
                "graph/VariablesSet.h",
                "graph/FlowPath.h",
                "graph/Intervals.h",
                "graph/Stash.h",
                "graph/GraphState.h",
                "graph/VariableSpace.h",
                "helpers/helper_generator.h",
                "graph/profiling/GraphProfile.h",
                "graph/profiling/NodeProfile.h",
                "graph/Context.h",
                "graph/ContextPrototype.h",
                "graph/ResultWrapper.h",
                "helpers/shape.h",
                "helpers/OpArgsHolder.h",
                "array/ShapeList.h",
                "system/type_boilerplate.h",
                "system/op_boilerplate.h",
                "ops/InputType.h",
                "ops/declarable/OpDescriptor.h",
                "ops/declarable/PlatformHelper.h",
                "ops/declarable/BroadcastableOp.h",
                "ops/declarable/BroadcastableBoolOp.h",
                "ops/declarable/DeclarableOp.h",
                "ops/declarable/DeclarableListOp.h",
                "ops/declarable/DeclarableReductionOp.h",
                "ops/declarable/DeclarableCustomOp.h",
                "ops/declarable/BooleanOp.h",
                "ops/declarable/LogicOp.h",
                "ops/declarable/OpRegistrator.h",
                "ops/declarable/CustomOperations.h",
                "ops/declarable/headers/activations.h",
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
                "ops/declarable/headers/tests.h",
                "ops/declarable/headers/bitwise.h",
                "ops/declarable/headers/loss.h",
                "ops/declarable/headers/datatypes.h",
                "execution/ContextBuffers.h",
                "execution/LaunchContext.h",
                "array/ShapeDescriptor.h",
                "array/TadDescriptor.h",
                "helpers/DebugInfo.h",
                //note: this is for the generated operations
                //libnd4j should be built with an include/generated/include_ops.h
                //before initiating a build, generally this will just default to
                //#define SD_ALL_OPS true but can also be the list of op definitions
                //declared for the cmake build
                "ops/declarable/headers/third_party.h"},
                exclude = {"ops/declarable/headers/activations.h",
                        "ops/declarable/headers/boolean.h",
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
                        "openblas_config.h",
                        "cblas.h",
                        "lapacke_config.h",
                        "lapacke_mangling.h",
                        "lapack.h",
                        "lapacke.h",
                        "lapacke_utils.h",
                        "cnpy/cnpy.h"
                },
                compiler = {"cpp11", "nowarnings"},
                library = "jnind4jcpu", link = "nd4jcpu", preload = "libnd4jcpu"),
                @Platform(value = "linux", preload = "gomp@.1", preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/"}),
                @Platform(value = "linux-armhf", preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
                @Platform(value = "linux-arm64", preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}),
                @Platform(value = "linux-ppc64", preloadpath = {"/usr/powerpc64-linux-gnu/lib/", "/usr/powerpc64le-linux-gnu/lib/", "/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
                @Platform(value = "windows", preload = {"libwinpthread-1", "libgcc_s_seh-1", "libgomp-1", "libstdc++-6", "libnd4jcpu"}),
                @Platform(extension = {"-onednn", "-onednn-avx512","-onednn-avx2", "-vednn", "-vednn-avx512", "-vednn-avx2", "-","-avx2","-avx512", "-compat"}, resource={"libnd4jcpu_device.vso"})
        })
public class Nd4jMinimalPresets implements InfoMapper, BuildEnabled {

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
    public void map(InfoMap infoMap) {
        OpExclusionUtils.processOps(logger, properties, infoMap);
    }


}
