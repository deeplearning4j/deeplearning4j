/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.nativeblas;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author saudet
 */
@Properties(target = "org.nd4j.nativeblas.Nd4jCuda", helper = "org.nd4j.nativeblas.Nd4jCudaHelper",
                value = {@Platform(define = "LIBND4J_ALL_OPS", include = {
                        "array/DataType.h",
                        "array/ConstantDescriptor.h",
                        "array/ConstantDataBuffer.h",
                        "array/TadPack.h",
                        "memory/MemoryType.h",
                        "Environment.h",
                        "types/utf8string.h",
                        "NativeOps.h",
                        "memory/ExternalWorkspace.h",
                        "memory/Workspace.h",
                        "indexing/NDIndex.h",
                        "indexing/IndicesList.h",
                        "array/DataType.h",
                        "graph/VariableType.h",
                        "graph/ArgumentsList.h",
                        "types/pair.h",
                        "types/pair.h",
                        "NDArray.h",
                        "array/NDArrayList.h",
                        "array/ResultSet.h",
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
                        "array/ShapeList.h",
                        //"op_boilerplate.h",
                        "ops/InputType.h",
                        "ops/declarable/OpDescriptor.h",
                        "ops/declarable/BroadcastableOp.h",                        
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
                        "array/TadPack.h",
                        "helpers/DebugInfo.h",
                        "ops/declarable/CustomOperations.h"},
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
                                library = "jnind4jcuda", link = "nd4jcuda", preload = "libnd4jcuda"),
                                @Platform(value = "linux", preload = "gomp@.1",
                                                preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/",
                                                                "/usr/lib/powerpc64-linux-gnu/",
                                                                "/usr/lib/powerpc64le-linux-gnu/"})})
public class Nd4jCudaPresets implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("thread_local", "ND4J_EXPORT", "INLINEDEF", "CUBLASWINAPI", "FORCEINLINE",
                             "_CUDA_H", "_CUDA_D", "_CUDA_G", "_CUDA_HD", "LIBND4J_ALL_OPS", "NOT_EXCLUDED").cppTypes().annotations())
                .put(new Info("NativeOps.h").objectify())
                .put(new Info("OpaqueTadPack").pointerTypes("OpaqueTadPack"))
                .put(new Info("OpaqueResultWrapper").pointerTypes("OpaqueResultWrapper"))
                .put(new Info("OpaqueShapeList").pointerTypes("OpaqueShapeList"))
                .put(new Info("OpaqueVariablesSet").pointerTypes("OpaqueVariablesSet"))
                .put(new Info("OpaqueVariable").pointerTypes("OpaqueVariable"))
                .put(new Info("OpaqueConstantDataBuffer").pointerTypes("OpaqueConstantDataBuffer"))
                .put(new Info("OpaqueContext").pointerTypes("OpaqueContext"))
                .put(new Info("OpaqueRandomGenerator").pointerTypes("OpaqueRandomGenerator"))
                .put(new Info("OpaqueLaunchContext").pointerTypes("OpaqueLaunchContext"))
                .put(new Info("const char").valueTypes("byte").pointerTypes("@Cast(\"char*\") String",
                        "@Cast(\"char*\") BytePointer"))
                .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") BytePointer",
                        "@Cast(\"char*\") String"))
                .put(new Info("Nd4jPointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                .put(new Info("Nd4jLong").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                        "long[]"))
                .put(new Info("Nd4jStatus").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                        "int[]"))
                .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                        "short[]"))
                .put(new Info("bfloat16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                        "short[]"));

        infoMap.put(new Info("__CUDACC__", "MAX_UINT", "HAVE_MKLDNN").define(false))
               .put(new Info("__JAVACPP_HACK__", "LIBND4J_ALL_OPS","__CUDABLAS__").define(true))
               .put(new Info("std::initializer_list", "cnpy::NpyArray", "nd4j::NDArray::applyLambda", "nd4j::NDArray::applyPairwiseLambda",
                             "nd4j::graph::FlatResult", "nd4j::graph::FlatVariable", "nd4j::NDArray::subarray").skip())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")
                                           .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("std::vector<std::vector<int> >").pointerTypes("IntVectorVector").define())
                .put(new Info("std::vector<std::vector<Nd4jLong> >").pointerTypes("LongVectorVector").define())
               .put(new Info("std::vector<nd4j::NDArray*>").pointerTypes("NDArrayVector").define())
                .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BooleanPointer", "boolean[]"))
                .put(new Info("nd4j::graph::ResultWrapper").base("org.nd4j.nativeblas.ResultWrapperAbstraction").define())
               .put(new Info("nd4j::IndicesList").purify());
/*
        String classTemplates[] = {
                "nd4j::NDArray",
                "nd4j::NDArrayList",
                "nd4j::ResultSet",
                "nd4j::OpArgsHolder",
                "nd4j::graph::GraphState",
                "nd4j::graph::Variable",
                "nd4j::graph::VariablesSet",
                "nd4j::graph::Stash",
                "nd4j::graph::VariableSpace",
                "nd4j::graph::Context",
                "nd4j::graph::ContextPrototype",
                "nd4j::ops::DeclarableOp",
                "nd4j::ops::DeclarableListOp",
                "nd4j::ops::DeclarableReductionOp",
                "nd4j::ops::DeclarableCustomOp",
                "nd4j::ops::BooleanOp",
                "nd4j::ops::BroadcastableOp",
                "nd4j::ops::LogicOp"};
        for (String t : classTemplates) {
            String s = t.substring(t.lastIndexOf(':') + 1);
            infoMap.put(new Info(t + "<float>").pointerTypes("Float" + s))
                    .put(new Info(t + "<float16>").pointerTypes("Half" + s))
                    .put(new Info(t + "<double>").pointerTypes("Double" + s));

        }
*/
        infoMap.put(new Info("nd4j::ops::OpRegistrator::updateMSVC").skip());
    }
}
