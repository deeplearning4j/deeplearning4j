/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */
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
@Properties(target = "org.nd4j.nativeblas.Nd4jCuda",
                value = {@Platform(include = {"NativeOps.h",
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
                        "NDArrayFactory.h",
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
                        "op_boilerplate.h",
                        "ops/InputType.h",
                        "ops/declarable/OpDescriptor.h",
                        "ops/declarable/BroadcastableOp.h",
                        "ops/declarable/DeclarableOp.h",
                        "ops/declarable/DeclarableListOp.h",
                        "ops/declarable/DeclarableReductionOp.h",
                        "ops/declarable/DeclarableCustomOp.h",
                        "ops/declarable/BooleanOp.h",
                        "ops/declarable/LogicOp.h",
                        "ops/declarable/OpRegistrator.h",
                        "ops/declarable/CustomOperations.h"}, compiler = {"cpp11", "nowarnings"},
                                library = "jnind4jcuda", link = "nd4jcuda", preload = "libnd4jcuda"),
                                @Platform(define = "LIBND4J_ALL_OPS"),
                                @Platform(value = "linux", preload = "gomp@.1",
                                                preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/",
                                                                "/usr/lib/powerpc64-linux-gnu/",
                                                                "/usr/lib/powerpc64le-linux-gnu/"})})
public class Nd4jCudaPresets implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("thread_local", "ND4J_EXPORT", "INLINEDEF", "CUBLASWINAPI", "FORCEINLINE", "_CUDA_H", "_CUDA_D", "_CUDA_G", "_CUDA_HD", "LIBND4J_ALL_OPS", "NOT_EXCLUDED").cppTypes().annotations())
                .put(new Info("NativeOps").base("org.nd4j.nativeblas.NativeOps"))
                .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") String",
                        "@Cast(\"char*\") BytePointer"))
                .put(new Info("Nd4jPointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                .put(new Info("Nd4jLong").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                        "long[]"))
                .put(new Info("Nd4jStatus").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                        "int[]"))
                .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                        "short[]"));

        infoMap.put(new Info("__CUDACC__").define(false))
               .put(new Info("__JAVACPP_HACK__").define(true))
               .put(new Info("LIBND4J_ALL_OPS").define(true))
                .put(new Info("MAX_UINT").translate(false))
               .put(new Info("std::initializer_list", "cnpy::NpyArray", "nd4j::NDArray::applyLambda", "nd4j::NDArray::applyPairwiseLambda",
                             "nd4j::graph::FlatResult", "nd4j::graph::FlatVariable").skip())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")
                                           .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("std::vector<std::vector<int> >").pointerTypes("IntVectorVector").define())
                .put(new Info("std::vector<std::vector<Nd4jLong> >").pointerTypes("LongVectorVector").define())
               .put(new Info("std::vector<nd4j::NDArray<float>*>").pointerTypes("FloatNDArrayVector").define())
               .put(new Info("std::vector<nd4j::NDArray<float16>*>").pointerTypes("HalfNDArrayVector").define())
               .put(new Info("std::vector<nd4j::NDArray<double>*>").pointerTypes("DoubleNDArrayVector").define())
                .put(new Info("nd4j::graph::ResultWrapper").base("org.nd4j.nativeblas.ResultWrapperAbstraction").define())
               .put(new Info("nd4j::IndicesList").purify());

        String classTemplates[] = {
                "nd4j::NDArray",
                "nd4j::NDArrayList",
                "nd4j::ResultSet",
                "nd4j::graph::GraphState",
                "nd4j::NDArrayFactory",
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

        infoMap.put(new Info("nd4j::ops::OpRegistrator::updateMSVC").skip());
    }
}
