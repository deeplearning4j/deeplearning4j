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
import org.bytedeco.javacpp.tools.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

/**
 *
 * @author saudet
 */
@Properties(target = "org.nd4j.nativeblas.Nd4jCpu",
                value = {@Platform(include = {"NativeOps.h",
                                              "memory/ExternalWorkspace.h",
                                              "memory/Workspace.h",
                                              "indexing/NDIndex.h",
                                              "indexing/IndicesList.h",
                                              "array/DataType.h",
                                              "graph/VariableType.h",
                                              "graph/ArgumentsList.h",
                                              "types/pair.h",
                                              "NDArray.h",
                                              "array/NDArrayList.h",
                                              "array/ResultSet.h",
                                              "types/pair.h",
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
                                              "ops/declarable/headers/third_party.h"},
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
                                              "ops/declarable/headers/third_party.h"},
                                compiler = {"cpp11", "nowarnings"}, library = "jnind4jcpu", link = "nd4jcpu", preload = "libnd4jcpu"),
                                @Platform(value = "linux", preload = "gomp@.1",
                                                preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/",
                                                                "/usr/lib/powerpc64-linux-gnu/",
                                                                "/usr/lib/powerpc64le-linux-gnu/"}),
                @Platform(define = "LIBND4J_ALL_OPS"),
                @Platform(value = "macosx", preload = {"gcc_s@.1", "gomp@.1", "stdc++@.6"},
                                preloadpath = {"/usr/local/lib/gcc/7/", "/usr/local/lib/gcc/6/", "/usr/local/lib/gcc/5/"}),
                @Platform(extension = {"-avx512", "-avx2"}) })
public class Nd4jCpuPresets implements InfoMapper, BuildEnabled {

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
        infoMap.put(new Info("thread_local", "ND4J_EXPORT", "INLINEDEF", "CUBLASWINAPI", "FORCEINLINE", "_CUDA_H", "_CUDA_D", "_CUDA_G", "_CUDA_HD", "LIBND4J_ALL_OPS", "NOT_EXCLUDED").cppTypes().annotations())
                        .put(new Info("NativeOps").base("org.nd4j.nativeblas.NativeOps"))
                        .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") String",
                                        "@Cast(\"char*\") BytePointer"))
                        .put(new Info("Nd4jPointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                        .put(new Info("Nd4jLong").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                                        "long[]"))
                        .put(new Info("int").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                        "int[]"))
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

        // pick up custom operations automatically from CustomOperations.h and headers in libnd4j
        String separator = properties.getProperty("platform.path.separator");
        String[] includePaths = properties.getProperty("platform.includepath").split(separator);
        File file = null;
        for (String path : includePaths) {
            file = new File(path, "ops/declarable/CustomOperations.h");
            if (file.exists()) {
                break;
            }
        }
        ArrayList<File> files = new ArrayList<File>();
        ArrayList<String> opTemplates = new ArrayList<String>();
        files.add(file);
        files.addAll(Arrays.asList(new File(file.getParent(), "headers").listFiles()));
        Collections.sort(files);
        for (File f : files) {
            try (Scanner scanner = new Scanner(f, "UTF-8")) {
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine().trim();
                    if (line.startsWith("DECLARE_")) {
                        try {
                            int start = line.indexOf('(') + 1;
                            int end = line.indexOf(',');
                            if (end < start) {
                                end = line.indexOf(')');
                            }
                            String name = line.substring(start, end).trim();
                            opTemplates.add(name);
                        } catch(Exception e) {
                            throw new RuntimeException("Could not parse line from CustomOperations.h and headers: \"" + line + "\"", e);
                        }
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException("Could not parse CustomOperations.h and headers", e);
            }
        }
        logger.info("Ops found in CustomOperations.h and headers: " + opTemplates);
        String floatOps = "", halfOps = "", doubleOps = "";
        for (String t : opTemplates) {
            String s = "nd4j::ops::" + t;
            infoMap.put(new Info(s + "<float>").pointerTypes("float_" + t))
                   .put(new Info(s + "<float16>").pointerTypes("half_" + t))
                   .put(new Info(s + "<double>").pointerTypes("double_" + t));
            floatOps  += "\n        float_" + t + ".class,";
            halfOps   += "\n        half_" + t + ".class,";
            doubleOps += "\n        double_" + t + ".class,";
        }
        infoMap.put(new Info().javaText("\n"
                                      + "    Class[] floatOps = {" + floatOps + "};" + "\n"
                                      + "    Class[] halfOps = {" + halfOps + "};" + "\n"
                                      + "    Class[] doubleOps = {" + doubleOps + "};"));

        infoMap.put(new Info("nd4j::ops::OpRegistrator::updateMSVC").skip());
    }
}
