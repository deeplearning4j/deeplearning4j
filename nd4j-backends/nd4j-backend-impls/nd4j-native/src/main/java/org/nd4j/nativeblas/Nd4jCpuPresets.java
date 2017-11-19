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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.BuildEnabled;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.javacpp.tools.Logger;

/**
 *
 * @author saudet
 */
@Properties(target = "org.nd4j.nativeblas.Nd4jCpu",
                value = {@Platform(include = {"NativeOps.h",
                                              "memory/Workspace.h",
                                              "indexing/NDIndex.h",
                                              "indexing/IndicesList.h",
                                              "array/DataType.h",
                                              "graph/VariableType.h",
                                              "NDArray.h",
                                              "array/NDArrayList.h",
                                              "array/ResultSet.h",
                                              "NDArrayFactory.h",
                                              "graph/Variable.h",
                                              "graph/FlowPath.h",
                                              "graph/Intervals.h",
                                              "graph/Stash.h",
                                              "graph/VariableSpace.h",
                                              "helpers/helper_generator.h",
                                              "graph/Context.h",
                                              "graph/ContextPrototype.h",
                                              "helpers/shape.h",
                                              "array/ShapeList.h",
                                              "op_boilerplate.h",
                                              "ops/InputType.h",
                                              "ops/declarable/OpDescriptor.h",
                                              "ops/declarable/DeclarableOp.h",
                                              "ops/declarable/DeclarableListOp.h",
                                              "ops/declarable/DeclarableReductionOp.h",
                                              "ops/declarable/DeclarableCustomOp.h",
                                              "ops/declarable/BooleanOp.h",
                                              "ops/declarable/LogicOp.h",
                                              "ops/declarable/OpRegistrator.h",
                                              "ops/declarable/CustomOperations.h"},
                                compiler = "cpp11", library = "jnind4jcpu", link = "nd4jcpu", preload = "libnd4jcpu"),
                                @Platform(value = "linux", preload = "gomp@.1",
                                                preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/",
                                                                "/usr/lib/powerpc64-linux-gnu/",
                                                                "/usr/lib/powerpc64le-linux-gnu/"})})
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
        infoMap.put(new Info("thread_local", "ND4J_EXPORT", "INLINEDEF", "CUBLASWINAPI", "FORCEINLINE").cppTypes().annotations())
                        .put(new Info("NativeOps").base("org.nd4j.nativeblas.NativeOps"))
                        .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") String",
                                        "@Cast(\"char*\") BytePointer"))
                        .put(new Info("Nd4jPointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                        .put(new Info("Nd4jIndex").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                                        "long[]"))
                        .put(new Info("Nd4jStatus").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                                        "int[]"))
                        .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                                        "short[]"));

        infoMap.put(new Info("__CUDACC__").define(false))
               .put(new Info("__JAVACPP_HACK__").define(true))
               .put(new Info("MAX_UINT").translate(false))
               .put(new Info("std::initializer_list", "cnpy::NpyArray", "nd4j::NDArray::applyLambda", "nd4j::NDArray::applyPairwiseLambda",
                             "nd4j::graph::FlatResult", "nd4j::graph::FlatVariable").skip())
               .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")
                                           .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
               .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
               .put(new Info("std::vector<nd4j::NDArray<float>*>").pointerTypes("FloatNDArrayVector").define())
               .put(new Info("std::vector<nd4j::NDArray<float16>*>").pointerTypes("HalfNDArrayVector").define())
               .put(new Info("std::vector<nd4j::NDArray<double>*>").pointerTypes("DoubleNDArrayVector").define())
               .put(new Info("nd4j::IndicesList").purify());

        String classTemplates[] = {
                "nd4j::NDArray",
                "nd4j::NDArrayList",
                "nd4j::ResultSet",
                "nd4j::NDArrayFactory",
                "nd4j::graph::Variable",
                "nd4j::graph::Stash",
                "nd4j::graph::VariableSpace",
                "nd4j::graph::Context",
                "nd4j::graph::ContextPrototype",
                "nd4j::ops::DeclarableOp",
                "nd4j::ops::DeclarableListOp",
                "nd4j::ops::DeclarableReductionOp",
                "nd4j::ops::DeclarableCustomOp",
                "nd4j::ops::BooleanOp",
                "nd4j::ops::LogicOp"};
        for (String t : classTemplates) {
            String s = t.substring(t.lastIndexOf(':') + 1);
            infoMap.put(new Info(t + "<float>").pointerTypes("Float" + s))
                   .put(new Info(t + "<float16>").pointerTypes("Half" + s))
                   .put(new Info(t + "<double>").pointerTypes("Double" + s));
        }

        // pick up custom operations automatically from CustomOperations.h in libnd4j
        String separator = properties.getProperty("platform.path.separator");
        String[] includePaths = properties.getProperty("platform.includepath").split(separator);
        File file = null;
        for (String path : includePaths) {
            file = new File(path, "ops/declarable/CustomOperations.h");
            if (file.exists()) {
                break;
            }
        }
        ArrayList<String> opTemplates = new ArrayList<String>();
        try (Scanner scanner = new Scanner(file)) {
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
                        throw new RuntimeException("Could not parse line from CustomOperations.h: \"" + line + "\"", e);
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not parse CustomOperations.h", e);
        }
        logger.info("Ops found in CustomOperations.h: " + opTemplates);
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
