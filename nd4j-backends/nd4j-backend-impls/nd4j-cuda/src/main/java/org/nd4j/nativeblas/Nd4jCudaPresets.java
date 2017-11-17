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
@Properties(target = "org.nd4j.nativeblas.Nd4jCuda", value = @Platform(include = "NativeOps.h", compiler = "cpp11",
                library = "jnind4jcuda", link = "nd4jcuda", preload = "libnd4jcuda"))
public class Nd4jCudaPresets implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("thread_local", "ND4J_EXPORT", "INLINEDEF", "CUBLASWINAPI").cppTypes().annotations())
                        .put(new Info("NativeOps").base("org.nd4j.nativeblas.NativeOps"))
                        .put(new Info("const char", "char").valueTypes("char").pointerTypes("String",
                                        "@Cast(\"const char*\") BytePointer"))
                        .put(new Info("Nd4jPointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                        .put(new Info("Nd4jIndex").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                                        "long[]"))
                        .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                                        "short[]"));
    }
}
