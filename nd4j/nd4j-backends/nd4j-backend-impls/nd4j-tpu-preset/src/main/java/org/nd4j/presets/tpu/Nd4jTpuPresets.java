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

package org.nd4j.presets.tpu;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;
import org.nd4j.presets.OpExclusionUtils;

/**
 * JavaCPP Presets for ND4J TPU backend using PJRT.
 *
 * This class defines the native library bindings for TPU operations
 * through Google's Portable Runtime (PJRT) API.
 */
@Properties(target = "org.nd4j.linalg.jtpu.bindings.Nd4jTpu", helper = "org.nd4j.presets.tpu.Nd4jTpuHelper",
        value = {@Platform(define = {"SD_ALL_OPS", "SD_TPU", "HAVE_PJRT", "SD_BACKEND_NAMESPACE sd_tpu"}, include = {
                // Keep this order aligned with the maintained CPU/Vulkan presets.
                "config.h",
                "array/DataType.h",
                "system/config/CoreConfig.h",
                "system/config/CudaDeviceConfig.h",
                "system/config/TritonConfig.h",
                "system/config/DspConfig.h",
                "system/config/LifecycleConfig.h",
                "system/config/MemoryConfig.h",
                "system/config/PrintConfig.h",
                "system/config/VulkanConfig.h",
                "system/Environment.h",
                "generated/include_ops.h",
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
                "graph/GraphState.h",
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
                "system/type_boilerplate.h",
                "system/op_boilerplate.h",
                "ops/InputType.h",
                "ops/declarable/OpDescriptor.h",
                "helpers/HelperVersionRegistry.h",
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
                "build_info.h"
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
                compiler = {"cpp17", "nowarnings"},
                library = "jnind4jtpu", link = "nd4jtpu", preload = "libnd4jtpu"),
                @Platform(value = "linux", link = {"nd4jtpu", "dl"}, preload = "gomp@.1", preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/"}),
                @Platform(value = "linux-arm64", preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"})})
public class Nd4jTpuPresets implements LoadEnabled, BuildEnabled, InfoMapper {
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

        // TPU is primarily supported on Linux x86_64 and arm64
        String[] defaultLibs = {"nd4jtpu"};

        for (String lib : defaultLibs) {
            if (platform.startsWith("linux")) {
                lib = lib + (lib.indexOf('@') < 0 ? "" : lib.substring(lib.indexOf('@')));
                if (!preloads.contains(lib)) {
                    preloads.add(lib);
                }
            }
        }

        // The selected PJRT plugin is loaded by PjrtClientManager after backend
        // discovery. Never preload a generic plugin into the JNI wrapper.
    }

    @Override
    public void map(InfoMap infoMap) {
        infoMap.put(new Info("thread_local", "SD_LIB_EXPORT", "SD_LIB_HIDDEN", "SD_INLINE",
                        "SD_TLS_EXPORT", "SD_HOST", "SD_DEVICE", "SD_HOST_DEVICE", "SD_KERNEL",
                        "SD_ALL_OPS", "SD_TPU", "HAVE_PJRT", "NOT_EXCLUDED", "DEFAULT_ENGINE")
                        .cppTypes().annotations())
                .put(new Info("NativeOps.h", "build_info.h").objectify())
                .put(new Info("OpaqueNDArray").pointerTypes("org.nd4j.nativeblas.OpaqueNDArray"))
                .put(new Info("OpaqueNDArrayArr").pointerTypes("org.nd4j.nativeblas.OpaqueNDArrayArr"))
                .put(new Info("createOpaqueNDArray").javaNames("create"))
                .put(new Info("OpaqueTadPack").pointerTypes("org.nd4j.nativeblas.OpaqueTadPack"))
                .put(new Info("OpaqueShapeList").pointerTypes("org.nd4j.nativeblas.OpaqueShapeList"))
                .put(new Info("OpaqueVariablesSet").pointerTypes("org.nd4j.nativeblas.OpaqueVariablesSet"))
                .put(new Info("OpaqueVariable").pointerTypes("org.nd4j.nativeblas.OpaqueVariable"))
                .put(new Info("OpaqueConstantDataBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueConstantDataBuffer"))
                .put(new Info("OpaqueConstantShapeBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueConstantShapeBuffer"))
                .put(new Info("OpaqueConstantOffsetsBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueConstantOffsetsBuffer"))
                .put(new Info("OpaqueDataBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueDataBuffer"))
                .put(new Info("dbCreateExternalDataBuffer").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"sd::Pointer\") Pointer primary, @Cast(\"sd::Pointer\") Pointer special);"))
                .put(new Info("dbCreateConstantExternalDataBuffer").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateConstantExternalDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"sd::Pointer\") Pointer primary, @Cast(\"sd::Pointer\") Pointer special);"))
                .put(new Info("dbAllocateDataBuffer").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbAllocateDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"bool\") boolean allocateBoth);"))
                .put(new Info("allocateDataBuffer").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer allocateDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"bool\") boolean allocateBoth);"))
                .put(new Info("dbCreateView").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateView(org.nd4j.nativeblas.OpaqueDataBuffer dataBuffer, @Cast(\"sd::LongType\") long length);"))
                .put(new Info("intermediateResultDataAt").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer intermediateResultDataAt(int index, org.nd4j.nativeblas.OpaqueContext contextPointer);"))
                .put(new Info("OpaqueContext").pointerTypes("org.nd4j.nativeblas.OpaqueContext"))
                .put(new Info("OpaqueWorkspace").cast().pointerTypes("Pointer"))
                .put(new Info("createNativeWorkspace").javaText(
                        "public native @Cast(\"OpaqueWorkspace\") Pointer createNativeWorkspace(@Cast(\"sd::LongType\") long initialSize);"))
                .put(new Info("destroyNativeWorkspace").javaText(
                        "public native void destroyNativeWorkspace(@Cast(\"OpaqueWorkspace\") Pointer workspace);"))
                .put(new Info("workspaceScopeIn").javaText(
                        "public native void workspaceScopeIn(@Cast(\"OpaqueWorkspace\") Pointer workspace);"))
                .put(new Info("workspaceScopeOut").javaText(
                        "public native void workspaceScopeOut(@Cast(\"OpaqueWorkspace\") Pointer workspace);"))
                .put(new Info("attachWorkspaceToContext").javaText(
                        "public native void attachWorkspaceToContext(org.nd4j.nativeblas.OpaqueContext ctx, @Cast(\"OpaqueWorkspace\") Pointer workspace);"))
                .put(new Info("detachWorkspaceFromContext").javaText(
                        "public native void detachWorkspaceFromContext(org.nd4j.nativeblas.OpaqueContext ctx);"))
                .put(new Info("getWorkspaceCurrentOffset").javaText(
                        "public native @Cast(\"sd::LongType\") long getWorkspaceCurrentOffset(@Cast(\"OpaqueWorkspace\") Pointer workspace);"))
                .put(new Info("getWorkspaceAllocatedSize").javaText(
                        "public native @Cast(\"sd::LongType\") long getWorkspaceAllocatedSize(@Cast(\"OpaqueWorkspace\") Pointer workspace);"))
                .put(new Info("OpaqueMultiBackendWorkspace").cast().pointerTypes("Pointer"))
                .put(new Info("OpaqueRandomGenerator").pointerTypes("org.nd4j.nativeblas.OpaqueRandomGenerator"))
                .put(new Info("createRandomGenerator").javaText(
                        "public native org.nd4j.nativeblas.OpaqueRandomGenerator createRandomGenerator(@Cast(\"sd::LongType\") long rootSeed, @Cast(\"sd::LongType\") long nodeSeed);"))
                .put(new Info("getGraphContextRandomGenerator").javaText(
                        "public native org.nd4j.nativeblas.OpaqueRandomGenerator getGraphContextRandomGenerator(org.nd4j.nativeblas.OpaqueContext ptr);"))
                .put(new Info("OpaqueLaunchContext").pointerTypes("org.nd4j.nativeblas.OpaqueLaunchContext"))
                .put(new Info("std::vector<std::string>", "std::vector<std::string>*")
                        .cast().pointerTypes("PointerPointer"))
                .put(new Info("ExecTrace").pointerTypes("Pointer"))
                .put(new Info("std::vector<sd::ops::ExecTrace*>", "OpExecTrace**")
                        .pointerTypes("org.nd4j.nativeblas.OpExecTraceVector"))
                .put(new Info("sd::Pointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                .put(new Info("sd::LongType", "sd::UnsignedLong").cast().valueTypes("long")
                        .pointerTypes("LongPointer", "LongBuffer", "long[]"))
                .put(new Info("sd::Status", "sd::Unsigned").cast().valueTypes("int")
                        .pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("float16", "bfloat16").cast().valueTypes("short")
                        .pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
                .put(new Info("SignedChar", "UnsignedChar", "Int8Type", "UInt8Type").cast()
                        .valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
                .put(new Info("Int16Type", "UInt16Type").cast().valueTypes("short")
                        .pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
                .put(new Info("Int32Type", "UInt32Type").cast().valueTypes("int")
                        .pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("UInt64Type").cast().valueTypes("long")
                        .pointerTypes("LongPointer", "LongBuffer", "long[]"))
                .put(new Info("const char").valueTypes("byte")
                        .pointerTypes("@Cast(\"char*\") String", "@Cast(\"char*\") BytePointer"))
                .put(new Info("char").valueTypes("char")
                        .pointerTypes("@Cast(\"char*\") BytePointer", "@Cast(\"char*\") String"))
                .put(new Info("std::string").annotations("@StdString")
                        .valueTypes("BytePointer", "String")
                        .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
                .put(new Info("std::initializer_list", "cnpy::NpyArray", "cnpy::npz_t",
                        "sd::PointerWrapper", "sd::PointerDeallocator").skip())
                .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
                .put(new Info("std::vector<std::vector<sd::LongType> >")
                        .pointerTypes("LongVectorVector").define())
                .put(new Info("std::vector<const sd::NDArray*>").pointerTypes("ConstNDArrayVector").define())
                .put(new Info("std::vector<sd::NDArray*>").pointerTypes("NDArrayVector").define())
                .put(new Info("bool").cast().valueTypes("boolean")
                        .pointerTypes("BooleanPointer", "boolean[]"))
                .put(new Info("__JAVACPP_HACK__", "SD_ALL_OPS", "SD_TPU", "HAVE_PJRT").define(true))
                .put(new Info("__CUDACC__", "__CUDABLAS__", "__NEC__", "HAVE_ONEDNN").define(false))
                .put(new Info("SD_PADDED_NEW_DELETE").cppText("#define SD_PADDED_NEW_DELETE"))
                .put(new Info("SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN")
                        .cppText("#define SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN"))
                .put(new Info("SD_BACKEND_ROOT_INLINE_NAMESPACE_END")
                        .cppText("#define SD_BACKEND_ROOT_INLINE_NAMESPACE_END"))
                .put(new Info("SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN")
                        .cppText("#define SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN"))
                .put(new Info("SD_BACKEND_OPS_INLINE_NAMESPACE_END")
                        .cppText("#define SD_BACKEND_OPS_INLINE_NAMESPACE_END"))
                .put(new Info("SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN")
                        .cppText("#define SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN"))
                .put(new Info("SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END")
                        .cppText("#define SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END"))
                .put(new Info("SD_DECLARABLE_OP_EXECUTION_METHODS")
                        .cppText("#define SD_DECLARABLE_OP_EXECUTION_METHODS"))
                .put(new Info("SD_VALIDATE_PTR", "SD_VALIDATE_THIS", "SD_VALIDATE_ALIGNED",
                        "SD_VALIDATE_MAGIC").cppText(""));

        infoMap.put(new Info("sd::ops::platforms::HelperVersion::toString")
                .javaNames("toVersionString"));
        infoMap.put(new Info("sd::ops::platforms::HelperInfo::getDetailedStatus")
                .javaNames("getDetailedStatusString"));
        infoMap.put(new Info("sd::ops::platforms::VersionProviderCallback",
                "sd::ops::platforms::VersionProviderRegistrar",
                "sd::ops::platforms::HelperVersionRegistry::registerProvider",
                "sd::ops::platforms::HelperVersionRegistry::getAllHelperInfo",
                "sd::graph::GraphState::getScope",
                "sd::graph::GraphState::graph").skip());
        infoMap.put(new Info("sd::ops::platforms::HelperCapability").cast()
                .valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"));
        infoMap.put(new Info("std::atomic<bool>", "std::atomic<int>",
                "std::atomic<int64_t>", "std::atomic<float>",
                "std::atomic<double>", "std::atomic<long>",
                "std::atomic<sd::DataType>", "std::atomic<size_t>").cast().skip());
        infoMap.put(new Info("sd::config::VulkanConfig",
                "sd::Environment::vulkan").skip());
        infoMap.put(new Info("sd::ops::OpRegistrator::updateMSVC",
                "shape::cuMalloc", "permutei").skip());

        OpExclusionUtils.processOps(logger, properties, infoMap);
    }
}
