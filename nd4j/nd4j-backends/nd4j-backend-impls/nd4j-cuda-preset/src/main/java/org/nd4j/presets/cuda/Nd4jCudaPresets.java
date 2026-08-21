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

package org.nd4j.presets.cuda;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.presets.OpExclusionUtils;
import org.nd4j.presets.SharedCompilerRuntime;

/**
 *
 * @author saudet
 */
@Properties(target = "org.nd4j.linalg.jcublas.bindings.Nd4jCuda", helper = "org.nd4j.presets.cuda.Nd4jCudaHelper",
        value = {@Platform(define = {"SD_ALL_OPS", "SD_CUDA", "SD_BACKEND_NAMESPACE sd_cuda"}, include = {
                //note, order matters here
                //this particular header file is either
                //going to be the source of ops, see also:
                //https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/blas/CMakeLists.txt#L76
                //https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/buildnativeoperations.sh#L517
                // Subsystem config classes + Environment.h MUST be listed BEFORE any
                // header that transitively #includes them. TadPack.h → NDArray.h →
                // DataTypeUtils.h → Environment.h → all config headers, which would
                // cause header guards to silently skip the explicit parse below.
                // DataType.h must come first because CoreConfig references sd::DataType.
                "array/DataType.h",
                "system/config/CoreConfig.h",
                "system/config/CudaDeviceConfig.h",
                "system/config/TritonConfig.h",
                "system/config/DspConfig.h",
                "system/config/LifecycleConfig.h",
                "system/config/MemoryConfig.h",
                "system/config/PrintConfig.h",
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
                "array/DataType.h",
                "graph/VariableType.h",
                "graph/ArgumentsList.h",
                "types/pair.h",
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
                "system/CudaLimitType.h",
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
                "array/TadPack.h",
                "helpers/DebugInfo.h",
                "ops/declarable/CustomOperations.h",
                "build_info.h",
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
                library = "jnind4jcuda", link = "nd4jcuda", preload = "libnd4jcuda"),
                @Platform(value = "linux", preload = "gomp@.1", preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/"},includepath = {"/usr/local/cuda/targets/x86_64-linux/include/"}),
                @Platform(value = "linux-armhf", preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
                @Platform(value = "linux-arm64", preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}),
                @Platform(value = "linux-ppc64", preloadpath = {"/usr/powerpc64-linux-gnu/lib/", "/usr/powerpc64le-linux-gnu/lib/", "/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
                @Platform(value = "windows", preload = {"libwinpthread-1", "libgcc_s_seh-1", "libgomp-1", "libstdc++-6", "libnd4jcpu"}),
                @Platform(extension = {"-cudnn", "-", "-compile", "-zluda",
                        "-zluda-rocm-7.2.4", "-zluda-rocm-6.2.4"}),
                @Platform(value = "linux",
                        extension = {"-zluda-rocm-7.2.4", "-zluda-rocm-6.2.4"},
                        resource = {"rocblas/library"})})
public class Nd4jCudaPresets implements LoadEnabled, BuildEnabled,InfoMapper {
    private static final String JAVACPP_PHASE_PROPERTY =
            "platform.nd4j.javacpp.phase";
    private static final String JAVACPP_PARSE_PHASE = "parse";
    private static final String CUDA_BINDINGS_RESOURCE_ROOT =
            "org/nd4j/linalg/jcublas/bindings/";
    private static final String ZLUDA_EXTENSION = "-zluda";

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;


    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;
    }

    @Override public void init(ClassProperties properties) {
        String platform = properties.getProperty("platform");
        List<String> preloads = properties.get("platform.preload");
        List<String> resources = properties.get("platform.preloadresource");

        // Maven properties are passed via System.getProperty during JavaCPP execution
        String calltraceProperty = System.getProperty(ND4JSystemProperties.LIBND4J_CALLTRACE, "OFF");
        boolean funcTrace = calltraceProperty.equalsIgnoreCase("ON");


        // Parsing only produces the platform-neutral Java binding now owned by
        // nd4j-cuda-backend-common. Native compilation and runtime loading still
        // require CMake's exact shared-runtime manifest.
        String javacppPhase = this.properties == null
                ? null
                : this.properties.getProperty(JAVACPP_PHASE_PROPERTY);
        if (javacppPhase == null) {
            javacppPhase = properties.getProperty(JAVACPP_PHASE_PROPERTY);
        }
        if (!JAVACPP_PARSE_PHASE.equals(javacppPhase)) {
            SharedCompilerRuntime.configure(
                    properties,
                    Nd4jCudaPresets.class,
                    CUDA_BINDINGS_RESOURCE_ROOT);
        }

        // Only apply the CUDA toolkit preloads at load time.
        if (!Loader.isLoadLibraries()) {
            return;
        }

        String extension = Loader.loadProperties().getProperty("platform.extension");
        if (extension == null || extension.isEmpty()) {
            extension = resolveBundledZludaExtension(
                    properties, platform, Nd4jCudaPresets.class.getClassLoader());
        }
        if (isZludaExtension(extension)) {
            String classifierResource = "/" + CUDA_BINDINGS_RESOURCE_ROOT
                    + platform + extension + "/";
            if (!resources.contains(classifierResource)) {
                // Extract the complete dependency closure before loading any member.
                // This lets DT_NEEDED dependencies resolve beside their consumers
                // even when the manifest's stable order is not dependency order.
                resources.add(0, classifierResource);
            }
            // The CUDA toolkit is a build-time toolchain for this classifier.
            // At runtime the manifest supplies ZLUDA's CUDA ABI implementations
            // and their AMD dependency closure, so never add Bytedeco's NVIDIA
            // platform resources or vendor preloads.
            return;
        }

        int i = sharedRuntimePreloadCount(preloads);

        // Add CUDA libraries to preload list with correct version suffixes for CUDA 12.x
        // Library version mapping (from /usr/local/cuda-12.9/lib64/):
        //   libcudart.so.12, libcublas.so.12, libcublasLt.so.12, libcusparse.so.12
        //   libcurand.so.10 (curand is still version 10), libcusolver.so.11 (cusolver is still version 11)
        String[] libs = {"cudart", "cublasLt", "cublas", "curand", "cusolver", "cusparse"};
        for (String lib : libs) {
            if (platform.startsWith("linux")) {
                // Version suffixes for CUDA 12.x
                if (lib.equals("curand")) {
                    lib += "@.10";  // curand is still version 10
                } else if (lib.equals("cusolver")) {
                    lib += "@.11";  // cusolver is still version 11
                } else {
                    lib += "@.12";  // cudart, cublas, cublasLt, cusparse use version 12
                }
            } else if (platform.startsWith("windows")) {
                // Windows version suffixes for CUDA 12.x
                if (lib.equals("curand")) {
                    lib += "64_10";
                } else if (lib.equals("cusolver")) {
                    lib += "64_11";
                } else {
                    lib += "64_12";
                }
            } else {
                continue; // no CUDA
            }
            if (!preloads.contains(lib)) {
                preloads.add(i++, lib);
            }
        }
        if (i > 0) {
            resources.add("/org/bytedeco/cuda/");
        }
    }

    static String resolveBundledZludaExtension(
            ClassProperties properties, String platform, ClassLoader resourceLoader) {
        if (platform == null || platform.isEmpty() || resourceLoader == null) {
            return null;
        }
        List<String> extensions = properties.get("platform.extension");
        for (int i = extensions.size() - 1; i >= 0; i--) {
            String extension = extensions.get(i);
            if (!isZludaExtension(extension)) {
                continue;
            }
            String manifestResource = CUDA_BINDINGS_RESOURCE_ROOT
                    + platform + extension + "/"
                    + SharedCompilerRuntime.MANIFEST_NAME;
            if (resourceLoader.getResource(manifestResource) != null) {
                return extension;
            }
        }
        return null;
    }

    static boolean isZludaExtension(String extension) {
        return ZLUDA_EXTENSION.equals(extension)
                || extension != null && extension.startsWith(ZLUDA_EXTENSION + "-");
    }

    private static int sharedRuntimePreloadCount(List<String> preloads) {
        int count = 0;
        while (count < preloads.size()
                && preloads.get(count).startsWith("nd4j_compiler_runtime_")) {
            count++;
        }
        return count;
    }

    @Override
    public void map(InfoMap infoMap) {
        //whether to include the SD_GCC_FUNCTRACE definition in the build. Not needed if we're not enabling the profiler.
        // Maven properties are passed via System.getProperty during JavaCPP execution
        String calltraceProperty = System.getProperty(ND4JSystemProperties.LIBND4J_CALLTRACE, "OFF");
        boolean funcTrace = calltraceProperty.equalsIgnoreCase("ON");

        logger.info("==============================================");
        logger.info("JavaCPP Preset (CUDA) - Functrace Configuration:");
        logger.info("  libnd4j.calltrace property: " + calltraceProperty);
        logger.info("  SD_GCC_FUNCTRACE will be: " + (funcTrace ? "DEFINED" : "UNDEFINED"));
        logger.info("==============================================");
        infoMap.put(new Info("thread_local", "SD_LIB_EXPORT", "SD_INLINE", "SD_TLS_EXPORT", "CUBLASWINAPI",
                        "SD_HOST", "SD_DEVICE", "SD_KERNEL", "SD_HOST_DEVICE", "SD_ALL_OPS", "NOT_EXCLUDED").cppTypes().annotations())
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
                .put(new Info("OpaqueContext").pointerTypes("org.nd4j.nativeblas.OpaqueContext"))
                .put(new Info("OpaqueWorkspace").cast().pointerTypes("Pointer"))
                // Workspace management functions - explicit javaText to ensure correct pointer semantics
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
                // Ensure RandomGenerator functions don't use @ByVal - they should return/accept pointers
                .put(new Info("createRandomGenerator").javaText(
                        "public native org.nd4j.nativeblas.OpaqueRandomGenerator createRandomGenerator(@Cast(\"sd::LongType\") long rootSeed, @Cast(\"sd::LongType\") long nodeSeed);"))
                .put(new Info("getGraphContextRandomGenerator").javaText(
                        "public native org.nd4j.nativeblas.OpaqueRandomGenerator getGraphContextRandomGenerator(org.nd4j.nativeblas.OpaqueContext ptr);"))
                .put(new Info("OpaqueLaunchContext").pointerTypes("org.nd4j.nativeblas.OpaqueLaunchContext"))
                .put(new Info("OpaqueDataBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueDataBuffer"))
                // Add @NoDeallocator to OpaqueDataBuffer-returning methods to prevent JavaCPP
                // from attaching a NativeDeallocator. ND4J's DeallocatorService manages buffer lifecycle.
                // Without this, JavaCPP's deallocator races with DeallocatorService causing use-after-free.
                .put(new Info("dbCreateExternalDataBuffer").javaText(
                        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"sd::Pointer\") Pointer primary, @Cast(\"sd::Pointer\") Pointer special);"))
                // This function marks the buffer constant in native code before returning to Java,
                // eliminating the race window between buffer creation and marking constant.
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
                .put (new Info("std::vector<std::string>","std::vector<std::string>*").cast().pointerTypes("PointerPointer"))

                .put(new Info("const char").valueTypes("byte").pointerTypes("@Cast(\"char*\") String",
                        "@Cast(\"char*\") BytePointer"))
                .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") BytePointer",
                        "@Cast(\"char*\") String"))
                .put(new Info("sd::Pointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                .put(new Info("CudaLimitType","sd::CudaLimitType").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                        "int[]"))
                .put(new Info("sd::LongType").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                        "long[]"))
                .put(new Info("sd::Status").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                        "int[]"))
                .put(new Info("sd::Unsigned").cast()
                        .valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                        "short[]"))
                .put(new Info("bfloat16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                        "short[]"));


        infoMap.put(funcTrace ? new Info("__CUDACC__", "MAX_UINT", "HAVE_MKLDNN", "__NEC__").define(false)
                        :  new Info("__CUDACC__", "MAX_UINT", "HAVE_MKLDNN", "__NEC__", "SD_GCC_FUNCTRACE").define(false))
                .put(funcTrace ? new Info("__JAVACPP_HACK__", "SD_ALL_OPS","__CUDABLAS__","SD_CUDA","SD_GCC_FUNCTRACE").define(true) :
                        new Info("__JAVACPP_HACK__", "SD_ALL_OPS","__CUDABLAS__","SD_CUDA").define(true))
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
                .put(new Info("SD_VALIDATE_PTR", "SD_VALIDATE_THIS", "SD_VALIDATE_ALIGNED", "SD_VALIDATE_MAGIC").cppText(""))
                .put(new Info("OpTraits").cast().valueTypes("int").pointerTypes("IntPointer"))
                .put(funcTrace ? new Info("std::initializer_list", "cnpy::NpyArray", "sd::NDArray::applyLambda", "sd::NDArray::applyPairwiseLambda",
                        "sd::graph::FlatResult",
                        "throwException",
                        "sd::graph::FlatVariable", "sd::NDArray::subarray", "std::shared_ptr", "sd::PointerWrapper",
                        "sd::PointerDeallocator").skip()
                        : new Info("std::initializer_list", "cnpy::NpyArray", "sd::NDArray::applyLambda", "sd::NDArray::applyPairwiseLambda",
                        "sd::graph::FlatResult",
                        "closeInstrumentOut",
                        "setInstrumentOut",
                        "instrumentFile",
                        "sd::graph::FlatVariable", "sd::NDArray::subarray", "std::shared_ptr", "sd::PointerWrapper",
                        "sd::PointerDeallocator").skip())
                .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")
                        .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
                .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
                .put(new Info("std::vector<std::vector<sd::LongType> >").pointerTypes("LongVectorVector").define())
                .put(new Info("std::vector<sd::NDArray*>").pointerTypes("NDArrayVector").define())
                .put(new Info("std::vector<const sd::NDArray*>").pointerTypes("ConstNDArrayVector").define())
                .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BooleanPointer", "boolean[]"))
                .put(new Info("Graph").pointerTypes("Pointer"))
                .put(new Info("sd::IndicesList").purify())
                .put(new Info("shape::cuMalloc").skip())
                .put(new Info("ErrorResult").skip())
                // Skip thread-local variables from DataBuffer.h — not callable from Java
                .put(new Info("sd::tl_graphExecutionActive", "sd::tl_captureWorkspace",
                        "sd::tl_captureWorkspaceSize", "sd::tl_captureWorkspaceOffset",
                        "sd::tl_capturedHostPtrs", "sd::tl_captureReplicateCache",
                        "sd::tl_graphCaptureStream").skip());

        OpExclusionUtils.processOps(logger, properties, infoMap);
        infoMap.put(new Info("sd::ops::OpRegistrator::updateMSVC").skip());
        //skip in case header definition not working
        infoMap.put(new Info("calculateOutputShapesNec").skip());
        infoMap.put(new Info("sd::ops::platforms::VersionProviderCallback",
                "sd::ops::platforms::VersionProviderRegistrar",
                "sd::ops::platforms::HelperVersionRegistry::registerProvider",
                "sd::ops::platforms::HelperVersionRegistry::getAllHelperInfo").skip());
        infoMap.put(new Info("sd::ops::platforms::HelperVersion::toString").javaNames("toVersionString"));
        infoMap.put(new Info("sd::ops::platforms::HelperInfo::getDetailedStatus").javaNames("getDetailedStatusString"));
        // Subsystem config classes — exposed to Java via env.triton(), env.dsp(), etc.
        // JavaCPP can't parse std::atomic<T>, so tell it to ignore those private members.
        // The public getter/setter methods return plain int/bool/int64_t and bind fine.
        infoMap.put(new Info("std::atomic<bool>", "std::atomic<int>", "std::atomic<int64_t>",
                "std::atomic<float>", "std::atomic<double>", "std::atomic<long>",
                "std::atomic<sd::DataType>", "std::atomic<size_t>").cast().skip());


    }


}
