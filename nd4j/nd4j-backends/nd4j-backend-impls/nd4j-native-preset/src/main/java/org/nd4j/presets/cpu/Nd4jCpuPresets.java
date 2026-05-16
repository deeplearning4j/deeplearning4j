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

package org.nd4j.presets.cpu;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.*;
import org.bytedeco.openblas.global.openblas;
import org.nd4j.presets.OpExclusionUtils;

import static org.nd4j.presets.OpExclusionUtils.getSkipClasses;

/**
 *
 * @author saudet
 */
@Properties(inherit = openblas.class, target = "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu", helper = "org.nd4j.presets.cpu.Nd4jCpuHelper",
        value = {@Platform(define = {"SD_ALL_OPS"}, include = {
                //note, order matters here
                //config.h MUST come first to define type availability macros (SD_SELECTIVE_TYPES, HAS_*)
                "config.h",
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
                "system/config/PrintConfig.h",
                "system/Environment.h",
                "generated/include_ops.h",
                "memory/MemoryType.h",
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
                "types/utf8string.h",
                "legacy/NativeOps.h",
                "dsp/NativeOpsDsp.h",
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
                "graph/Intervals.h",
                "graph/Stash.h",
                "graph/GraphState.h",
                "graph/VariableSpace.h",
                "helpers/helper_generator.h",
                "graph/profiling/GraphProfile.h",
                "graph/profiling/NodeProfile.h",
                "graph/Context.h",
                "graph/ContextPrototype.h",
                "helpers/shape.h",
                "helpers/OpArgsHolder.h",
                "array/ShapeList.h",
                "system/type_boilerplate.h",
                "system/op_boilerplate.h",
                "ops/InputType.h",
                "ops/declarable/OpDescriptor.h",
                "helpers/HelperVersionRegistry.h",
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
                        "cnpy/cnpy.h",

                },
                compiler = {"cpp17", "nowarnings"},
                library = "jnind4jcpu", link = "nd4jcpu", preload = "libnd4jcpu"),
                @Platform(value = "linux", link = {"nd4jcpu", "dl"}, preload = {"gomp@.1", "omp"}, preloadpath = {"/lib64/", "/lib/", "/usr/lib64/", "/usr/lib/"}),
                @Platform(value = "linux-armhf",preload = "gomp@.1", preloadpath = {"/usr/arm-linux-gnueabihf/lib/", "/usr/lib/arm-linux-gnueabihf/"}),
                @Platform(value = "linux-arm64",preload = "gomp@.1", preloadpath = {"/usr/aarch64-linux-gnu/lib/", "/usr/lib/aarch64-linux-gnu/"}),
                @Platform(value = "linux-ppc64", preloadpath = {"/usr/powerpc64-linux-gnu/lib/", "/usr/powerpc64le-linux-gnu/lib/", "/usr/lib/powerpc64-linux-gnu/", "/usr/lib/powerpc64le-linux-gnu/"}),
                @Platform(value = "windows", preload = {"libwinpthread-1", "libgcc_s_seh-1", "libgomp-1", "libstdc++-6", "libnd4jcpu"}),
                @Platform(value = "android-arm64",
                        preload = { "libnd4jcpu"}),

                @Platform(extension = {"-onednn", "-onednn-avx512","-onednn-avx2", "-","-avx2","-avx512", "-compat"})
        })
public class Nd4jCpuPresets implements InfoMapper, BuildEnabled {

    private Logger logger;
    private java.util.Properties properties;
    private String encoding;

    @Override
    public void init(Logger logger, java.util.Properties properties, String encoding) {
        this.logger = logger;
        this.properties = properties;
        this.encoding = encoding;

        // Only apply sanitizer configuration during build/link phase, not parser phase
        // During parser phase, config.h doesn't exist yet, so skip sanitizer setup
        String builderName = properties.getProperty("platform.builder", "");
        boolean isBuilderPhase = builderName != null && !builderName.isEmpty();

        // Check if sanitizers are enabled
        // Sanitizer flags are handled by the Maven POM configuration
        // No manual RPATH manipulation needed - clang's -fsanitize flag handles everything
    }

    @Override
    public void map(InfoMap infoMap) {
        //whether to include the SD_GCC_FUNCTRACE definition in the build. Not needed if we're not enabling the profiler.
        // Maven properties are passed via System.getProperty during JavaCPP execution
        String calltraceProperty = System.getProperty("libnd4j.calltrace", "OFF");
        boolean funcTrace = calltraceProperty.equalsIgnoreCase("ON");

        System.out.println("==============================================");
        System.out.println("JavaCPP Preset - Functrace Configuration:");
        System.out.println("  libnd4j.calltrace property: " + calltraceProperty);
        System.out.println("  SD_GCC_FUNCTRACE will be: " + (funcTrace ? "DEFINED" : "UNDEFINED"));
        System.out.println("==============================================");
        infoMap.put(new Info("thread_local", "SD_LIB_EXPORT", "SD_INLINE", "SD_TLS_EXPORT", "CUBLASWINAPI",
                        "SD_HOST", "SD_DEVICE", "SD_KERNEL", "SD_HOST_DEVICE", "SD_ALL_OPS", "NOT_EXCLUDED", "DEFAULT_ENGINE").cppTypes().annotations())
                .put(new Info("openblas_config.h", "cblas.h", "lapacke_config.h", "lapacke_mangling.h", "lapack.h", "lapacke.h", "lapacke_utils.h").skip())
                .put(new Info("NativeOps.h", "build_info.h").objectify())
                .put(new Info("OpaqueNDArray").pointerTypes("org.nd4j.nativeblas.OpaqueNDArray"))
                .put(new Info("OpaqueNDArrayArr").pointerTypes("org.nd4j.nativeblas.OpaqueNDArrayArr"))
               //android arm64

                .put(new Info("createOpaqueNDArray").javaNames("create"))
                .put(new Info("OpaqueTadPack").pointerTypes("org.nd4j.nativeblas.OpaqueTadPack"))
                .put(new Info("OpaqueShapeList").pointerTypes("org.nd4j.nativeblas.OpaqueShapeList"))
                .put(new Info("OpaqueVariablesSet").pointerTypes("org.nd4j.nativeblas.OpaqueVariablesSet"))
                .put(new Info("OpaqueVariable").pointerTypes("org.nd4j.nativeblas.OpaqueVariable"))
                .put(new Info("OpaqueConstantDataBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueConstantDataBuffer"))
                .put(new Info("OpaqueConstantShapeBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueConstantShapeBuffer"))
                .put(new Info("OpaqueConstantOffsetsBuffer").pointerTypes("org.nd4j.nativeblas.OpaqueConstantOffsetsBuffer"))
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
                .put (new Info("std::vector<std::string>","std::vector<std::string>*").cast().pointerTypes("PointerPointer"))
                .put(new Info("ExecTrace").pointerTypes("Pointer"))
                .put(new Info("std::vector<sd::ops::ExecTrace*>","OpExecTrace**")
                        .pointerTypes("org.nd4j.nativeblas.OpExecTraceVector"))
                .put(new Info("const char").valueTypes("byte").pointerTypes("@Cast(\"char*\") String",
                        "@Cast(\"char*\") BytePointer"))
                .put(new Info("char").valueTypes("char").pointerTypes("@Cast(\"char*\") BytePointer",
                        "@Cast(\"char*\") String"))
                .put(new Info("sd::Pointer").cast().valueTypes("Pointer").pointerTypes("PointerPointer"))
                .put(new Info("sd::LongType").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                        "long[]"))
                .put(new Info("sd::UnsignedLong").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer",
                        "long[]"))
                .put(new Info("sd::Status").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer",
                        "int[]"))
                .put(new Info("sd::Unsigned").cast()
                        .valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("float16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer",
                        "short[]"))
                .put(new Info("bfloat16").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))

                // Map types.h typedefs - these are the CANONICAL types used in generated instantiations
                .put(new Info("SignedChar").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
                .put(new Info("UnsignedChar").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
                .put(new Info("Int8Type").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
                .put(new Info("UInt8Type").cast().valueTypes("byte").pointerTypes("BytePointer", "ByteBuffer", "byte[]"))
                .put(new Info("Int16Type").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
                .put(new Info("UInt16Type").cast().valueTypes("short").pointerTypes("ShortPointer", "ShortBuffer", "short[]"))
                .put(new Info("Int32Type").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("UInt32Type").cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))
                .put(new Info("UInt64Type").cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"));

        infoMap.put(funcTrace ? new Info("__CUDACC__", "MAX_UINT", "HAVE_ONEDNN", "__CUDABLAS__", "__NEC__").define(false)
                        : new Info("__CUDACC__", "MAX_UINT", "HAVE_ONEDNN", "__CUDABLAS__", "__NEC__","SD_GCC_FUNCTRACE").define(false))
                .put(funcTrace ?  new Info("__JAVACPP_HACK__", "SD_ALL_OPS","SD_GCC_FUNCTRACE").define(true) :
                        new Info("__JAVACPP_HACK__", "SD_ALL_OPS").define(true))
                // Skip raw template class definitions from loop headers
                // JavaCPP should only see explicit instantiations from javacpp_instantiations.h
                .put(new Info("functions::scalar::ScalarTransform",
                        "functions::scalar::ScalarBoolTransform",
                        "functions::scalar::ScalarIntTransform",
                        "functions::pairwise_transforms::PairWiseTransform",
                        "functions::pairwise_transforms::PairWiseBoolTransform",
                        "functions::pairwise_transforms::PairWiseIntTransform",
                        "functions::broadcast::Broadcast",
                        "functions::broadcast::BroadcastBool",
                        "functions::broadcast::BroadcastInt",
                        "functions::transform::TransformAny",
                        "functions::transform::TransformBool",
                        "functions::transform::TransformFloat",
                        "functions::transform::TransformSame",
                        "functions::transform::TransformStrict",
                        "functions::reduce::ReduceFloatFunction",
                        "functions::reduce::ReduceSameFunction",
                        "functions::reduce::ReduceBoolFunction",
                        "functions::reduce::ReduceLongFunction",
                        "functions::reduce::Reduce3",
                        "functions::indexreduce::IndexReduce",
                        "functions::summarystats::SummaryStatsReduce",
                        "functions::random::RandomFunction").purify())
                .put(funcTrace ? new Info("std::initializer_list", "cnpy::NpyArray", "sd::NDArray::applyLambda", "sd::NDArray::applyPairwiseLambda",
                        "sd::graph::FlatResult",
                        "sd::graph::FlatVariable", "sd::NDArray::subarray", "std::shared_ptr", "sd::PointerWrapper",
                        "sd::PointerDeallocator").skip()
                        : new Info("std::initializer_list", "cnpy::NpyArray", "sd::NDArray::applyLambda", "sd::NDArray::applyPairwiseLambda",
                        "sd::graph::FlatResult",
                        "instrumentFile",
                        "setInstrumentOut",
                        "closeInstrumentOut",
                        "__cyg_profile_func_exit",
                        "__cyg_profile_func_enter",
                        "sd::graph::FlatVariable", "sd::NDArray::subarray", "std::shared_ptr", "sd::PointerWrapper",
                        "sd::PointerDeallocator").skip())
                .put(new Info("std::string").annotations("@StdString").valueTypes("BytePointer", "String")

                        .pointerTypes("@Cast({\"char*\", \"std::string*\"}) BytePointer"))
                .put(new Info("std::pair<int,int>").pointerTypes("IntIntPair").define())
                .put(new Info("std::vector<std::vector<sd::LongType> >").pointerTypes("LongVectorVector").define())
                .put(new Info("std::vector<const sd::NDArray*>").pointerTypes("ConstNDArrayVector").define())
                .put(new Info("std::vector<sd::NDArray*>").pointerTypes("NDArrayVector").define())
                .put(new Info("bool").cast().valueTypes("boolean").pointerTypes("BooleanPointer", "boolean[]"))
                .put(new Info("shape::cuMalloc").skip())
                .put(new Info("permutei").skip())
                .put(new Info(OpExclusionUtils.getPrefixedSkipOps()).skip())
                .put(new Info("sd::LaunchContext").skip())
                .put(new Info(OpExclusionUtils.getPrefixedShapeFunctions()).skip())
                .put(new Info(getSkipClasses()).skip())
                .put(new Info("ErrorResult").skip())
                // Skip thread-local variables from DataBuffer.h — not callable from Java
                .put(new Info("sd::tl_graphExecutionActive", "sd::tl_captureWorkspace",
                        "sd::tl_captureWorkspaceSize", "sd::tl_captureWorkspaceOffset",
                        "sd::tl_capturedHostPtrs", "sd::tl_captureReplicateCache",
                        "sd::tl_graphCaptureStream").skip());

        OpExclusionUtils.processOps(logger, properties, infoMap);


        infoMap.put(new Info("sd::ops::OpRegistrator::updateMSVC").skip());
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
