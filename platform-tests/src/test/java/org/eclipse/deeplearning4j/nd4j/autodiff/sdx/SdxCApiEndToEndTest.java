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
package org.eclipse.deeplearning4j.nd4j.autodiff.sdx;

import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end test of the SDX public C ABI ({@code dsp_runtime_c.h}) — the
 * contract every external language binding (Python, Rust, C#, Swift, Kotlin,
 * Java) programs against.
 *
 * Unlike {@link SdxRuntimeTest}, which exercises the underlying native DSP
 * plumbing through the JNI {@code NativeOps} surface, this test calls the
 * exported {@code sdx*} symbols through JNA — the same dlopen/dlsym route a
 * non-JVM consumer takes: sdxCreateRuntime → sdxLoadBundle(.sdz) →
 * sdxCreateContext → sdxRun → sdxGetExecutionReport / sdxFreezeShapes /
 * sdxGetPlanPhase / sdxGetExecutionCount → teardown.
 */
@Slf4j
@Tag("vulkan")
public class SdxCApiEndToEndTest extends BaseND4JTest {

    private static final int SDX_STATUS_OK = 0;
    private static final int SDX_STATUS_MODEL_LOAD_FAILED = 3;
    private static final int SDX_GPU_TARGET_VULKAN = 3;
    private static final int SDX_DEVICE_HOST = 0;
    /** sd::DataType::FLOAT32 */
    private static final int SDX_DTYPE_FLOAT = 5;

    private static SdxCApi api;
    private final List<File> tempFiles = new ArrayList<>();

    @Override
    public long getTimeoutMilliseconds() {
        return 5 * 60 * 1000L;
    }

    // ── JNA surface: the full 15-function C ABI ─────────────────────────────

    public interface SdxCApi extends Library {
        int sdxGetRuntimeAbiVersion();

        int sdxCreateRuntime(RuntimeOptions options, PointerByReference outRuntime);
        void sdxDestroyRuntime(Pointer runtime);

        int sdxLoadBundle(Pointer runtime, String bundlePath, ModelOptions options, PointerByReference outModel);
        void sdxUnloadModel(Pointer model);
        Pointer sdxGetTokenizerPath(Pointer model);
        Pointer sdxGetTextGenerationConfigPath(Pointer model);

        int sdxCreateContext(Pointer model, StringArray requestedOutputs, int numRequestedOutputs,
                             PointerByReference outContext);
        int sdxCreateContextWithOptions(Pointer model, StringArray requestedOutputs, int numRequestedOutputs,
                                        ContextOptions options, PointerByReference outContext);
        void sdxDestroyContext(Pointer context);

        int sdxRun(Pointer context, TensorView[] inputs, int numInputs,
                   TensorView[] outputs, int numOutputs, RunOptions options);
        int sdxRunAllocating(Pointer context, TensorView[] inputs, int numInputs,
                             RunOptions options);

        Pointer sdxGetLastError(Pointer runtime);
        int sdxGetExecutionReport(Pointer context, ExecutionReport outReport);

        int sdxMarkInputVariable(Pointer context, int inputIndex);
        int sdxMarkInputPlaceholder(Pointer context, int inputIndex);
        int sdxFreezeShapes(Pointer context);
        int sdxGetPlanPhase(Pointer context);
        int sdxGetExecutionCount(Pointer context);

        int sdxGetNumInputs(Pointer context);
        int sdxGetNumOutputs(Pointer context);
        Pointer sdxGetInputName(Pointer context, int inputIndex);
        Pointer sdxGetOutputName(Pointer context, int outputIndex);
        int sdxGetOutputTensor(Pointer context, int outputIndex,
                               TensorView outTensor);
    }

    public static class RuntimeOptions extends Structure {
        public int struct_size;

        public RuntimeOptions() {
            struct_size = size();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Collections.singletonList("struct_size");
        }
    }

    public static class ModelOptions extends Structure {
        public int struct_size;
        public int backend;
        public int strict_backend;
        public int allow_runtime_jit;
        public int gpu_target;

        public ModelOptions() {
            struct_size = size();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "backend", "strict_backend", "allow_runtime_jit", "gpu_target");
        }
    }

    public static class ContextOptions extends Structure {
        public int struct_size;
        public int bind_model_parameters;

        public ContextOptions() {
            struct_size = size();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "bind_model_parameters");
        }
    }

    public static class RunOptions extends Structure {
        public int struct_size;
        public int backend;
        public int strict_signature;
        public int gpu_target;

        public RunOptions() {
            struct_size = size();
            strict_signature = 1;
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "backend", "strict_signature", "gpu_target");
        }
    }

    public static class TensorView extends Structure {
        public Pointer data;
        public Pointer shape;
        public int rank;
        public int dtype;
        public long bytes;
        public int device_type;
        public int device_id;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("data", "shape", "rank", "dtype", "bytes", "device_type", "device_id");
        }
    }

    public static class ExecutionReport extends Structure {
        public int struct_size;
        public int requested_backend;
        public int applied_backend;
        public int status_code;
        public int used_fallback;
        public long execution_time_ns;
        public int requested_gpu_target;
        public int applied_gpu_target;
        public int plan_phase;
        public int execution_count;

        public ExecutionReport() {
            struct_size = size();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "requested_backend", "applied_backend", "status_code",
                    "used_fallback", "execution_time_ns", "requested_gpu_target", "applied_gpu_target",
                    "plan_phase", "execution_count");
        }
    }

    // ── Library resolution ───────────────────────────────────────────────────

    /**
     * The backend .so is already loaded in this process by ND4J via JavaCPP;
     * dlopen-ing the same absolute path just bumps its refcount, so JNA binds
     * to the identical, already-initialized library.
     */
    private static String resolveRuntimeLibrary() {
        String override = System.getProperty("org.nd4j.sdx.test.library");
        if (override != null && !override.isEmpty() && new File(override).exists()) {
            return override;
        }

        String backendName = Nd4j.getBackend().getClass().getName().toLowerCase();
        boolean cpu = backendName.contains("cpu");
        boolean vulkan = backendName.contains("vulkan");
        String libName = vulkan ? "libnd4jvulkan.so" : cpu ? "libnd4jcpu.so" : "libnd4jcuda.so";

        try {
            File cacheDir = org.bytedeco.javacpp.Loader.getCacheDir();
            if (cacheDir != null && cacheDir.isDirectory()) {
                try (Stream<Path> walk = Files.walk(cacheDir.toPath())) {
                    Path hit = walk.filter(p -> p.getFileName().toString().equals(libName))
                            .findFirst().orElse(null);
                    if (hit != null) {
                        return hit.toAbsolutePath().toString();
                    }
                }
            }
        } catch (Exception e) {
            log.warn("JavaCPP cache walk failed", e);
        }

        String chip = vulkan ? "vulkan" : cpu ? "cpu" : "cuda";
        File[] candidates = {
                new File("../libnd4j/blasbuild/" + chip + "/blas/" + libName).getAbsoluteFile(),
                new File("../libnd4j/blasbuild/" + chip + "/linux-x86_64/" + libName).getAbsoluteFile()
        };
        for (File candidate : candidates) {
            if (candidate.exists()) {
                return candidate.getAbsolutePath();
            }
        }
        return null;
    }

    @BeforeAll
    public static void loadApi() {
        Nd4j.create(1); // force backend init so the native library is materialized
        String libPath = resolveRuntimeLibrary();
        assumeTrue(libPath != null, "SDX runtime library not found (set -Dorg.nd4j.sdx.test.library=/path/to/lib)");
        log.info("Binding SDX C ABI from {}", libPath);
        api = Native.load(libPath, SdxCApi.class);
    }

    @AfterEach
    public void cleanupTempFiles() {
        for (File f : tempFiles) {
            try {
                Files.deleteIfExists(f.toPath());
            } catch (Exception ignored) {
            }
        }
        tempFiles.clear();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private File saveSdz(SameDiff sd, String prefix) throws Exception {
        File tempFile = File.createTempFile(prefix, ".sdz");
        tempFiles.add(tempFile);
        SDZSerializer.save(sd, tempFile, false, Collections.emptyMap());
        return tempFile;
    }

    /** Model with exactly one external input: output = x mmul w. */
    private SameDiff buildMatmulModel(INDArray w) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable wv = sd.var("w", w);
        sd.linalg.mmul("output", x, wv);
        return sd;
    }

    private static Memory floatsToMemory(float[] values) {
        Memory mem = new Memory((long) values.length * Float.BYTES);
        mem.write(0, values, 0, values.length);
        return mem;
    }

    private static Memory shapeToMemory(long[] shape) {
        Memory mem = new Memory((long) shape.length * Long.BYTES);
        mem.write(0, shape, 0, shape.length);
        return mem;
    }

    private static TensorView[] hostTensorArray(Memory[] dataPtrs, long[][] shapes, Memory[] shapeOwners) {
        TensorView[] views = (TensorView[]) new TensorView().toArray(dataPtrs.length);
        for (int i = 0; i < dataPtrs.length; i++) {
            shapeOwners[i] = shapeToMemory(shapes[i]);
            long elements = 1;
            for (long d : shapes[i]) elements *= d;
            views[i].data = dataPtrs[i];
            views[i].shape = shapeOwners[i];
            views[i].rank = shapes[i].length;
            views[i].dtype = SDX_DTYPE_FLOAT;
            views[i].bytes = elements * Float.BYTES;
            views[i].device_type = SDX_DEVICE_HOST;
            views[i].device_id = -1;
            views[i].write();
        }
        return views;
    }

    private String lastError(Pointer runtime) {
        Pointer err = api.sdxGetLastError(runtime);
        return err == null ? "" : err.getString(0);
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    @Test
    public void testAbiVersion() {
        assertEquals(1, api.sdxGetRuntimeAbiVersion(), "SDX_RUNTIME_ABI_VERSION drifted");
    }

    /**
     * Full lifecycle: load a real .sdz through the C ABI, run twice, freeze,
     * run again, and verify every output against the SameDiff reference.
     */
    @Test
    public void testEndToEndRunMatchesSameDiff() throws Exception {
        INDArray w = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape('c', 4, 2);
        SameDiff sd = buildMatmulModel(w);
        File sdzFile = saveSdz(sd, "sdx-capi-e2e-");

        PointerByReference outRuntime = new PointerByReference();
        RuntimeOptions runtimeOptions = new RuntimeOptions();
        runtimeOptions.write();
        assertEquals(SDX_STATUS_OK, api.sdxCreateRuntime(runtimeOptions, outRuntime));
        Pointer runtime = outRuntime.getValue();
        assertNotNull(runtime);

        try {
            PointerByReference outModel = new PointerByReference();
            ModelOptions modelOptions = new ModelOptions();
            // These positive execution tests load a raw .sdz without a precompiled
            // artifact bundle, so the development runtime must be allowed to compile.
            modelOptions.allow_runtime_jit = 1;
            modelOptions.write();
            int loadStatus = api.sdxLoadBundle(runtime, sdzFile.getAbsolutePath(), modelOptions, outModel);
            assertEquals(SDX_STATUS_OK, loadStatus, "sdxLoadBundle failed: " + lastError(runtime));
            Pointer model = outModel.getValue();
            assertNotNull(model);

            try {
                PointerByReference outContext = new PointerByReference();
                int ctxStatus = api.sdxCreateContext(model, new StringArray(new String[]{"output"}), 1, outContext);
                assertEquals(SDX_STATUS_OK, ctxStatus, "sdxCreateContext failed: " + lastError(runtime));
                Pointer context = outContext.getValue();
                assertNotNull(context);

                try {
                    assertEquals(0, api.sdxGetExecutionCount(context), "fresh context should report 0 executions");

                    // Discover the plan's external input contract via the ABI:
                    // externals cover variables AND placeholders (here: w, x).
                    int numInputs = api.sdxGetNumInputs(context);
                    assertEquals(2, numInputs, "plan externals should be {w, x}");
                    assertEquals(1, api.sdxGetNumOutputs(context));
                    String[] inputNames = new String[numInputs];
                    for (int i = 0; i < numInputs; i++) {
                        Pointer namePtr = api.sdxGetInputName(context, i);
                        assertNotNull(namePtr, "sdxGetInputName(" + i + ") returned null");
                        inputNames[i] = namePtr.getString(0);
                    }
                    assertTrue(Arrays.asList(inputNames).contains("x"), Arrays.toString(inputNames));
                    assertTrue(Arrays.asList(inputNames).contains("w"), Arrays.toString(inputNames));
                    assertNull(api.sdxGetInputName(context, 9999), "out-of-range name query must return null");

                    int totalRuns = 0;
                    for (int iter = 0; iter < 2; iter++) {
                        totalRuns++;
                        runAndVerify(context, runtime, sdzFile, w, inputNames, iter + 1, totalRuns);
                    }

                    assertEquals(SDX_STATUS_OK, api.sdxFreezeShapes(context),
                            "sdxFreezeShapes failed: " + lastError(runtime));
                    int frozenPhase = api.sdxGetPlanPhase(context);
                    assertTrue(frozenPhase >= 1,
                            "phase after freeze should be SHAPES_FROZEN(1)+, got: " + frozenPhase);

                    totalRuns++;
                    runAndVerify(context, runtime, sdzFile, w, inputNames, 7, totalRuns);

                    ExecutionReport report = new ExecutionReport();
                    report.write();
                    assertEquals(SDX_STATUS_OK, api.sdxGetExecutionReport(context, report));
                    report.read();
                    assertEquals(SDX_STATUS_OK, report.status_code);
                    assertEquals(totalRuns, report.execution_count, "report execution_count mismatch");
                    assertTrue(report.execution_time_ns > 0, "execution_time_ns should be measured");
                    assertTrue(report.plan_phase >= 1, "post-freeze report phase should be >= 1");
                    assertEquals(totalRuns, api.sdxGetExecutionCount(context));
                } finally {
                    api.sdxDestroyContext(context);
                }
            } finally {
                api.sdxUnloadModel(model);
            }
        } finally {
            api.sdxDestroyRuntime(runtime);
        }
    }

    /** Mobile inference contexts bind bundle-owned weights and expose only live inputs. */
    @Test
    public void testParameterBoundContextHidesWeightsAndRuns() throws Exception {
        INDArray w = Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f).reshape(4, 2);
        File sdzFile = saveSdz(buildMatmulModel(w), "sdx-capi-bound-");

        PointerByReference outRuntime = new PointerByReference();
        RuntimeOptions runtimeOptions = new RuntimeOptions();
        runtimeOptions.write();
        assertEquals(SDX_STATUS_OK, api.sdxCreateRuntime(runtimeOptions, outRuntime));
        Pointer runtime = outRuntime.getValue();

        try {
            PointerByReference outModel = new PointerByReference();
            ModelOptions modelOptions = new ModelOptions();
            // The mobile production path is no-JIT and bundle-backed. This focused
            // lifecycle test intentionally uses a raw .sdz while exercising weight binding.
            modelOptions.allow_runtime_jit = 1;
            modelOptions.write();
            assertEquals(SDX_STATUS_OK,
                    api.sdxLoadBundle(runtime, sdzFile.getAbsolutePath(), modelOptions, outModel),
                    "sdxLoadBundle failed: " + lastError(runtime));
            Pointer model = outModel.getValue();

            try {
                ContextOptions contextOptions = new ContextOptions();
                contextOptions.bind_model_parameters = 1;
                contextOptions.write();
                PointerByReference outContext = new PointerByReference();
                assertEquals(SDX_STATUS_OK,
                        api.sdxCreateContextWithOptions(model,
                                new StringArray(new String[]{"output"}), 1,
                                contextOptions, outContext),
                        "sdxCreateContextWithOptions failed: " + lastError(runtime));
                Pointer context = outContext.getValue();

                try {
                    assertEquals(1, api.sdxGetNumInputs(context),
                            "bundle-owned w must not be a public mobile input");
                    assertEquals("x", api.sdxGetInputName(context, 0).getString(0));
                    assertNull(api.sdxGetInputName(context, 1));

                    float[] x = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
                    Memory inputData = floatsToMemory(x);
                    Memory outputData = new Memory(4L * Float.BYTES);
                    Memory[] inputShapeOwners = new Memory[1];
                    Memory[] outputShapeOwners = new Memory[1];
                    TensorView[] inputs = hostTensorArray(
                            new Memory[]{inputData}, new long[][]{{2, 4}}, inputShapeOwners);
                    TensorView[] outputs = hostTensorArray(
                            new Memory[]{outputData}, new long[][]{{2, 2}}, outputShapeOwners);
                    RunOptions runOptions = new RunOptions();
                    runOptions.write();

                    assertEquals(SDX_STATUS_OK,
                            api.sdxRun(context, inputs, 1, outputs, 1, runOptions),
                            "parameter-bound sdxRun failed: " + lastError(runtime));
                    float[] actual = new float[4];
                    outputData.read(0, actual, 0, actual.length);
                    float[] expected = {50f, 60f, 114f, 140f};
                    assertArrayEquals(expected, actual, 1e-4f);

                    Pointer outputName = api.sdxGetOutputName(context, 0);
                    assertNotNull(outputName);
                    assertEquals("output", outputName.getString(0));
                    assertNull(api.sdxGetOutputName(context, 1));
                    assertNull(api.sdxGetTokenizerPath(model));
                    assertNull(api.sdxGetTextGenerationConfigPath(model));

                    assertEquals(SDX_STATUS_OK,
                            api.sdxRunAllocating(context, inputs, 1, runOptions),
                            "sdxRunAllocating failed: " + lastError(runtime));
                    TensorView dynamicOutput = new TensorView();
                    dynamicOutput.write();
                    assertEquals(SDX_STATUS_OK,
                            api.sdxGetOutputTensor(context, 0, dynamicOutput),
                            "sdxGetOutputTensor failed: " + lastError(runtime));
                    dynamicOutput.read();
                    assertEquals(2, dynamicOutput.rank);
                    assertArrayEquals(new long[]{2, 2},
                            dynamicOutput.shape.getLongArray(0, dynamicOutput.rank));
                    assertEquals(SDX_DTYPE_FLOAT, dynamicOutput.dtype);
                    assertEquals(4L * Float.BYTES, dynamicOutput.bytes);
                    assertEquals(SDX_DEVICE_HOST, dynamicOutput.device_type);
                    assertArrayEquals(expected,
                            dynamicOutput.data.getFloatArray(0, expected.length),
                            1e-4f);
                } finally {
                    api.sdxDestroyContext(context);
                }
            } finally {
                api.sdxUnloadModel(model);
            }
        } finally {
            api.sdxDestroyRuntime(runtime);
        }
    }

    private void runAndVerify(Pointer context, Pointer runtime, File sdzFile, INDArray w,
                              String[] inputNames, int seedScale, int expectedExecutionCount) throws Exception {
        INDArray x = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape('c', 2, 4).mul(seedScale);
        SameDiff reference = SDZSerializer.load(sdzFile, false);
        Map<String, INDArray> expectedMap =
                reference.output(Collections.singletonMap("x", x), Collections.singletonList("output"));
        float[] expected = expectedMap.get("output").dup('c').data().asFloat();

        // Bind every external by its discovered name, positionally in plan order.
        Map<String, INDArray> valueByName = new java.util.HashMap<>();
        valueByName.put("x", x);
        valueByName.put("w", w);
        int n = inputNames.length;
        Memory[] inputData = new Memory[n];
        long[][] inputShapes = new long[n][];
        for (int i = 0; i < n; i++) {
            INDArray value = valueByName.get(inputNames[i]);
            assertNotNull(value, "no test value for plan input '" + inputNames[i] + "'");
            inputData[i] = floatsToMemory(value.dup('c').data().asFloat());
            inputShapes[i] = value.shape();
        }
        Memory outputData = new Memory((long) expected.length * Float.BYTES);
        Memory[] inputShapeOwners = new Memory[n];
        Memory[] outputShapeOwner = new Memory[1];
        TensorView[] inputs = hostTensorArray(inputData, inputShapes, inputShapeOwners);
        TensorView[] outputs = hostTensorArray(new Memory[]{outputData}, new long[][]{{2, 2}}, outputShapeOwner);

        RunOptions runOptions = new RunOptions();
        runOptions.write();
        int runStatus = api.sdxRun(context, inputs, n, outputs, 1, runOptions);
        assertEquals(SDX_STATUS_OK, runStatus, "sdxRun failed: " + lastError(runtime));

        float[] actual = new float[expected.length];
        outputData.read(0, actual, 0, actual.length);
        assertArrayEquals(expected, actual, 1e-4f,
                "C ABI output diverged from SameDiff reference (run " + expectedExecutionCount + ")");
        assertEquals(expectedExecutionCount, api.sdxGetExecutionCount(context));
    }

    /** Input marking is part of the public ABI — must accept valid indices and reject bad ones. */
    @Test
    public void testInputMarkingApis() throws Exception {
        SameDiff sd = buildMatmulModel(Nd4j.createFromArray(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f).reshape(4, 2));
        File sdzFile = saveSdz(sd, "sdx-capi-mark-");

        PointerByReference outRuntime = new PointerByReference();
        RuntimeOptions runtimeOptions = new RuntimeOptions();
        runtimeOptions.write();
        assertEquals(SDX_STATUS_OK, api.sdxCreateRuntime(runtimeOptions, outRuntime));
        Pointer runtime = outRuntime.getValue();

        try {
            PointerByReference outModel = new PointerByReference();
            assertEquals(SDX_STATUS_OK,
                    api.sdxLoadBundle(runtime, sdzFile.getAbsolutePath(), null, outModel));
            Pointer model = outModel.getValue();

            try {
                PointerByReference outContext = new PointerByReference();
                assertEquals(SDX_STATUS_OK,
                        api.sdxCreateContext(model, new StringArray(new String[]{"output"}), 1, outContext));
                Pointer context = outContext.getValue();

                try {
                    assertEquals(SDX_STATUS_OK, api.sdxMarkInputVariable(context, 0),
                            "markInputVariable(0): " + lastError(runtime));
                    assertEquals(SDX_STATUS_OK, api.sdxMarkInputPlaceholder(context, 0),
                            "markInputPlaceholder(0): " + lastError(runtime));
                    assertNotEquals(SDX_STATUS_OK, api.sdxMarkInputVariable(context, 9999),
                            "out-of-range input index must be rejected");
                } finally {
                    api.sdxDestroyContext(context);
                }
            } finally {
                api.sdxUnloadModel(model);
            }
        } finally {
            api.sdxDestroyRuntime(runtime);
        }
    }

    /** Mobile Vulkan bundles must never silently lower kernels when runtime JIT is disabled. */
    @Test
    public void testPrecompiledOnlyVulkanRejectsBundleWithoutSpirvArtifacts() throws Exception {
        SameDiff sd = buildMatmulModel(Nd4j.createFromArray(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f).reshape(4, 2));
        File sdzFile = saveSdz(sd, "sdx-capi-vulkan-aot-");

        PointerByReference outRuntime = new PointerByReference();
        RuntimeOptions runtimeOptions = new RuntimeOptions();
        runtimeOptions.write();
        assertEquals(SDX_STATUS_OK, api.sdxCreateRuntime(runtimeOptions, outRuntime));
        Pointer runtime = outRuntime.getValue();

        try {
            ModelOptions modelOptions = new ModelOptions();
            modelOptions.allow_runtime_jit = 0;
            modelOptions.gpu_target = SDX_GPU_TARGET_VULKAN;
            modelOptions.write();

            PointerByReference outModel = new PointerByReference();
            int status = api.sdxLoadBundle(
                    runtime, sdzFile.getAbsolutePath(), modelOptions, outModel);
            assertEquals(SDX_STATUS_MODEL_LOAD_FAILED, status,
                    "precompiled-only Vulkan load must fail without bundle SPIR-V");
            assertTrue(lastError(runtime).contains("compiledArtifacts.vulkanSpirv"),
                    "error should identify the missing mobile artifact: " + lastError(runtime));
            assertNull(outModel.getValue());
        } finally {
            api.sdxDestroyRuntime(runtime);
        }
    }

    /** Error paths: bad bundle path must fail with a status and a readable error string. */
    @Test
    public void testLoadBundleErrorPath() {
        PointerByReference outRuntime = new PointerByReference();
        RuntimeOptions runtimeOptions = new RuntimeOptions();
        runtimeOptions.write();
        assertEquals(SDX_STATUS_OK, api.sdxCreateRuntime(runtimeOptions, outRuntime));
        Pointer runtime = outRuntime.getValue();

        try {
            PointerByReference outModel = new PointerByReference();
            int status = api.sdxLoadBundle(runtime, "/tmp/definitely-not-a-model-2026.sdz", null, outModel);
            assertNotEquals(SDX_STATUS_OK, status, "loading a nonexistent bundle must fail");
            assertFalse(lastError(runtime).isEmpty(), "sdxGetLastError should describe the failure");

            assertEquals(1, api.sdxCreateRuntime(null, null),
                    "null out_runtime must be SDX_STATUS_INVALID_ARGUMENT");
            assertEquals(-1, api.sdxGetPlanPhase(null));
            assertEquals(-1, api.sdxGetExecutionCount(null));
        } finally {
            api.sdxDestroyRuntime(runtime);
        }
    }
}
