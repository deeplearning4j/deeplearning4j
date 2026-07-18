/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.dsp.runtime;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.dsp.runtime.bindings.SdxNative;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_context_options_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_context_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_execution_report_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_model_options_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_model_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_run_options_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_runtime_options_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_runtime_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_tensor_view_t;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Objects;

/**
 * Shared Java session API for the stable SDX C runtime ABI.
 *
 * <p>JavaCPP is transport only. Bundle loading, model-parameter binding,
 * tensor ownership, backend selection, execution reporting, and AOT-only
 * enforcement remain in {@code dsp_runtime_c.h} and its libnd4j
 * implementation. Platform packaging selects the backend linked by the
 * generated JNI shim: Vulkan on Android, Metal on iOS, and CPU by default on
 * desktop.</p>
 *
 * <p>Models must be closed after their contexts and text-generation sessions,
 * and the runtime must be closed after its models. The wrapper enforces that
 * order to prevent native use-after-free bugs.</p>
 */
public final class SdxRuntime implements AutoCloseable {

    public static final int SDX_STATUS_OK = SdxNative.SDX_STATUS_OK;

    public static final int SDX_BACKEND_AUTO = SdxNative.SDX_BACKEND_AUTO;
    public static final int SDX_BACKEND_SLOT_BY_SLOT = SdxNative.SDX_BACKEND_SLOT_BY_SLOT;
    public static final int SDX_BACKEND_CUDA_GRAPHS = SdxNative.SDX_BACKEND_CUDA_GRAPHS;
    public static final int SDX_BACKEND_NVRTC = SdxNative.SDX_BACKEND_NVRTC;
    public static final int SDX_BACKEND_PTX = SdxNative.SDX_BACKEND_PTX;
    public static final int SDX_BACKEND_TRITON = SdxNative.SDX_BACKEND_TRITON;
    public static final int SDX_BACKEND_MLX = SdxNative.SDX_BACKEND_MLX;
    public static final int SDX_BACKEND_ARM_HYBRID = SdxNative.SDX_BACKEND_ARM_HYBRID;
    public static final int SDX_BACKEND_NNAPI = SdxNative.SDX_BACKEND_NNAPI;
    public static final int SDX_BACKEND_HIP_GRAPHS = SdxNative.SDX_BACKEND_HIP_GRAPHS;
    public static final int SDX_BACKEND_LEVEL_ZERO = SdxNative.SDX_BACKEND_LEVEL_ZERO;
    public static final int SDX_BACKEND_VULKAN = SdxNative.SDX_BACKEND_VULKAN;
    public static final int SDX_BACKEND_METAL = SdxNative.SDX_BACKEND_METAL;
    public static final int SDX_BACKEND_TPU = SdxNative.SDX_BACKEND_TPU;
    public static final int SDX_BACKEND_HEXAGON = SdxNative.SDX_BACKEND_HEXAGON;

    public static final int SDX_DEVICE_HOST = SdxNative.SDX_DEVICE_HOST;
    public static final int SDX_DEVICE_CUDA = SdxNative.SDX_DEVICE_CUDA;
    public static final int SDX_DEVICE_AMD = SdxNative.SDX_DEVICE_AMD;

    public static final int SDX_GPU_TARGET_AUTO = SdxNative.SDX_GPU_TARGET_AUTO;
    public static final int SDX_GPU_TARGET_CUDA = SdxNative.SDX_GPU_TARGET_CUDA;
    public static final int SDX_GPU_TARGET_AMD = SdxNative.SDX_GPU_TARGET_AMD;
    public static final int SDX_GPU_TARGET_VULKAN = SdxNative.SDX_GPU_TARGET_VULKAN;
    public static final int SDX_GPU_TARGET_METAL = SdxNative.SDX_GPU_TARGET_METAL;

    public static final class RuntimeOptions {
        public int struct_size = Integer.BYTES;
    }

    public static final class ModelOptions {
        public int struct_size = 5 * Integer.BYTES;
        public int backend = SDX_BACKEND_AUTO;
        public int strict_backend;
        /** Runtime compilation is opt-in. Mobile bundles remain AOT-only by default. */
        public int allow_runtime_jit;
        public int gpu_target = SDX_GPU_TARGET_AUTO;

        public static ModelOptions mobileVulkan() {
            return new ModelOptions()
                    .backend(SDX_BACKEND_VULKAN)
                    .strictBackend(true)
                    .allowRuntimeJit(false)
                    .gpuTarget(SDX_GPU_TARGET_VULKAN);
        }

        public static ModelOptions mobileMetal() {
            return new ModelOptions()
                    .backend(SDX_BACKEND_METAL)
                    .strictBackend(true)
                    .allowRuntimeJit(false)
                    .gpuTarget(SDX_GPU_TARGET_METAL);
        }

        /** Qualcomm Hexagon/HTP with shape-keyed kernels from the SDX bundle. */
        public static ModelOptions mobileHexagon() {
            return new ModelOptions()
                    .backend(SDX_BACKEND_HEXAGON)
                    .strictBackend(true)
                    .allowRuntimeJit(false)
                    .gpuTarget(SDX_GPU_TARGET_AUTO);
        }

        public ModelOptions backend(int value) {
            backend = value;
            return this;
        }

        public ModelOptions strictBackend(boolean value) {
            strict_backend = value ? 1 : 0;
            return this;
        }

        public ModelOptions allowRuntimeJit(boolean value) {
            allow_runtime_jit = value ? 1 : 0;
            return this;
        }

        public ModelOptions gpuTarget(int value) {
            gpu_target = value;
            return this;
        }
    }

    public static final class ContextOptions {
        public int struct_size = 2 * Integer.BYTES;
        public int bind_model_parameters;

        public static ContextOptions inference() {
            ContextOptions options = new ContextOptions();
            options.bind_model_parameters = 1;
            return options;
        }
    }

    public static final class RunOptions {
        public int struct_size = 4 * Integer.BYTES;
        public int backend = SDX_BACKEND_AUTO;
        public int strict_signature = 1;
        public int gpu_target = SDX_GPU_TARGET_AUTO;

        public static RunOptions mobileVulkan() {
            return new RunOptions()
                    .backend(SDX_BACKEND_VULKAN)
                    .gpuTarget(SDX_GPU_TARGET_VULKAN);
        }

        public static RunOptions mobileMetal() {
            return new RunOptions()
                    .backend(SDX_BACKEND_METAL)
                    .gpuTarget(SDX_GPU_TARGET_METAL);
        }

        public static RunOptions mobileHexagon() {
            return new RunOptions()
                    .backend(SDX_BACKEND_HEXAGON)
                    .gpuTarget(SDX_GPU_TARGET_AUTO);
        }

        public RunOptions backend(int value) {
            backend = value;
            return this;
        }

        public RunOptions strictSignature(boolean value) {
            strict_signature = value ? 1 : 0;
            return this;
        }

        public RunOptions gpuTarget(int value) {
            gpu_target = value;
            return this;
        }
    }

    /**
     * Java-owned description of an SDX tensor view. The data pointer is never
     * freed by this class. Shapes created by {@link #tensor} are kept alive by
     * the view; output shapes are borrowed from the context until its next run.
     */
    public static final class TensorView {
        public Pointer data;
        public LongPointer shape;
        public int rank;
        public int dtype;
        public long bytes;
        public int device_type = SDX_DEVICE_HOST;
        public int device_id = -1;

        private LongPointer shapeOwner;

        public static TensorView hostTensor(Pointer data, long[] shapeValues, int dtype, long bytes) {
            return tensor(data, shapeValues, dtype, bytes, SDX_DEVICE_HOST, -1);
        }

        public static TensorView tensor(
                Pointer data,
                long[] shapeValues,
                int dtype,
                long bytes,
                int deviceType,
                int deviceId) {
            Objects.requireNonNull(data, "data");
            Objects.requireNonNull(shapeValues, "shapeValues");
            TensorView view = new TensorView();
            view.data = data;
            view.rank = shapeValues.length;
            view.dtype = dtype;
            view.bytes = bytes;
            view.device_type = deviceType;
            view.device_id = deviceId;
            if (shapeValues.length > 0) {
                view.shapeOwner = new LongPointer(shapeValues);
                view.shape = view.shapeOwner;
            }
            return view;
        }

        /** Copy the current native shape into JVM-owned storage. */
        public long[] shapeValues() {
            if (rank <= 0 || Pointer.isNull(shape)) {
                return new long[0];
            }
            long[] values = new long[rank];
            shape.get(values);
            return values;
        }

        private void writeTo(sdx_tensor_view_t nativeView) {
            nativeView
                    .data(data)
                    .shape(shape)
                    .rank(rank)
                    .dtype(dtype)
                    .bytes(bytes)
                    .device_type(device_type)
                    .device_id(device_id);
        }

        private static TensorView borrowedFrom(sdx_tensor_view_t nativeView) {
            TensorView view = new TensorView();
            view.data = nativeView.data();
            view.shape = nativeView.shape();
            view.rank = nativeView.rank();
            view.dtype = nativeView.dtype();
            view.bytes = nativeView.bytes();
            view.device_type = nativeView.device_type();
            view.device_id = nativeView.device_id();
            return view;
        }
    }

    public static final class ExecutionReport {
        public int struct_size;
        public int requested_backend;
        public int applied_backend;
        public int status_code;
        public int used_fallback;
        public long execution_time_ns;
        public int requested_gpu_target;
        public int applied_gpu_target;
        /** 0=warmup, 1=shapes frozen, 2=replaying, 3=replay blocked. */
        public int plan_phase;
        public int execution_count;

        private static ExecutionReport fromNative(sdx_execution_report_t nativeReport) {
            ExecutionReport report = new ExecutionReport();
            report.struct_size = nativeReport.struct_size();
            report.requested_backend = nativeReport.requested_backend();
            report.applied_backend = nativeReport.applied_backend();
            report.status_code = nativeReport.status_code();
            report.used_fallback = nativeReport.used_fallback();
            report.execution_time_ns = nativeReport.execution_time_ns();
            report.requested_gpu_target = nativeReport.requested_gpu_target();
            report.applied_gpu_target = nativeReport.applied_gpu_target();
            report.plan_phase = nativeReport.plan_phase();
            report.execution_count = nativeReport.execution_count();
            return report;
        }
    }

    private sdx_runtime_t runtimeHandle;
    private int openModels;

    private SdxRuntime(sdx_runtime_t runtimeHandle) {
        this.runtimeHandle = runtimeHandle;
    }

    /** Load the JavaCPP transport and create a backend-neutral SDX runtime. */
    public static SdxRuntime create() {
        try (sdx_runtime_options_t options = new sdx_runtime_options_t()) {
            options.struct_size(options.sizeof());
            sdx_runtime_t outRuntime = new sdx_runtime_t();
            int status = SdxNative.sdxCreateRuntime(options, outRuntime);
            if (status != SDX_STATUS_OK || Pointer.isNull(outRuntime)) {
                throw new IllegalStateException(
                        "sdxCreateRuntime failed with status=" + status);
            }
            return new SdxRuntime(outRuntime);
        }
    }

    public int abiVersion() {
        ensureOpen();
        return SdxNative.sdxGetRuntimeAbiVersion();
    }

    public synchronized SdxModel loadModel(String bundlePath, ModelOptions options) {
        ensureOpen();
        Objects.requireNonNull(bundlePath, "bundlePath");
        ModelOptions effective = options == null ? new ModelOptions() : options;
        try (sdx_model_options_t nativeOptions = toNative(effective)) {
            sdx_model_t outModel = new sdx_model_t();
            int status = SdxNative.sdxLoadBundle(
                    runtimeHandle, bundlePath, nativeOptions, outModel);
            checkStatus(status, "sdxLoadBundle");
            if (Pointer.isNull(outModel)) {
                throw new IllegalStateException("sdxLoadBundle returned a null model");
            }
            openModels++;
            return new SdxModel(outModel);
        }
    }

    public String lastError() {
        ensureOpen();
        return stringValue(SdxNative.sdxGetLastError(runtimeHandle), "");
    }

    private void checkStatus(int status, String operation) {
        if (status != SDX_STATUS_OK) {
            throw new IllegalStateException(
                    operation + " failed: " + lastError() + " (status=" + status + ")");
        }
    }

    private void ensureOpen() {
        if (Pointer.isNull(runtimeHandle)) {
            throw new IllegalStateException("SDX runtime is closed");
        }
    }

    private synchronized void modelClosed() {
        openModels--;
    }

    @Override
    public synchronized void close() {
        if (Pointer.isNull(runtimeHandle)) {
            return;
        }
        if (openModels != 0) {
            throw new IllegalStateException(
                    "Close all SDX models before closing the runtime (open=" + openModels + ")");
        }
        SdxNative.sdxDestroyRuntime(runtimeHandle);
        runtimeHandle.setNull();
        runtimeHandle = null;
    }

    public final class SdxModel implements AutoCloseable {
        private sdx_model_t modelHandle;
        private int openContexts;
        private int openTextSessions;

        private SdxModel(sdx_model_t modelHandle) {
            this.modelHandle = modelHandle;
        }

        public SdxContext createContext(String[] requestedOutputs) {
            return createContext(requestedOutputs, new ContextOptions());
        }

        /** Recommended mobile/offline context with bundle-owned weights bound internally. */
        public SdxContext createInferenceContext(String[] requestedOutputs) {
            return createContext(requestedOutputs, ContextOptions.inference());
        }

        public String tokenizerPath() {
            ensureModelOpen();
            return stringValue(SdxNative.sdxGetTokenizerPath(modelHandle), null);
        }

        public String textGenerationConfigPath() {
            ensureModelOpen();
            return stringValue(
                    SdxNative.sdxGetTextGenerationConfigPath(modelHandle), null);
        }

        /**
         * Create the shared metadata-driven AOT text-generation session. The
         * returned JavaCPP facade contains no tensor, KV-cache, or sampling loop.
         */
        public synchronized SdxTextSession createTextSession() {
            ensureModelOpen();
            SdxTextSession session = SdxTextSession.create(
                    SdxRuntime.this, this, modelHandle);
            openTextSessions++;
            return session;
        }

        public SdxContext createContext(
                String[] requestedOutputs, boolean bindModelParameters) {
            ContextOptions options = new ContextOptions();
            options.bind_model_parameters = bindModelParameters ? 1 : 0;
            return createContext(requestedOutputs, options);
        }

        public synchronized SdxContext createContext(
                String[] requestedOutputs, ContextOptions options) {
            ensureModelOpen();
            ContextOptions effective =
                    options == null ? new ContextOptions() : options;
            try (NativeStrings outputs = new NativeStrings(requestedOutputs);
                 sdx_context_options_t nativeOptions = toNative(effective);
                 PointerPointer<sdx_context_t> outContextPointer =
                         new PointerPointer<>(1)) {
                int status = SdxNative.sdxCreateContextWithOptions(
                        modelHandle,
                        outputs.pointer(),
                        outputs.size(),
                        nativeOptions,
                        outContextPointer);
                checkStatus(status, "sdxCreateContextWithOptions");
                sdx_context_t outContext =
                        outContextPointer.get(sdx_context_t.class, 0);
                if (Pointer.isNull(outContext)) {
                    throw new IllegalStateException(
                            "sdxCreateContextWithOptions returned a null context");
                }
                openContexts++;
                return new SdxContext(this, outContext);
            }
        }

        private void ensureModelOpen() {
            if (Pointer.isNull(modelHandle)) {
                throw new IllegalStateException("SDX model is closed");
            }
        }

        private synchronized void contextClosed() {
            openContexts--;
        }

        synchronized void textSessionClosed() {
            openTextSessions--;
        }

        @Override
        public synchronized void close() {
            if (Pointer.isNull(modelHandle)) {
                return;
            }
            if (openContexts != 0 || openTextSessions != 0) {
                throw new IllegalStateException(
                        "Close all SDX contexts and text sessions before closing the model "
                                + "(contexts=" + openContexts
                                + ", textSessions=" + openTextSessions + ")");
            }
            SdxNative.sdxUnloadModel(modelHandle);
            modelHandle.setNull();
            modelHandle = null;
            modelClosed();
        }
    }

    public final class SdxContext implements AutoCloseable {
        private final SdxModel modelOwner;
        private sdx_context_t contextHandle;

        private SdxContext(SdxModel modelOwner, sdx_context_t contextHandle) {
            this.modelOwner = modelOwner;
            this.contextHandle = contextHandle;
        }

        public synchronized void run(
                TensorView[] inputs, TensorView[] outputs, RunOptions options) {
            ensureContextOpen();
            Objects.requireNonNull(inputs, "inputs");
            Objects.requireNonNull(outputs, "outputs");
            try (NativeTensorArray nativeInputs = new NativeTensorArray(inputs);
                 NativeTensorArray nativeOutputs = new NativeTensorArray(outputs);
                 sdx_run_options_t nativeOptions = toNative(options)) {
                int status = SdxNative.sdxRun(
                        contextHandle,
                        nativeInputs.pointer(),
                        nativeInputs.size(),
                        nativeOutputs.pointer(),
                        nativeOutputs.size(),
                        nativeOptions);
                checkStatus(status, "sdxRun");
            }
        }

        /**
         * Run with runtime-owned dynamic outputs. Returned data and shape
         * pointers remain valid until the next run on this context.
         */
        public synchronized TensorView[] runAllocating(
                TensorView[] inputs, RunOptions options) {
            ensureContextOpen();
            Objects.requireNonNull(inputs, "inputs");
            try (NativeTensorArray nativeInputs = new NativeTensorArray(inputs);
                 sdx_run_options_t nativeOptions = toNative(options)) {
                int status = SdxNative.sdxRunAllocating(
                        contextHandle,
                        nativeInputs.pointer(),
                        nativeInputs.size(),
                        nativeOptions);
                checkStatus(status, "sdxRunAllocating");
            }

            int count = numOutputs();
            TensorView[] result = new TensorView[count];
            for (int i = 0; i < count; i++) {
                result[i] = outputTensor(i);
            }
            return result;
        }

        public TensorView[] runAllocating(TensorView[] inputs) {
            return runAllocating(inputs, null);
        }

        /**
         * Convenience adapter for ND4J INDArrays without a hard nd4j-api
         * dependency. It intentionally exposes host pointers: the portable SDX
         * layer owns Vulkan/Metal staging and device placement.
         */
        public void runNd4j(Object[] inputs, Object[] outputs, RunOptions options) {
            Nd4jArrayLease[] inputLeases =
                    Nd4jInterop.prepareLeases(inputs, false);
            Nd4jArrayLease[] outputLeases =
                    Nd4jInterop.prepareLeases(outputs, true);
            run(
                    Nd4jInterop.extractTensorViews(inputLeases),
                    Nd4jInterop.extractTensorViews(outputLeases),
                    options);
            Nd4jInterop.copyBackOutputs(outputLeases);
        }

        public void runNd4j(Object[] inputs, Object[] outputs) {
            runNd4j(inputs, outputs, null);
        }

        public synchronized void markInputVariable(int inputIndex) {
            ensureContextOpen();
            checkStatus(
                    SdxNative.sdxMarkInputVariable(contextHandle, inputIndex),
                    "sdxMarkInputVariable");
        }

        public synchronized void markInputPlaceholder(int inputIndex) {
            ensureContextOpen();
            checkStatus(
                    SdxNative.sdxMarkInputPlaceholder(contextHandle, inputIndex),
                    "sdxMarkInputPlaceholder");
        }

        public synchronized void freezeShapes() {
            ensureContextOpen();
            checkStatus(
                    SdxNative.sdxFreezeShapes(contextHandle),
                    "sdxFreezeShapes");
        }

        public int planPhase() {
            ensureContextOpen();
            return SdxNative.sdxGetPlanPhase(contextHandle);
        }

        /**
         * Functional-replay segment evidence for AOT planning and diagnostics.
         * Segment start/end indices use the native inclusive range contract.
         */
        public synchronized String planSegmentsSummaryJson() {
            ensureContextOpen();
            return stringValue(
                    SdxNative.sdxGetPlanSegmentsSummaryJson(contextHandle), "[]");
        }

        public int executionCount() {
            ensureContextOpen();
            return SdxNative.sdxGetExecutionCount(contextHandle);
        }

        public int numInputs() {
            ensureContextOpen();
            return SdxNative.sdxGetNumInputs(contextHandle);
        }

        public int numOutputs() {
            ensureContextOpen();
            return SdxNative.sdxGetNumOutputs(contextHandle);
        }

        public String inputName(int inputIndex) {
            ensureContextOpen();
            return stringValue(
                    SdxNative.sdxGetInputName(contextHandle, inputIndex), null);
        }

        public String[] inputNames() {
            int count = numInputs();
            if (count < 0) {
                return new String[0];
            }
            String[] names = new String[count];
            for (int i = 0; i < count; i++) {
                names[i] = inputName(i);
            }
            return names;
        }

        public String outputName(int outputIndex) {
            ensureContextOpen();
            return stringValue(
                    SdxNative.sdxGetOutputName(contextHandle, outputIndex), null);
        }

        public String[] outputNames() {
            int count = numOutputs();
            if (count < 0) {
                return new String[0];
            }
            String[] names = new String[count];
            for (int i = 0; i < count; i++) {
                names[i] = outputName(i);
            }
            return names;
        }

        public TensorView outputTensor(int outputIndex) {
            ensureContextOpen();
            try (sdx_tensor_view_t nativeTensor = new sdx_tensor_view_t()) {
                int status = SdxNative.sdxGetOutputTensor(
                        contextHandle, outputIndex, nativeTensor);
                checkStatus(status, "sdxGetOutputTensor");
                return TensorView.borrowedFrom(nativeTensor);
            }
        }

        public ExecutionReport executionReport() {
            ensureContextOpen();
            try (sdx_execution_report_t nativeReport =
                         new sdx_execution_report_t()) {
                nativeReport.struct_size(nativeReport.sizeof());
                int status = SdxNative.sdxGetExecutionReport(
                        contextHandle, nativeReport);
                checkStatus(status, "sdxGetExecutionReport");
                return ExecutionReport.fromNative(nativeReport);
            }
        }

        private void ensureContextOpen() {
            if (Pointer.isNull(contextHandle)) {
                throw new IllegalStateException("SDX context is closed");
            }
        }

        @Override
        public synchronized void close() {
            if (Pointer.isNull(contextHandle)) {
                return;
            }
            SdxNative.sdxDestroyContext(contextHandle);
            contextHandle.setNull();
            contextHandle = null;
            modelOwner.contextClosed();
        }
    }

    private static sdx_model_options_t toNative(ModelOptions options) {
        sdx_model_options_t nativeOptions = new sdx_model_options_t();
        nativeOptions
                .struct_size(nativeOptions.sizeof())
                .backend(options.backend)
                .strict_backend(options.strict_backend)
                .allow_runtime_jit(options.allow_runtime_jit)
                .gpu_target(options.gpu_target);
        return nativeOptions;
    }

    private static sdx_context_options_t toNative(ContextOptions options) {
        sdx_context_options_t nativeOptions = new sdx_context_options_t();
        nativeOptions
                .struct_size(nativeOptions.sizeof())
                .bind_model_parameters(options.bind_model_parameters);
        return nativeOptions;
    }

    private static sdx_run_options_t toNative(RunOptions options) {
        if (options == null) {
            return null;
        }
        sdx_run_options_t nativeOptions = new sdx_run_options_t();
        nativeOptions
                .struct_size(nativeOptions.sizeof())
                .backend(options.backend)
                .strict_signature(options.strict_signature)
                .gpu_target(options.gpu_target);
        return nativeOptions;
    }

    private static String stringValue(BytePointer value, String nullValue) {
        return Pointer.isNull(value) ? nullValue : value.getString();
    }

    private static final class NativeStrings implements AutoCloseable {
        private final BytePointer[] values;
        private final PointerPointer<BytePointer> pointers;

        private NativeStrings(String[] strings) {
            if (strings == null || strings.length == 0) {
                values = new BytePointer[0];
                pointers = null;
                return;
            }
            values = new BytePointer[strings.length];
            for (int i = 0; i < strings.length; i++) {
                values[i] = new BytePointer(
                        Objects.requireNonNull(strings[i], "requested output name"));
            }
            pointers = new PointerPointer<>(values);
        }

        private PointerPointer<BytePointer> pointer() {
            return pointers;
        }

        private int size() {
            return values.length;
        }

        @Override
        public void close() {
            if (pointers != null) {
                pointers.close();
            }
            for (BytePointer value : values) {
                value.close();
            }
        }
    }

    private static final class NativeTensorArray implements AutoCloseable {
        private final sdx_tensor_view_t values;
        private final int size;

        private NativeTensorArray(TensorView[] tensors) {
            size = tensors.length;
            if (size == 0) {
                values = null;
                return;
            }
            values = new sdx_tensor_view_t(size);
            for (int i = 0; i < size; i++) {
                TensorView tensor =
                        Objects.requireNonNull(tensors[i], "tensor[" + i + "]");
                tensor.writeTo(values.position(i));
            }
            values.position(0);
        }

        private sdx_tensor_view_t pointer() {
            return values;
        }

        private int size() {
            return size;
        }

        @Override
        public void close() {
            if (values != null) {
                values.close();
            }
        }
    }

    private static final class Nd4jArrayLease {
        private final Object original;
        private final Object working;
        private final TensorView view;
        private final boolean copyBack;

        private Nd4jArrayLease(
                Object original, Object working, TensorView view, boolean copyBack) {
            this.original = original;
            this.working = working;
            this.view = view;
            this.copyBack = copyBack;
        }
    }

    private static final class Nd4jInterop {
        private static final String ND4J_CLASS_NAME =
                "org.nd4j.linalg.factory.Nd4j";
        private static final String INDARRAY_CLASS_NAME =
                "org.nd4j.linalg.api.ndarray.INDArray";

        private static final Class<?> ND4J_CLASS = loadClass(ND4J_CLASS_NAME);
        private static final Class<?> INDARRAY_CLASS =
                loadClass(INDARRAY_CLASS_NAME);

        private Nd4jInterop() {
        }

        private static Class<?> loadClass(String className) {
            try {
                return Class.forName(className);
            } catch (ClassNotFoundException e) {
                throw new IllegalStateException(
                        "ND4J class not found: " + className
                                + ". Add ND4J to use runNd4j().",
                        e);
            }
        }

        private static Method method(
                Class<?> type, String name, Class<?>... argumentTypes) {
            try {
                return type.getMethod(name, argumentTypes);
            } catch (NoSuchMethodException e) {
                throw new IllegalStateException(
                        "ND4J API mismatch: missing "
                                + type.getName() + "." + name,
                        e);
            }
        }

        private static Object invoke(
                Object target,
                String name,
                Class<?>[] argumentTypes,
                Object... arguments) {
            try {
                return method(target.getClass(), name, argumentTypes)
                        .invoke(target, arguments);
            } catch (IllegalAccessException | InvocationTargetException e) {
                throw new IllegalStateException(
                        "Failed to call "
                                + target.getClass().getName() + "." + name,
                        e);
            }
        }

        private static Object invokeNoArgs(Object target, String name) {
            return invoke(target, name, new Class<?>[0]);
        }

        private static boolean isCContiguous(Object array) {
            char ordering = (Character) invokeNoArgs(array, "ordering");
            if (ordering != 'c' && ordering != 'C') {
                return false;
            }
            long[] shape = (long[]) invokeNoArgs(array, "shape");
            long[] stride = (long[]) invokeNoArgs(array, "stride");
            long expected = 1;
            for (int i = shape.length - 1; i >= 0; i--) {
                if (shape[i] != 1 && stride[i] != expected) {
                    return false;
                }
                expected *= shape[i];
            }
            return true;
        }

        private static Nd4jArrayLease toLease(Object array, boolean output) {
            Objects.requireNonNull(array, "INDArray");
            if (!INDARRAY_CLASS.isInstance(array)) {
                throw new IllegalArgumentException(
                        "Expected ND4J INDArray, got "
                                + array.getClass().getName());
            }

            Object working = array;
            boolean copyBack = false;
            if (!isCContiguous(working)) {
                working = invoke(
                        working, "dup", new Class<?>[]{char.class}, 'c');
                copyBack = output;
            }

            Object dataBuffer = invokeNoArgs(working, "data");
            Object dataType = invokeNoArgs(working, "dataType");
            int dtypeCode = ((Number) invoke(
                    dataType, "toInt", new Class<?>[0])).intValue();
            long[] shape = (long[]) invokeNoArgs(working, "shape");
            long length = ((Number) invokeNoArgs(working, "length")).longValue();
            long offset = ((Number) invokeNoArgs(working, "offset")).longValue();
            int elementSize = ((Number) invokeNoArgs(
                    dataBuffer, "getElementSize")).intValue();
            Pointer base = (Pointer) invokeNoArgs(dataBuffer, "addressPointer");
            Pointer data = new Pointer(base).position(offset * elementSize);

            TensorView view = TensorView.hostTensor(
                    data, shape, dtypeCode, length * elementSize);
            return new Nd4jArrayLease(array, working, view, copyBack);
        }

        private static Nd4jArrayLease[] prepareLeases(
                Object[] arrays, boolean output) {
            Objects.requireNonNull(arrays, "INDArray array");
            Nd4jArrayLease[] leases = new Nd4jArrayLease[arrays.length];
            for (int i = 0; i < arrays.length; i++) {
                leases[i] = toLease(arrays[i], output);
            }
            return leases;
        }

        private static TensorView[] extractTensorViews(
                Nd4jArrayLease[] leases) {
            TensorView[] views = new TensorView[leases.length];
            for (int i = 0; i < leases.length; i++) {
                views[i] = leases[i].view;
            }
            return views;
        }

        private static void copyBackOutputs(
                Nd4jArrayLease[] outputLeases) {
            for (Nd4jArrayLease lease : outputLeases) {
                if (lease.copyBack) {
                    invoke(
                            lease.original,
                            "assign",
                            new Class<?>[]{INDARRAY_CLASS},
                            lease.working);
                }
            }
        }
    }
}
