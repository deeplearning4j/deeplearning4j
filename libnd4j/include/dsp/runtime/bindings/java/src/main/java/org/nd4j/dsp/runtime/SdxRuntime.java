package org.nd4j.dsp.runtime;

import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

/**
 * Java wrapper for the SDX C runtime ABI (`dsp_runtime_c.h`).
 *
 * This implementation uses JNA and supports the runtime lifecycle:
 * create runtime -> load model -> create context -> run -> destroy.
 */
public final class SdxRuntime implements AutoCloseable {

    public static final int SDX_STATUS_OK = 0;

    public static final int SDX_BACKEND_AUTO = 0;
    public static final int SDX_BACKEND_SLOT_BY_SLOT = 1;
    public static final int SDX_BACKEND_CUDA_GRAPHS = 2;
    public static final int SDX_BACKEND_NVRTC = 3;
    public static final int SDX_BACKEND_PTX = 4;
    public static final int SDX_BACKEND_TRITON = 5;
    public static final int SDX_BACKEND_MLX = 6;
    public static final int SDX_BACKEND_ARM_HYBRID = 7;
    public static final int SDX_BACKEND_NNAPI = 8;

    public static final int SDX_DEVICE_HOST = 0;
    public static final int SDX_DEVICE_CUDA = 1;
    public static final int SDX_DEVICE_AMD = 2;

    public static final int SDX_GPU_TARGET_AUTO = 0;
    public static final int SDX_GPU_TARGET_CUDA = 1;
    public static final int SDX_GPU_TARGET_AMD = 2;

    interface NativeApi extends Library {
        int sdxGetRuntimeAbiVersion();

        int sdxCreateRuntime(RuntimeOptions options, PointerByReference outRuntime);

        void sdxDestroyRuntime(Pointer runtime);

        int sdxLoadBundle(Pointer runtime, String bundlePath, ModelOptions options, PointerByReference outModel);

        void sdxUnloadModel(Pointer model);

        int sdxCreateContext(Pointer model, StringArray requestedOutputs, int numRequestedOutputs, PointerByReference outContext);

        void sdxDestroyContext(Pointer context);

        int sdxRun(Pointer context,
                   TensorView[] inputs,
                   int numInputs,
                   TensorView[] outputs,
                   int numOutputs,
                   RunOptions options);

        Pointer sdxGetLastError(Pointer runtime);

        int sdxGetExecutionReport(Pointer context, ExecutionReport outReport);
    }

    public static final class RuntimeOptions extends Structure {
        public int struct_size;

        public RuntimeOptions() {
            struct_size = size();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Collections.singletonList("struct_size");
        }
    }

    public static final class ModelOptions extends Structure {
        public int struct_size;
        public int backend;
        public int strict_backend;
        public int allow_runtime_jit;
        public int gpu_target;

        public ModelOptions() {
            struct_size = size();
            backend = SDX_BACKEND_AUTO;
            strict_backend = 0;
            allow_runtime_jit = 0;
            gpu_target = SDX_GPU_TARGET_AUTO;
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "backend", "strict_backend", "allow_runtime_jit", "gpu_target");
        }
    }

    public static final class RunOptions extends Structure {
        public int struct_size;
        public int backend;
        public int strict_signature;
        public int gpu_target;

        public RunOptions() {
            struct_size = size();
            backend = SDX_BACKEND_AUTO;
            strict_signature = 1;
            gpu_target = SDX_GPU_TARGET_AUTO;
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("struct_size", "backend", "strict_signature", "gpu_target");
        }
    }

    public static final class TensorView extends Structure {
        public Pointer data;
        public Pointer shape;
        public int rank;
        public int dtype;
        public long bytes;
        public int device_type;
        public int device_id;

        private transient Memory shapeOwner;

        public TensorView() {
            device_type = SDX_DEVICE_HOST;
            device_id = -1;
        }

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
            TensorView tv = new TensorView();
            tv.data = data;
            tv.rank = shapeValues.length;
            tv.dtype = dtype;
            tv.bytes = bytes;
            tv.device_type = deviceType;
            tv.device_id = deviceId;
            if (shapeValues.length > 0) {
                Memory shapeMem = new Memory((long) shapeValues.length * Long.BYTES);
                for (int i = 0; i < shapeValues.length; i++) {
                    shapeMem.setLong((long) i * Long.BYTES, shapeValues[i]);
                }
                tv.shapeOwner = shapeMem;
                tv.shape = shapeMem;
            }
            return tv;
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("data", "shape", "rank", "dtype", "bytes", "device_type", "device_id");
        }
    }

    public static final class ExecutionReport extends Structure {
        public int struct_size;
        public int requested_backend;
        public int applied_backend;
        public int status_code;
        public int used_fallback;
        public long execution_time_ns;
        public int requested_gpu_target;
        public int applied_gpu_target;

        public ExecutionReport() {
            struct_size = size();
        }

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList(
                "struct_size",
                "requested_backend",
                "applied_backend",
                "status_code",
                "used_fallback",
                "execution_time_ns",
                "requested_gpu_target",
                "applied_gpu_target");
        }
    }

    private final NativeApi api;
    private Pointer runtimeHandle;

    private SdxRuntime(NativeApi api, Pointer runtimeHandle) {
        this.api = api;
        this.runtimeHandle = runtimeHandle;
    }

    public static SdxRuntime create() {
        UnsatisfiedLinkError lastError = null;
        String[] candidates = defaultLibraryCandidates();
        for (String candidate : candidates) {
            try {
                return create(candidate);
            } catch (UnsatisfiedLinkError e) {
                lastError = e;
            }
        }

        StringBuilder message = new StringBuilder("Unable to load SDX runtime library. Tried: ");
        message.append(String.join(", ", candidates));
        if (lastError != null && lastError.getMessage() != null) {
            message.append(" (last error: ").append(lastError.getMessage()).append(")");
        }
        throw new UnsatisfiedLinkError(message.toString());
    }

    public static SdxRuntime create(String libraryNameOrPath) {
        NativeApi api = Native.load(libraryNameOrPath, NativeApi.class);
        PointerByReference outRuntime = new PointerByReference();
        RuntimeOptions options = new RuntimeOptions();
        options.write();
        int status = api.sdxCreateRuntime(options, outRuntime);
        if (status != SDX_STATUS_OK) {
            throw new IllegalStateException("sdxCreateRuntime failed with status=" + status);
        }
        return new SdxRuntime(api, outRuntime.getValue());
    }

    private static String[] defaultLibraryCandidates() {
        List<String> names = new ArrayList<>();
        names.add("nd4jcpu");
        names.add("nd4jcuda");
        names.add("nd4jamd");
        return names.toArray(new String[0]);
    }

    private static final class Nd4jArrayLease {
        private final Object original;
        private final Object working;
        private final TensorView view;
        private final boolean copyBack;

        private Nd4jArrayLease(Object original, Object working, TensorView view, boolean copyBack) {
            this.original = original;
            this.working = working;
            this.view = view;
            this.copyBack = copyBack;
        }
    }

    private static final class Nd4jInterop {
        private static final String ND4J_CLASS_NAME = "org.nd4j.linalg.factory.Nd4j";
        private static final String INDARRAY_CLASS_NAME = "org.nd4j.linalg.api.ndarray.INDArray";

        private static final Class<?> ND4J_CLASS = loadClass(ND4J_CLASS_NAME);
        private static final Class<?> INDARRAY_CLASS = loadClass(INDARRAY_CLASS_NAME);

        private Nd4jInterop() {}

        private static Class<?> loadClass(String className) {
            try {
                return Class.forName(className);
            } catch (ClassNotFoundException e) {
                throw new IllegalStateException(
                    "ND4J class not found: " + className + ". Add ND4J to the classpath to use runNd4j().", e);
            }
        }

        private static Method method(Class<?> type, String name, Class<?>... argTypes) {
            try {
                return type.getMethod(name, argTypes);
            } catch (NoSuchMethodException e) {
                throw new IllegalStateException("ND4J API mismatch: missing method " + type.getName() + "." + name, e);
            }
        }

        private static Object invoke(Object target, String name, Class<?>[] argTypes, Object... args) {
            try {
                Method m = method(target.getClass(), name, argTypes);
                return m.invoke(target, args);
            } catch (IllegalAccessException | InvocationTargetException e) {
                throw new IllegalStateException("Failed to call ND4J method " + target.getClass().getName() + "." + name, e);
            }
        }

        private static Object invokeNoArgs(Object target, String name) {
            return invoke(target, name, new Class<?>[0]);
        }

        private static Object invokeStatic(Class<?> type, String name, Class<?>[] argTypes, Object... args) {
            try {
                Method m = method(type, name, argTypes);
                return m.invoke(null, args);
            } catch (IllegalAccessException | InvocationTargetException e) {
                throw new IllegalStateException("Failed to call ND4J static method " + type.getName() + "." + name, e);
            }
        }

        private static boolean isCpuBackend() {
            Object environment = invokeStatic(ND4J_CLASS, "getEnvironment", new Class<?>[0]);
            Object cpuFlag = invokeNoArgs(environment, "isCPU");
            return Boolean.TRUE.equals(cpuFlag);
        }

        private static int currentDeviceId() {
            try {
                Object affinityManager = invokeStatic(ND4J_CLASS, "getAffinityManager", new Class<?>[0]);
                Object value = invokeNoArgs(affinityManager, "getDeviceForCurrentThread");
                return ((Number) value).intValue();
            } catch (RuntimeException e) {
                return -1;
            }
        }

        private static boolean isCContiguous(Object indArray) {
            char ordering = (Character) invokeNoArgs(indArray, "ordering");
            int ews = ((Number) invokeNoArgs(indArray, "elementWiseStride")).intValue();
            return (ordering == 'c' || ordering == 'C') && ews == 1;
        }

        private static Nd4jArrayLease toLease(Object indArray, boolean output) {
            if (indArray == null) {
                throw new IllegalArgumentException("INDArray entry must not be null");
            }
            if (!INDARRAY_CLASS.isInstance(indArray)) {
                throw new IllegalArgumentException("Expected ND4J INDArray, got: " + indArray.getClass().getName());
            }

            Object working = indArray;
            boolean copyBack = false;
            if (!isCContiguous(working)) {
                working = invoke(working, "dup", new Class<?>[]{char.class}, 'c');
                copyBack = output;
            }

            Object dataBuffer = invokeNoArgs(working, "data");
            Object dataType = invokeNoArgs(working, "dataType");
            int dtypeCode = ((Number) invoke(dataType, "toInt", new Class<?>[0])).intValue();
            long[] shape = (long[]) invokeNoArgs(working, "shape");
            long length = ((Number) invokeNoArgs(working, "length")).longValue();
            long offset = ((Number) invokeNoArgs(working, "offset")).longValue();
            int elementSize = ((Number) invokeNoArgs(dataBuffer, "getElementSize")).intValue();

            boolean cpu = isCpuBackend();
            long baseAddress = ((Number) invokeNoArgs(dataBuffer, cpu ? "address" : "platformAddress")).longValue();
            long address = baseAddress + (offset * elementSize);
            int deviceType = cpu ? SDX_DEVICE_HOST : SDX_DEVICE_CUDA;
            int deviceId = cpu ? -1 : currentDeviceId();

            TensorView view = TensorView.tensor(
                Pointer.createConstant(address),
                shape,
                dtypeCode,
                length * elementSize,
                deviceType,
                deviceId);
            return new Nd4jArrayLease(indArray, working, view, copyBack);
        }

        private static Nd4jArrayLease[] prepareLeases(Object[] arrays, boolean output) {
            if (arrays == null) {
                throw new IllegalArgumentException("INDArray array must not be null");
            }
            Nd4jArrayLease[] leases = new Nd4jArrayLease[arrays.length];
            for (int i = 0; i < arrays.length; i++) {
                leases[i] = toLease(arrays[i], output);
            }
            return leases;
        }

        private static TensorView[] extractTensorViews(Nd4jArrayLease[] leases) {
            TensorView[] views = new TensorView[leases.length];
            for (int i = 0; i < leases.length; i++) {
                views[i] = leases[i].view;
            }
            return views;
        }

        private static void copyBackOutputs(Nd4jArrayLease[] outputLeases) {
            for (Nd4jArrayLease lease : outputLeases) {
                if (!lease.copyBack) {
                    continue;
                }
                invoke(lease.original, "assign", new Class<?>[]{INDARRAY_CLASS}, lease.working);
            }
        }
    }

    public int abiVersion() {
        return api.sdxGetRuntimeAbiVersion();
    }

    public SdxModel loadModel(String bundlePath, ModelOptions options) {
        ensureOpen();
        PointerByReference outModel = new PointerByReference();
        if (options != null) {
            options.write();
        }
        int status = api.sdxLoadBundle(runtimeHandle, bundlePath, options, outModel);
        if (status != SDX_STATUS_OK) {
            throw new IllegalStateException("sdxLoadBundle failed: " + lastError() + " (status=" + status + ")");
        }
        return new SdxModel(outModel.getValue());
    }

    public String lastError() {
        ensureOpen();
        Pointer err = api.sdxGetLastError(runtimeHandle);
        if (err == null) {
            return "";
        }
        return err.getString(0);
    }

    private void ensureOpen() {
        if (runtimeHandle == null) {
            throw new IllegalStateException("SDX runtime is closed");
        }
    }

    @Override
    public void close() {
        if (runtimeHandle != null) {
            api.sdxDestroyRuntime(runtimeHandle);
            runtimeHandle = null;
        }
    }

    public final class SdxModel implements AutoCloseable {
        private Pointer modelHandle;

        private SdxModel(Pointer modelHandle) {
            this.modelHandle = modelHandle;
        }

        public SdxContext createContext(String[] requestedOutputs) {
            if (modelHandle == null) {
                throw new IllegalStateException("Model handle is closed");
            }

            StringArray outputArray = null;
            int numOutputs = 0;
            if (requestedOutputs != null && requestedOutputs.length > 0) {
                outputArray = new StringArray(requestedOutputs);
                numOutputs = requestedOutputs.length;
            }

            PointerByReference outContext = new PointerByReference();
            int status = api.sdxCreateContext(modelHandle, outputArray, numOutputs, outContext);
            if (status != SDX_STATUS_OK) {
                throw new IllegalStateException("sdxCreateContext failed: " + lastError() + " (status=" + status + ")");
            }
            return new SdxContext(outContext.getValue());
        }

        @Override
        public void close() {
            if (modelHandle != null) {
                api.sdxUnloadModel(modelHandle);
                modelHandle = null;
            }
        }
    }

    public final class SdxContext implements AutoCloseable {
        private Pointer contextHandle;

        private SdxContext(Pointer contextHandle) {
            this.contextHandle = contextHandle;
        }

        public void run(TensorView[] inputs, TensorView[] outputs, RunOptions options) {
            if (contextHandle == null) {
                throw new IllegalStateException("Context handle is closed");
            }
            if (inputs == null || outputs == null) {
                throw new IllegalArgumentException("inputs and outputs must not be null");
            }

            for (TensorView in : inputs) {
                in.write();
            }
            for (TensorView out : outputs) {
                out.write();
            }
            if (options != null) {
                options.write();
            }

            int status = api.sdxRun(contextHandle, inputs, inputs.length, outputs, outputs.length, options);
            if (status != SDX_STATUS_OK) {
                throw new IllegalStateException("sdxRun failed: " + lastError() + " (status=" + status + ")");
            }

            for (TensorView out : outputs) {
                out.read();
            }
        }

        /**
         * Run with ND4J INDArrays. This method accepts {@code INDArray[]} without introducing
         * a hard compile-time dependency in this wrapper.
         *
         * Inputs that are not C-contiguous are duplicated to C-order before execution.
         * Outputs that are not C-contiguous are written to a temporary C-order array and copied back.
         */
        public void runNd4j(Object[] inputs, Object[] outputs, RunOptions options) {
            Nd4jArrayLease[] inputLeases = Nd4jInterop.prepareLeases(inputs, false);
            Nd4jArrayLease[] outputLeases = Nd4jInterop.prepareLeases(outputs, true);
            run(Nd4jInterop.extractTensorViews(inputLeases), Nd4jInterop.extractTensorViews(outputLeases), options);
            Nd4jInterop.copyBackOutputs(outputLeases);
        }

        public void runNd4j(Object[] inputs, Object[] outputs) {
            runNd4j(inputs, outputs, null);
        }

        public ExecutionReport executionReport() {
            if (contextHandle == null) {
                throw new IllegalStateException("Context handle is closed");
            }
            ExecutionReport report = new ExecutionReport();
            report.write();
            int status = api.sdxGetExecutionReport(contextHandle, report);
            if (status != SDX_STATUS_OK) {
                throw new IllegalStateException("sdxGetExecutionReport failed: status=" + status);
            }
            report.read();
            return report;
        }

        @Override
        public void close() {
            if (contextHandle != null) {
                api.sdxDestroyContext(contextHandle);
                contextHandle = null;
            }
        }
    }
}
