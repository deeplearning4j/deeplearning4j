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

package org.nd4j.linalg.api.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.profiler.ProfilerConfig;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Device-aware OpExecutioner wrapper that automatically handles cross-device
 * data transfers before op execution.
 *
 * <p>This wrapper intercepts all op execution calls and:
 * <ul>
 *   <li>Determines the optimal execution device based on input locations</li>
 *   <li>Automatically transfers inputs to the target device if needed</li>
 *   <li>Delegates actual execution to the underlying OpExecutioner</li>
 *   <li>Records transfer statistics for monitoring</li>
 * </ul>
 *
 * <h3>Usage:</h3>
 * <pre>{@code
 * // Install the device-aware executor wrapper
 * DeviceAwareOpExecutioner.install();
 *
 * // Now all Nd4j operations automatically handle cross-device transfers
 * INDArray cpuArray = Nd4j.create(new float[]{1, 2, 3, 4});
 * INDArray gpuArray = DeviceAwareNd4j.createOnGpu(new long[]{4});
 *
 * // This automatically transfers cpuArray to GPU before execution
 * INDArray result = cpuArray.add(gpuArray);
 * }</pre>
 *
 * Adam Gibson
 * @see OpExecutionDelegator
 * @see DeviceRoutingConfiguration
 */
@Slf4j
public class DeviceAwareOpExecutioner implements OpExecutioner {

    private final OpExecutioner delegate;
    private final OpExecutionDelegator delegator;
    private volatile boolean enabled = true;

    private static volatile DeviceAwareOpExecutioner INSTANCE;
    private static volatile OpExecutioner originalExecutioner;
    private static final Object LOCK = new Object();

    // Multi-backend support: map of device type to executioner
    private final Map<DeviceType, OpExecutioner> backendExecutioners = new ConcurrentHashMap<>();
    private volatile boolean multiBackendEnabled = false;

    /**
     * Create a device-aware wrapper around the given executioner.
     *
     * @param delegate the underlying executioner to wrap
     */
    public DeviceAwareOpExecutioner(OpExecutioner delegate) {
        this.delegate = delegate;
        this.delegator = OpExecutionDelegator.getInstance();

        // Register the primary delegate based on its type
        DeviceType primaryType = detectExecutionerDeviceType(delegate);
        backendExecutioners.put(primaryType, delegate);
    }

    /**
     * Detect the device type for an executioner based on its class name.
     */
    private DeviceType detectExecutionerDeviceType(OpExecutioner executioner) {
        if (executioner == null) {
            return DeviceType.CPU;
        }
        String className = executioner.getClass().getName().toLowerCase();
        if (className.contains("cuda") || className.contains("gpu")) {
            return DeviceType.CUDA_GPU;
        }
        return DeviceType.CPU;
    }

    /**
     * Register an additional backend executioner for a device type.
     * This enables true multi-backend routing where ops can be executed
     * on different backends based on data location.
     *
     * @param deviceType the device type this executioner handles
     * @param executioner the executioner for that device type
     */
    public void registerBackendExecutioner(DeviceType deviceType, OpExecutioner executioner) {
        backendExecutioners.put(deviceType, executioner);
        multiBackendEnabled = backendExecutioners.size() > 1;
        log.info("Registered {} backend executioner: {}", deviceType, executioner.getClass().getSimpleName());
    }

    /**
     * Get the executioner for a specific device type.
     *
     * @param deviceType the device type
     * @return the executioner, or delegate if not found
     */
    public OpExecutioner getExecutionerForDeviceType(DeviceType deviceType) {
        OpExecutioner exec = backendExecutioners.get(deviceType);
        return exec != null ? exec : delegate;
    }

    /**
     * Get the executioner for a specific device.
     *
     * @param device the device descriptor
     * @return the appropriate executioner
     */
    public OpExecutioner getExecutionerForDevice(DeviceDescriptor device) {
        if (device == null) {
            return delegate;
        }
        return getExecutionerForDeviceType(device.getDeviceType());
    }

    /**
     * Check if multi-backend mode is enabled (multiple backends registered).
     */
    public boolean isMultiBackendEnabled() {
        return multiBackendEnabled;
    }

    /**
     * Get information about registered backends.
     */
    public String getBackendInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("DeviceAwareOpExecutioner backends:\n");
        sb.append("  Multi-backend enabled: ").append(multiBackendEnabled).append("\n");
        for (Map.Entry<DeviceType, OpExecutioner> entry : backendExecutioners.entrySet()) {
            sb.append("  ").append(entry.getKey()).append(": ")
              .append(entry.getValue().getClass().getSimpleName()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Install this wrapper as the global OpExecutioner.
     * This enables automatic device transfers for all operations.
     */
    public static void install() {
        synchronized (LOCK) {
            if (INSTANCE != null) {
                log.debug("DeviceAwareOpExecutioner already installed");
                return;
            }

            try {
                // Get the current executioner via reflection since Nd4j.getExecutioner()
                // returns the field directly
                java.lang.reflect.Field field = Nd4j.class
                        .getDeclaredField("OP_EXECUTIONER_INSTANCE");
                field.setAccessible(true);

                originalExecutioner = (OpExecutioner) field.get(null);
                INSTANCE = new DeviceAwareOpExecutioner(originalExecutioner);
                field.set(null, INSTANCE);

                log.info("DeviceAwareOpExecutioner installed - automatic device transfers enabled");
            } catch (Exception e) {
                log.warn("Failed to install DeviceAwareOpExecutioner: {}", e.getMessage());
                throw new RuntimeException("Failed to install device-aware executioner", e);
            }
        }
    }

    /**
     * Install this wrapper with automatic multi-backend discovery.
     * This will:
     * <ul>
     *   <li>Install the DeviceAwareOpExecutioner wrapper</li>
     *   <li>Attempt to load the CPU backend if available</li>
     *   <li>Register CPU as a fallback executioner for spillover</li>
     * </ul>
     *
     * <p>If the primary backend is CUDA and the CPU backend (nd4j-native) is on the classpath,
     * this enables true CPU fallback execution for operations when GPU memory is constrained
     * or data has spilled to CPU.</p>
     *
     * @return true if multi-backend mode was enabled (CPU backend found), false if single-backend only
     */
    public static boolean installWithMultiBackend() {
        synchronized (LOCK) {
            // First do standard install
            install();

            if (INSTANCE == null) {
                return false;
            }

            // Check if primary is GPU-based (CUDA/ROCm)
            DeviceType primaryType = INSTANCE.detectExecutionerDeviceType(INSTANCE.delegate);
            if (primaryType != DeviceType.CUDA_GPU && primaryType != DeviceType.ROCM_GPU) {
                log.debug("Primary backend is not GPU-based, multi-backend mode not needed");
                return false;
            }

            // Try to load CPU backend as fallback
            if (!CpuBackendLoader.isCpuBackendAvailable()) {
                log.info("CPU backend (nd4j-native) not available - single-backend mode only");
                return false;
            }

            OpExecutioner cpuExecutioner = CpuBackendLoader.loadCpuExecutioner();
            if (cpuExecutioner == null) {
                log.warn("Failed to load CPU executioner - single-backend mode only");
                return false;
            }

            // Register CPU backend
            INSTANCE.registerBackendExecutioner(DeviceType.CPU, cpuExecutioner);

            log.info("Multi-backend mode enabled: {} (primary) + CPU (fallback)",
                    primaryType);
            log.info(INSTANCE.getBackendInfo());

            return true;
        }
    }

    /**
     * Check if multi-backend installation should be attempted automatically.
     * Controlled by system property: nd4j.multibackend.enabled
     *
     * @return true if auto multi-backend is enabled
     */
    public static boolean isAutoMultiBackendEnabled() {
        return Boolean.parseBoolean(
                System.getProperty("nd4j.multibackend.enabled", "false"));
    }

    /**
     * Conditionally install with multi-backend if the system property is set.
     * This can be called during Nd4j initialization.
     */
    public static void installIfConfigured() {
        if (isAutoMultiBackendEnabled()) {
            log.info("Auto multi-backend mode enabled via system property");
            installWithMultiBackend();
        }
    }

    /**
     * Uninstall this wrapper and restore the original executioner.
     */
    public static void uninstall() {
        synchronized (LOCK) {
            if (INSTANCE == null || originalExecutioner == null) {
                return;
            }

            try {
                java.lang.reflect.Field field = Nd4j.class
                        .getDeclaredField("OP_EXECUTIONER_INSTANCE");
                field.setAccessible(true);
                field.set(null, originalExecutioner);

                INSTANCE = null;
                originalExecutioner = null;

                log.info("DeviceAwareOpExecutioner uninstalled");
            } catch (Exception e) {
                log.warn("Failed to uninstall DeviceAwareOpExecutioner: {}", e.getMessage());
            }
        }
    }

    /**
     * Check if the device-aware executioner is installed.
     */
    public static boolean isInstalled() {
        return INSTANCE != null;
    }

    /**
     * Get the installed instance (may be null).
     */
    public static DeviceAwareOpExecutioner getInstance() {
        return INSTANCE;
    }

    /**
     * Enable or disable automatic device transfers.
     *
     * @param enabled true to enable, false to disable
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    /**
     * Check if automatic device transfers are enabled.
     */
    public boolean isEnabled() {
        return enabled;
    }

    // =========================================================================
    // Op Execution with Device Awareness and Multi-Backend Routing
    // =========================================================================

    @Override
    public INDArray[] exec(CustomOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op, context);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op, context);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray exec(ReduceOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray exec(BroadcastOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray exec(ScalarOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op, opContext);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op, opContext);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray exec(RandomOp op) {
        // Random ops use the current device context
        return delegate.exec(op);
    }

    @Override
    public INDArray exec(RandomOp op, Random rng) {
        return delegate.exec(op, rng);
    }

    @Override
    public INDArray exec(Variance accumulation) {
        DeviceDescriptor targetDevice = prepareOpForExecution(accumulation);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(accumulation);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    /**
     * Select the appropriate executioner for a target device and set the CUDA device context.
     * If multi-backend is not enabled or no specific backend is found,
     * returns the default delegate.
     */
    private OpExecutioner selectExecutionerForDevice(DeviceDescriptor device) {
        if (!multiBackendEnabled || device == null) {
            return delegate;
        }
        return getExecutionerForDevice(device);
    }

    /**
     * Set the CUDA device context for the current thread before op execution.
     * This is critical for multi-GPU setups where data may reside on different devices.
     *
     * <p>The method checks if the target device differs from the current thread's device
     * and switches if necessary. This ensures that CUDA operations execute on the
     * device where the data actually resides.</p>
     *
     * @param targetDevice the device where execution should occur (may be null)
     */
    /**
     * Switch to the target device for op execution.
     *
     * @return the original device ID (for restore), or -1 if no switch was needed
     */
    private int setDeviceContextForExecution(DeviceDescriptor targetDevice) {
        if (targetDevice == null) {
            return -1;
        }

        // Only relevant for GPU devices
        if (targetDevice.getDeviceType() != DeviceType.CUDA_GPU &&
            targetDevice.getDeviceType() != DeviceType.ROCM_GPU) {
            return -1;
        }

        try {
            AffinityManager affinityManager = Nd4j.getAffinityManager();
            int currentDevice = affinityManager.getDeviceForCurrentThread();
            int targetDeviceIndex = targetDevice.getDeviceIndex();

            if (currentDevice != targetDeviceIndex) {
                if (log.isDebugEnabled()) {
                    log.debug("Switching CUDA device from {} to {} for op execution",
                              currentDevice, targetDeviceIndex);
                }
                DeviceMemoryManager.getInstance().switchDevice(targetDeviceIndex, "DeviceAwareOpExecutioner.exec", "target-device");
                return currentDevice;
            }
        } catch (Exception e) {
            // Log but don't fail - device may already be correct or single-GPU system
            if (log.isTraceEnabled()) {
                log.trace("Could not set device context: {}", e.getMessage());
            }
        }
        return -1;
    }

    /**
     * Restore the device context after op execution.
     *
     * @param originalDevice the device to restore, or -1 if no restore is needed
     */
    private void restoreDeviceContext(int originalDevice) {
        if (originalDevice >= 0) {
            try {
                DeviceMemoryManager.getInstance().switchDevice(originalDevice, "DeviceAwareOpExecutioner.exec", "restore-device");
            } catch (Exception e) {
                if (log.isTraceEnabled()) {
                    log.trace("Could not restore device context: {}", e.getMessage());
                }
            }
        }
    }

    /**
     * Helper class to save and restore device context around operations.
     * Use this for external code that needs to temporarily switch devices.
     * The exec/execAndReturn methods handle save/restore internally.
     */
    public static class DeviceContextScope implements AutoCloseable {
        private final int originalDevice;
        private final boolean switched;

        public DeviceContextScope(int targetDevice) {
            AffinityManager affinityManager = Nd4j.getAffinityManager();
            this.originalDevice = affinityManager.getDeviceForCurrentThread();
            this.switched = (originalDevice != targetDevice);

            if (switched) {
                DeviceMemoryManager.getInstance().switchDevice(targetDevice, "DeviceAwareOpExecutioner.exec", "target-device");
            }
        }

        @Override
        public void close() {
            if (switched) {
                DeviceMemoryManager.getInstance().switchDevice(originalDevice, "DeviceAwareOpExecutioner.exec", "restore-device");
            }
        }
    }

    // =========================================================================
    // Device Transfer Logic
    // =========================================================================

    /**
     * Prepare a CustomOp for execution by transferring inputs if needed.
     *
     * @return the target device for execution
     */
    private DeviceDescriptor prepareOpForExecution(CustomOp op) {
        if (!enabled) {
            return null;
        }

        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();
        if (!config.isAutoTransferEnabled()) {
            return getDefaultTargetDevice(op);
        }

        DeviceDescriptor targetDevice = delegator.prepareForExecution(op);

        if (config.isLogRoutingDecisions() && targetDevice != null) {
            log.debug("Op {} executing on device: {}", op.opName(), targetDevice.getDeviceId());
        }

        return targetDevice;
    }

    /**
     * Prepare a CustomOp with OpContext for execution.
     *
     * @return the target device for execution
     */
    private DeviceDescriptor prepareOpForExecution(CustomOp op, OpContext context) {
        if (!enabled || context == null) {
            return prepareOpForExecution(op);
        }

        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();
        if (!config.isAutoTransferEnabled()) {
            return getDefaultTargetDevice(op);
        }

        // Transfer inputs from context
        List<INDArray> inputs = context.getInputArrays();
        DeviceDescriptor targetDevice = null;

        if (inputs != null && !inputs.isEmpty()) {
            targetDevice = delegator.selectTargetDeviceForArrays(inputs);

            for (INDArray input : inputs) {
                if (input != null) {
                    delegator.transferArrayToDevice(input, targetDevice, config);
                }
            }
        }

        return targetDevice;
    }

    /**
     * Prepare a standard Op for execution.
     *
     * @return the target device for execution
     */
    private DeviceDescriptor prepareOpForExecution(Op op) {
        if (!enabled || op == null) {
            return null;
        }

        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();
        if (!config.isAutoTransferEnabled()) {
            return getDefaultTargetDevice(op);
        }

        // Determine target device from all operands (inputs + output)
        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        // For ops with a pre-allocated output (z != x), the output device is the
        // correct target. The output was allocated there intentionally (e.g., by
        // device routing) and migrating it back would defeat the purpose.
        // For in-place ops (z == x) or no output, use input device.
        DeviceDescriptor targetDevice = null;
        if (z != null && z != x) {
            targetDevice = getArrayDevice(z);
        }
        if (targetDevice == null) {
            List<INDArray> inputs = new ArrayList<>(2);
            if (x != null) inputs.add(x);
            if (y != null) inputs.add(y);
            if (!inputs.isEmpty()) {
                targetDevice = delegator.selectTargetDeviceForArrays(inputs);
            }
        }

        if (targetDevice == null) {
            return null;
        }

        // Transfer all operands to target device
        if (x != null) {
            delegator.transferArrayToDevice(x, targetDevice, config);
        }
        if (y != null) {
            delegator.transferArrayToDevice(y, targetDevice, config);
        }
        if (z != null && z != x) {
            delegator.transferArrayToDevice(z, targetDevice, config);
        }

        return targetDevice;
    }

    /**
     * Prepare a standard Op for execution using OpContext inputs when available.
     *
     * @return the target device for execution
     */
    private DeviceDescriptor prepareOpForExecution(Op op, OpContext context) {
        if (!enabled || op == null) {
            return null;
        }

        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();
        if (!config.isAutoTransferEnabled()) {
            return getDefaultTargetDevice(op);
        }

        if (context != null) {
            List<INDArray> inputs = context.getInputArrays();
            if (inputs != null && !inputs.isEmpty()) {
                DeviceDescriptor targetDevice = delegator.selectTargetDeviceForArrays(inputs);
                for (INDArray input : inputs) {
                    if (input != null) {
                        delegator.transferArrayToDevice(input, targetDevice, config);
                    }
                }
                return targetDevice;
            }
        }

        return prepareOpForExecution(op);
    }

    /**
     * Get the default target device based on op inputs.
     */
    private DeviceDescriptor getDefaultTargetDevice(CustomOp op) {
        if (op == null) return null;
        List<INDArray> inputs = op.inputArguments();
        if (inputs != null && !inputs.isEmpty()) {
            return delegator.selectTargetDeviceForArrays(inputs);
        }
        return null;
    }

    /**
     * Get the default target device based on op inputs.
     */
    private DeviceDescriptor getDefaultTargetDevice(Op op) {
        if (op == null) return null;
        List<INDArray> inputs = new ArrayList<>();
        if (op.x() != null) {
            inputs.add(op.x());
        }
        if (op.y() != null) {
            inputs.add(op.y());
        }
        if (op.z() != null) {
            inputs.add(op.z());
        }
        if (!inputs.isEmpty()) {
            return delegator.selectTargetDeviceForArrays(inputs);
        }
        return null;
    }

    /**
     * Get the device an array resides on.
     */
    private DeviceDescriptor getArrayDevice(INDArray array) {
        if (array == null) {
            return null;
        }

        DataBuffer buffer = array.data();
        if (buffer != null && buffer.isHybrid()) {
            DeviceDescriptor effective = buffer.asHybrid().getEffectiveDevice();
            return effective != null ? effective : DeviceDescriptor.cpu();
        }
        return DeviceDescriptor.cpu();
    }

    // =========================================================================
    // Delegate Methods (pass-through to underlying executioner)
    // =========================================================================

    @Override
    public Nd4jEventLog getNd4jEventLog() {
        return delegate.getNd4jEventLog();
    }

    @Override
    public boolean isVerbose() {
        return delegate.isVerbose();
    }

    @Override
    public boolean isDebug() {
        return delegate.isDebug();
    }

    @Override
    public ExecutionerType type() {
        return delegate.type();
    }

    @Override
    public OpContext injectNewContext() {
        return delegate.injectNewContext();
    }

    @Override
    public void clearOpContext() {
        delegate.clearOpContext();
    }

    @Override
    public void setNextOpContext(OpContext context) {
        delegate.setNextOpContext(context);
    }

    @Override
    public String getLastOp() {
        return delegate.getLastOp();
    }

    @Override
    public void push() {
        delegate.push();
    }

    @Override
    public void commit() {
        delegate.commit();
    }

    @Override
    public Map<String, CustomOpDescriptor> getCustomOperations() {
        return delegate.getCustomOperations();
    }

    @Override
    public CustomOp execAndReturn(CustomOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public INDArray[] allocateOutputArrays(CustomOp op) {
        return delegate.allocateOutputArrays(op);
    }

    @Override
    public void enableDebugMode(boolean reallyEnable) {
        delegate.enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        delegate.enableVerboseMode(reallyEnable);
    }

    @Override
    public void registerGraph(long id, Pointer flatBufferGraph) {
        delegate.registerGraph(id, flatBufferGraph);
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, Map<String, INDArray> map, Map<String, Integer> reverseMap) {
        return delegate.executeGraph(id, map, reverseMap);
    }

    @Override
    public void forgetGraph(long id) {
        delegate.forgetGraph(id);
    }

    @Override
    public TADManager getTADManager() {
        return delegate.getTADManager();
    }

    @Override
    public void setProfilingMode(ProfilingMode mode) {
        delegate.setProfilingMode(mode);
    }

    @Override
    public void setProfilingConfig(ProfilerConfig config) {
        delegate.setProfilingConfig(config);
    }

    @Override
    public ProfilingMode getProfilingMode() {
        return delegate.getProfilingMode();
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension) {
        return delegate.tadShapeInfoAndOffsets(array, dimension);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        return delegate.createShapeInfo(shape, stride, elementWiseStride, order, dtype, empty);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extras) {
        return delegate.createShapeInfo(shape, stride, elementWiseStride, order, dtype, extras);
    }

    @Override
    public OpContext buildContext() {
        return delegate.buildContext();
    }

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        return delegate.inspectArray(array);
    }

    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        return delegate.createConstantBuffer(values, desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType) {
        return delegate.createConstantBuffer(values, desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(int[] values, DataType desiredType) {
        return delegate.createConstantBuffer(values, desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(float[] values, DataType desiredType) {
        return delegate.createConstantBuffer(values, desiredType);
    }

    @Override
    public Properties getEnvironmentInformation() {
        return delegate.getEnvironmentInformation();
    }

    @Override
    public INDArray exec(Op op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.exec(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public TransformOp execAndReturn(TransformOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public ReduceOp execAndReturn(ReduceOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public Variance execAndReturn(Variance op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public ScalarOp execAndReturn(ScalarOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public BroadcastOp execAndReturn(BroadcastOp op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public Op execAndReturn(Op op) {
        DeviceDescriptor targetDevice = prepareOpForExecution(op);
        int originalDevice = setDeviceContextForExecution(targetDevice);
        try {
            OpExecutioner targetExecutioner = selectExecutionerForDevice(targetDevice);
            return targetExecutioner.execAndReturn(op);
        } finally {
            restoreDeviceContext(originalDevice);
        }
    }

    @Override
    public void exec(MetaOp op) {
        delegate.exec(op);
    }

    @Override
    public void exec(GridOp op) {
        delegate.exec(op);
    }

    @Override
    public void printEnvironmentInformation() {
        delegate.printEnvironmentInformation();
    }

    @Override
    public void setSkipDeviceCoherency(boolean skip) {
        delegate.setSkipDeviceCoherency(skip);
    }

    @Override
    public boolean isExperimentalMode() {
        return delegate.isExperimentalMode();
    }

    @Override
    public void setElementsThreshold(int threshold) {
        delegate.setElementsThreshold(threshold);
    }

    @Override
    public void setTadThreshold(int threshold) {
        delegate.setTadThreshold(threshold);
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        return delegate.getString(buffer, index);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride,
                                       char order, DataType dtype, boolean empty, boolean isView) {
        return delegate.createShapeInfo(shape, stride, elementWiseStride, order, dtype, empty, isView);
    }

    @Override
    public int useCount(DataBuffer buffer) {
        return delegate.useCount(buffer);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(CustomOp op) {
        return delegate.calculateOutputShape(op);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(CustomOp op, OpContext opContext) {
        return delegate.calculateOutputShape(op, opContext);
    }
}
