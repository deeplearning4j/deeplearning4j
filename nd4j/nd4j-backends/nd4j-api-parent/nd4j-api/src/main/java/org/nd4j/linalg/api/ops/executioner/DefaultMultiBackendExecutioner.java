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
import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceRoutingConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.routing.*;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.BackendRegistry;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.MultiBackendContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.ProfilerConfig;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Default implementation of MultiBackendExecutioner.
 * Routes operations to appropriate backend executioners based on routing policy.
 */
@Slf4j
public class DefaultMultiBackendExecutioner implements MultiBackendExecutioner {

    private final BackendRegistry registry;
    private final OpExecutioner primaryExecutioner;

    private volatile RoutingPolicy routingPolicy;
    private volatile boolean multiBackendEnabled = true;

    // Statistics
    private final AtomicLong totalOps = new AtomicLong(0);
    private final AtomicLong routedOps = new AtomicLong(0);
    private final AtomicLong transferCount = new AtomicLong(0);
    private final AtomicLong transferBytes = new AtomicLong(0);
    private final Map<String, AtomicLong> opsPerDevice = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> timePerDevice = new ConcurrentHashMap<>();

    public DefaultMultiBackendExecutioner(OpExecutioner primaryExecutioner) {
        this.primaryExecutioner = primaryExecutioner;
        this.registry = BackendRegistry.getInstance();
        this.routingPolicy = new DataLocalityPolicy();  // Default policy
    }

    @Override
    public void setRoutingPolicy(RoutingPolicy policy) {
        this.routingPolicy = policy;
    }

    @Override
    public RoutingPolicy getRoutingPolicy() {
        return routingPolicy;
    }

    @Override
    public RoutingDecision route(Op op) {
        // Check context for override
        RoutingPolicy contextPolicy = MultiBackendContext.currentPolicy();
        DeviceDescriptor contextDevice = MultiBackendContext.currentDevice();

        if (contextDevice != null) {
            return RoutingDecision.simple(contextDevice);
        }

        RoutingPolicy effectivePolicy = contextPolicy != null ? contextPolicy : routingPolicy;
        List<DeviceDescriptor> devices = getAvailableDevices();

        if (devices.isEmpty()) {
            throw new IllegalStateException("No devices available for routing");
        }

        return effectivePolicy.selectDevice(op, devices);
    }

    @Override
    public RoutingDecision route(CustomOp op) {
        // Check context for override
        RoutingPolicy contextPolicy = MultiBackendContext.currentPolicy();
        DeviceDescriptor contextDevice = MultiBackendContext.currentDevice();

        if (contextDevice != null) {
            return RoutingDecision.simple(contextDevice);
        }

        RoutingPolicy effectivePolicy = contextPolicy != null ? contextPolicy : routingPolicy;
        List<DeviceDescriptor> devices = getAvailableDevices();

        if (devices.isEmpty()) {
            throw new IllegalStateException("No devices available for routing");
        }

        return effectivePolicy.selectDevice(op, devices);
    }

    @Override
    public INDArray exec(Op op, DeviceDescriptor device) {
        OpExecutioner executioner = getExecutioner(device);
        if (executioner == null) {
            throw new IllegalArgumentException("No executioner for device: " + device.getDeviceId());
        }

        long startTime = System.nanoTime();
        try {
            return executioner.exec(op);
        } finally {
            recordExecution(device, System.nanoTime() - startTime);
        }
    }

    @Override
    public INDArray[] exec(CustomOp op, DeviceDescriptor device) {
        OpExecutioner executioner = getExecutioner(device);
        if (executioner == null) {
            throw new IllegalArgumentException("No executioner for device: " + device.getDeviceId());
        }

        long startTime = System.nanoTime();
        try {
            return executioner.exec(op);
        } finally {
            recordExecution(device, System.nanoTime() - startTime);
        }
    }

    @Override
    public INDArray exec(Op op) {
        totalOps.incrementAndGet();

        if (!multiBackendEnabled || getAvailableDevices().size() <= 1) {
            return primaryExecutioner.exec(op);
        }

        RoutingDecision decision = route(op);
        routedOps.incrementAndGet();

        // Execute transfers if needed
        executeTransfers(decision);

        // Execute on target device
        return exec(op, decision.getTargetDevice());
    }

    @Override
    public INDArray[] exec(CustomOp op) {
        totalOps.incrementAndGet();

        if (!multiBackendEnabled || getAvailableDevices().size() <= 1) {
            return primaryExecutioner.exec(op);
        }

        RoutingDecision decision = route(op);
        routedOps.incrementAndGet();

        // Execute transfers if needed
        executeTransfers(decision);

        // Execute on target device
        return exec(op, decision.getTargetDevice());
    }

    private void executeTransfers(RoutingDecision decision) {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();

        for (RoutingDecision.TransferInfo transfer : decision.getRequiredTransfers()) {
            DeviceDescriptor sourceDevice = getArrayDevice(transfer.getArray());
            DeviceDescriptor targetDevice = transfer.getTargetDevice();

            // Track transfer with timing
            long bytes = transfer.getBytes();
            long startNanos = System.nanoTime();

            transferTo(transfer.getArray(), targetDevice);

            long endNanos = System.nanoTime();
            long transferTimeNanos = endNanos - startNanos;

            // Update local statistics
            transferCount.incrementAndGet();
            transferBytes.addAndGet(bytes);

            // Record in TransferMetrics for detailed tracking
            TransferMetrics.getInstance().recordTransfer(sourceDevice, targetDevice, bytes, transferTimeNanos);

            // Log if configured
            if (config.isLogDataTransfers() && bytes >= config.getMinBytesForTransferLogging()) {
                double bandwidthGBps = transferTimeNanos > 0 ? (bytes / 1e9) / (transferTimeNanos / 1e9) : 0;
                log.info("[EXEC TRANSFER] {} -> {} | {} bytes | {:.2f} ms | {:.2f} GB/s",
                        sourceDevice != null ? sourceDevice.getDeviceId() : "unknown",
                        targetDevice.getDeviceId(),
                        bytes,
                        transferTimeNanos / 1_000_000.0,
                        bandwidthGBps);
            }
        }
    }

    /**
     * Get the device where an array currently resides.
     */
    private DeviceDescriptor getArrayDevice(INDArray array) {
        if (array == null || array.data() == null) {
            return null;
        }

        if (array.data().isHybrid()) {
            DeviceDescriptor effective = array.data().asHybrid().getEffectiveDevice();
            return effective != null ? effective : DeviceDescriptor.cpu();
        }

        // Default to CPU if not hybrid
        return DeviceDescriptor.cpu();
    }

    private void recordExecution(DeviceDescriptor device, long timeNanos) {
        String deviceId = device.getDeviceId();
        opsPerDevice.computeIfAbsent(deviceId, k -> new AtomicLong(0)).incrementAndGet();
        timePerDevice.computeIfAbsent(deviceId, k -> new AtomicLong(0)).addAndGet(timeNanos);

        // Record in TransferMetrics for overhead calculation
        TransferMetrics.getInstance().recordOpExecution(device, timeNanos);
    }

    @Override
    public List<DeviceDescriptor> getAvailableDevices() {
        return registry.getAllDevices();
    }

    @Override
    public OpExecutioner getExecutioner(DeviceDescriptor device) {
        return registry.getExecutioner(device);
    }

    @Override
    public INDArray transferTo(INDArray array, DeviceDescriptor device) {
        if (array == null || array.data() == null) {
            return array;
        }

        // If hybrid buffer, use its transfer mechanism
        if (array.data().isHybrid()) {
            DeviceDescriptor pinned = array.data().asHybrid().getPinnedDevice();
            if (pinned != null && !pinned.getDeviceId().equals(device.getDeviceId())) {
                throw new IllegalStateException("Array is pinned to " + pinned.getDeviceId()
                        + " and cannot be transferred to " + device.getDeviceId());
            }
            array.data().asHybrid().ensureReadableOn(device);
        }

        // Otherwise, the array stays on its current device
        // Actual transfer implementation depends on backend specifics
        return array;
    }

    @Override
    public void prefetch(INDArray array, DeviceDescriptor device) {
        if (array == null || array.data() == null) {
            return;
        }

        if (array.data().isHybrid()) {
            DeviceDescriptor pinned = array.data().asHybrid().getPinnedDevice();
            if (pinned != null && !pinned.getDeviceId().equals(device.getDeviceId())) {
                throw new IllegalStateException("Array is pinned to " + pinned.getDeviceId()
                        + " and cannot be prefetched to " + device.getDeviceId());
            }
            array.data().asHybrid().prefetch(device);
        }
    }

    @Override
    public void syncAll() {
        primaryExecutioner.commit();
        // Sync other executioners
        for (DeviceDescriptor device : getAvailableDevices()) {
            sync(device);
        }
    }

    @Override
    public void sync(DeviceDescriptor device) {
        OpExecutioner executioner = getExecutioner(device);
        if (executioner != null) {
            executioner.commit();
        }
    }

    @Override
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalOps", totalOps.get());
        stats.put("routedOps", routedOps.get());
        stats.put("transferCount", transferCount.get());
        stats.put("transferBytes", transferBytes.get());
        stats.put("opsPerDevice", new HashMap<>(opsPerDevice));
        stats.put("timePerDevice", new HashMap<>(timePerDevice));

        // Include TransferMetrics data
        TransferMetrics metrics = TransferMetrics.getInstance();
        stats.put("transferMetrics", metrics.getStatistics());
        stats.put("transferOverheadPercent", metrics.getOverheadPercent());
        stats.put("averageBandwidthGBps", metrics.getAverageBandwidthGBps());

        return stats;
    }

    @Override
    public void resetStatistics() {
        totalOps.set(0);
        routedOps.set(0);
        transferCount.set(0);
        transferBytes.set(0);
        opsPerDevice.clear();
        timePerDevice.clear();

        // Reset TransferMetrics as well
        TransferMetrics.getInstance().reset();
    }

    /**
     * Log a summary of transfer metrics.
     * Useful for debugging transfer overhead issues.
     */
    public void logTransferMetricsSummary() {
        TransferMetrics.getInstance().logSummary();
    }

    /**
     * Get the transfer overhead percentage.
     *
     * @return percentage of execution time spent on transfers
     */
    public double getTransferOverheadPercent() {
        return TransferMetrics.getInstance().getOverheadPercent();
    }

    /**
     * Check if transfer overhead exceeds the warning threshold.
     *
     * @return true if overhead exceeds threshold
     */
    public boolean isTransferOverheadHigh() {
        DeviceRoutingConfiguration config = DeviceRoutingConfiguration.current();
        return TransferMetrics.getInstance().getOverheadPercent() > config.getTransferOverheadWarningThreshold();
    }

    @Override
    public boolean isMultiBackendEnabled() {
        return multiBackendEnabled;
    }

    @Override
    public void setMultiBackendEnabled(boolean enabled) {
        this.multiBackendEnabled = enabled;
    }

    // Delegate remaining OpExecutioner methods to primary executioner

    @Override
    public ExecutionerType type() {
        return ExecutionerType.MULTI_BACKEND;
    }

    @Override
    public Nd4jEventLog getNd4jEventLog() {
        return primaryExecutioner.getNd4jEventLog();
    }

    @Override
    public boolean isVerbose() {
        return primaryExecutioner.isVerbose();
    }

    @Override
    public boolean isDebug() {
        return primaryExecutioner.isDebug();
    }

    @Override
    public OpContext injectNewContext() {
        return primaryExecutioner.injectNewContext();
    }

    @Override
    public void clearOpContext() {
        primaryExecutioner.clearOpContext();
    }

    @Override
    public void setNextOpContext(OpContext context) {
        primaryExecutioner.setNextOpContext(context);
    }

    @Override
    public String getLastOp() {
        return primaryExecutioner.getLastOp();
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        return primaryExecutioner.exec(op, opContext);
    }

    @Override
    public TransformOp execAndReturn(TransformOp op) {
        exec(op);
        return op;
    }

    @Override
    public ReduceOp execAndReturn(ReduceOp op) {
        exec(op);
        return op;
    }

    @Override
    public Variance execAndReturn(Variance op) {
        exec(op);
        return op;
    }

    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        exec(op);
        return op;
    }

    @Override
    public ScalarOp execAndReturn(ScalarOp op) {
        exec(op);
        return op;
    }

    @Override
    public BroadcastOp execAndReturn(BroadcastOp op) {
        exec(op);
        return op;
    }

    @Override
    public INDArray exec(ReduceOp reduceOp) {
        return primaryExecutioner.exec(reduceOp);
    }

    @Override
    public INDArray exec(BroadcastOp broadcast) {
        return primaryExecutioner.exec(broadcast);
    }

    @Override
    public INDArray exec(ScalarOp broadcast) {
        return primaryExecutioner.exec(broadcast);
    }

    @Override
    public INDArray exec(Variance accumulation) {
        return primaryExecutioner.exec(accumulation);
    }

    @Override
    public INDArray exec(IndexAccumulation indexAccum) {
        return primaryExecutioner.exec(indexAccum);
    }

    @Override
    public Op execAndReturn(Op op) {
        exec(op);
        return op;
    }

    @Override
    public void exec(MetaOp op) {
        primaryExecutioner.exec(op);
    }

    @Override
    public void exec(GridOp op) {
        primaryExecutioner.exec(op);
    }

    @Override
    public INDArray exec(RandomOp op) {
        return primaryExecutioner.exec(op);
    }

    @Override
    public INDArray exec(RandomOp op, Random rng) {
        return primaryExecutioner.exec(op, rng);
    }

    @Override
    public Properties getEnvironmentInformation() {
        return primaryExecutioner.getEnvironmentInformation();
    }

    @Override
    public void setProfilingMode(ProfilingMode mode) {
        primaryExecutioner.setProfilingMode(mode);
    }

    @Override
    public void setProfilingConfig(ProfilerConfig config) {
        primaryExecutioner.setProfilingConfig(config);
    }

    @Override
    public ProfilingMode getProfilingMode() {
        return primaryExecutioner.getProfilingMode();
    }

    @Override
    public TADManager getTADManager() {
        return primaryExecutioner.getTADManager();
    }

    @Override
    public void printEnvironmentInformation() {
        primaryExecutioner.printEnvironmentInformation();
    }

    @Override
    public void push() {
        primaryExecutioner.push();
    }

    @Override
    public void commit() {
        primaryExecutioner.commit();
    }

    @Override
    public Map<String, CustomOpDescriptor> getCustomOperations() {
        return primaryExecutioner.getCustomOperations();
    }

    @Override
    public CustomOp execAndReturn(CustomOp op) {
        exec(op);
        return op;
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context) {
        return primaryExecutioner.exec(op, context);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(CustomOp op) {
        return primaryExecutioner.calculateOutputShape(op);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(CustomOp op, OpContext opContext) {
        return primaryExecutioner.calculateOutputShape(op, opContext);
    }

    @Override
    public INDArray[] allocateOutputArrays(CustomOp op) {
        return primaryExecutioner.allocateOutputArrays(op);
    }

    @Override
    public void enableDebugMode(boolean reallyEnable) {
        primaryExecutioner.enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        primaryExecutioner.enableVerboseMode(reallyEnable);
    }

    @Override
    public boolean isExperimentalMode() {
        return primaryExecutioner.isExperimentalMode();
    }

    @Override
    public void registerGraph(long id, Pointer graph) {
        primaryExecutioner.registerGraph(id, graph);
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, Map<String, INDArray> map, Map<String, Integer> reverseMap) {
        return primaryExecutioner.executeGraph(id, map, reverseMap);
    }

    @Override
    public void forgetGraph(long id) {
        primaryExecutioner.forgetGraph(id);
    }

    @Override
    public void setElementsThreshold(int threshold) {
        primaryExecutioner.setElementsThreshold(threshold);
    }

    @Override
    public void setTadThreshold(int threshold) {
        primaryExecutioner.setTadThreshold(threshold);
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        return primaryExecutioner.getString(buffer, index);
    }

    @Override
    public OpContext buildContext() {
        return primaryExecutioner.buildContext();
    }

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        return primaryExecutioner.inspectArray(array);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty, boolean isView) {
        return primaryExecutioner.createShapeInfo(shape, stride, elementWiseStride, order, dtype, empty, isView);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        return primaryExecutioner.createShapeInfo(shape, stride, elementWiseStride, order, dtype, empty);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extra) {
        return primaryExecutioner.createShapeInfo(shape, stride, elementWiseStride, order, dtype, extra);
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension) {
        return primaryExecutioner.tadShapeInfoAndOffsets(array, dimension);
    }

    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        return primaryExecutioner.createConstantBuffer(values, desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(int[] values, DataType desiredType) {
        return primaryExecutioner.createConstantBuffer(values, desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(float[] values, DataType desiredType) {
        return primaryExecutioner.createConstantBuffer(values, desiredType);
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType) {
        return primaryExecutioner.createConstantBuffer(values, desiredType);
    }

    @Override
    public int useCount(DataBuffer buffer) {
        return primaryExecutioner.useCount(buffer);
    }
}
