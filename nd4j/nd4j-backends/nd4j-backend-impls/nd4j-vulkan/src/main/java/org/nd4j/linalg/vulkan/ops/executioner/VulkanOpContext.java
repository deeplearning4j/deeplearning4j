/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan.ops.executioner;

import lombok.NonNull;
import org.bytedeco.javacpp.BooleanPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.ExecutionMode;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.vulkan.VulkanAffinityManager;
import org.nd4j.linalg.vulkan.VulkanDataBufferFactory;
import org.nd4j.linalg.vulkan.VulkanNDArray;
import org.nd4j.linalg.vulkan.VulkanRuntime;
import org.nd4j.linalg.vulkan.bindings.Nd4jVulkan;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.nativeblas.OpaqueRandomGenerator;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Vulkan wrapper around libnd4j's native graph context.
 *
 * <p>This is the Vulkan counterpart of CUDA's device op context. It owns native
 * graph metadata and Vulkan array references; it does not authorize eager CPU
 * execution.</p>
 */
public final class VulkanOpContext extends BaseOpContext implements OpContext, Deallocatable {
    private final VulkanRuntime runtime;
    private final Nd4jVulkan nativeOps;
    private final VulkanAffinityManager affinityManager;
    private final VulkanExecutioner executioner;
    private final VulkanDataBufferFactory dataBufferFactory;
    private volatile OpaqueContext context;
    private volatile boolean closed;
    private final transient long id;
    private long deallocationId;
    private final int deviceId;

    private final Map<Integer, INDArray> singleInputArrayRefs = new HashMap<>();
    private final Map<Integer, INDArray> singleOutputArrayRefs = new HashMap<>();
    private final Map<Integer, OpaqueNDArray> inputOpaqueArrayRefs = new HashMap<>();
    private final Map<Integer, OpaqueNDArray> outputOpaqueArrayRefs = new HashMap<>();

    public VulkanOpContext() {
        this(VulkanRuntime.getInstance());
    }

    public VulkanOpContext(VulkanRuntime runtime) {
        if (runtime == null) {
            throw new IllegalArgumentException("VulkanRuntime cannot be null");
        }
        this.runtime = runtime;
        this.nativeOps = runtime.nativeOps();
        this.affinityManager = runtime.affinityManager();
        this.executioner = runtime.executioner();
        this.dataBufferFactory = runtime.dataBufferFactory();
        this.context = OpaqueContext.create(runtime, 1);
        this.deviceId = runtime.currentDevice();

        OpaqueLaunchContext launchContext = nativeOps.defaultLaunchContext();
        if (launchContext == null || launchContext.isNull()) {
            context.close();
            throw new IllegalStateException("Vulkan backend returned no launch context");
        }
        Pointer stream = nativeOps.lcExecutionStream(launchContext);
        if (stream == null || stream.isNull()) {
            context.close();
            throw new IllegalStateException("Vulkan backend returned no execution stream");
        }
        nativeOps.clearLastError();
        nativeOps.setGraphContextCudaContext(context, stream, null, null);
        int errorCode = nativeOps.lastErrorCode();
        if (errorCode != 0) {
            String errorMessage = nativeOps.lastErrorMessage();
            nativeOps.clearLastError();
            context.close();
            throw new IllegalStateException(
                    "Could not bind Vulkan graph context to its execution stream (native error "
                            + errorCode + "): " + errorMessage);
        }

        this.id = runtime.deallocatorService().nextValue();
        this.deallocationId = runtime.deallocatorService().pickObject(this);
    }

    public VulkanOpContext(NativeOps nativeOps, VulkanAffinityManager affinityManager,
                           OpExecutioner executioner, VulkanDataBufferFactory dataBufferFactory) {
        this(requireRuntime(nativeOps, affinityManager, executioner, dataBufferFactory));
    }

    private static VulkanRuntime requireRuntime(
            NativeOps nativeOps, VulkanAffinityManager affinityManager,
            OpExecutioner executioner, VulkanDataBufferFactory dataBufferFactory) {
        VulkanRuntime runtime = VulkanRuntime.forNativeOps(nativeOps);
        if (affinityManager != runtime.affinityManager()
                || executioner != runtime.executioner()
                || dataBufferFactory != runtime.dataBufferFactory()) {
            throw new IllegalArgumentException(
                    "Vulkan op-context services must belong to the selected VulkanRuntime");
        }
        return runtime;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }

        synchronized (this) {
            if (closed) {
                return;
            }

            int currentDevice = runtime.currentDevice();
            boolean switched = currentDevice != deviceId;
            if (switched) {
                runtime.setDevice(deviceId);
            }

            try {
                executioner.commit();
                purge();
                clearArrayReferences();
                runtime.deallocatorService().getReferenceMap().remove(deallocationId);
                if (context != null) {
                    context.close();
                }
            } finally {
                closed = true;
                if (switched) {
                    runtime.setDevice(currentDevice);
                }
            }
        }
    }

    private void clearArrayReferences() {
        closeOpaqueArrays(inputOpaqueArrayRefs);
        closeOpaqueArrays(outputOpaqueArrayRefs);
        singleInputArrayRefs.clear();
        singleOutputArrayRefs.clear();
    }

    private static void closeOpaqueArrays(Map<Integer, OpaqueNDArray> arrays) {
        RuntimeException failure = null;
        for (OpaqueNDArray array : arrays.values()) {
            try {
                if (array != null) {
                    array.close();
                }
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        arrays.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public void setIArguments(long... arguments) {
        super.setIArguments(arguments);
        nativeOps.setGraphContextIArguments(
                context, arguments.length == 0 ? new LongPointer(0) : new LongPointer(arguments), arguments.length);
    }

    @Override
    public void setBArguments(boolean... arguments) {
        super.setBArguments(arguments);
        nativeOps.setGraphContextBArguments(
                context, arguments.length == 0 ? new BooleanPointer(0) : new BooleanPointer(arguments), arguments.length);
    }

    @Override
    public void setTArguments(double... arguments) {
        super.setTArguments(arguments);
        nativeOps.setGraphContextTArguments(
                context, arguments.length == 0 ? new DoublePointer(0) : new DoublePointer(arguments), arguments.length);
    }

    @Override
    public void setDArguments(DataType... arguments) {
        super.setDArguments(arguments);
        int[] nativeTypes = new int[arguments.length];
        for (int i = 0; i < arguments.length; i++) {
            nativeTypes[i] = arguments[i].toInt();
        }
        nativeOps.setGraphContextDArguments(
                context, nativeTypes.length == 0 ? new IntPointer(0) : new IntPointer(nativeTypes), nativeTypes.length);
    }

    @Override
    public void setSArguments(String... arguments) {
        super.setSArguments(arguments);
        for (int i = 0; i < arguments.length; i++) {
            nativeOps.setGraphContextSArgument(context, arguments[i], i);
        }
    }

    @Override
    public void setInputArrays(@NonNull List<INDArray> arrays) {
        int index = 0;
        for (INDArray array : arrays) {
            if (array == null) {
                continue;
            }
            setInputArray(index++, array);
        }
    }

    @Override
    public void setOutputArrays(@NonNull List<INDArray> arrays) {
        int index = 0;
        for (INDArray array : arrays) {
            if (array == null) {
                continue;
            }
            setOutputArray(index++, array);
        }
    }

    @Override
    public void setInputArrays(INDArray... arrays) {
        setInputArrays(Arrays.asList(arrays));
    }

    @Override
    public void setOutputArrays(INDArray... arrays) {
        setOutputArrays(Arrays.asList(arrays));
    }

    @Override
    public void setInputArray(int index, @NonNull INDArray array) {
        singleInputArrayRefs.put(index, array);
        OpaqueNDArray opaqueArray = OpaqueNDArray.fromINDArray(runtime, array);
        OpaqueNDArray previous = inputOpaqueArrayRefs.put(index, opaqueArray);
        nativeOps.setGraphContextInputArray(context, index, opaqueArray);
        if (previous != null) {
            previous.close();
        }
        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        singleOutputArrayRefs.put(index, array);
        OpaqueNDArray opaqueArray = OpaqueNDArray.fromINDArrayNoSync(runtime, array);
        OpaqueNDArray previous = outputOpaqueArrayRefs.put(index, opaqueArray);
        nativeOps.setGraphContextOutputArray(context, index, opaqueArray);
        if (previous != null) {
            previous.close();
        }
        super.setOutputArray(index, array);
    }

    @Override
    public long id() {
        return id;
    }

    @Override
    public void setIArguments(Pointer arguments, int length) {
        nativeOps.setGraphContextIArguments(
                context, arguments instanceof LongPointer ? (LongPointer) arguments : new LongPointer(arguments), length);
    }

    @Override
    public void setTArguments(Pointer arguments, int length) {
        nativeOps.setGraphContextTArguments(
                context, arguments instanceof DoublePointer ? (DoublePointer) arguments : new DoublePointer(arguments), length);
    }

    @Override
    public void setDArguments(Pointer arguments, int length) {
        nativeOps.setGraphContextDArguments(
                context, arguments instanceof IntPointer ? (IntPointer) arguments : new IntPointer(arguments), length);
    }

    @Override
    public void setBArguments(Pointer arguments, int length) {
        nativeOps.setGraphContextBArguments(
                context, arguments instanceof BooleanPointer ? (BooleanPointer) arguments : new BooleanPointer(arguments), length);
    }

    @Override
    public int numIntermediateResults() {
        return nativeOps.numIntermediateResults(context);
    }

    @Override
    public void setIntermediateResult(int index, INDArray array) {
        if (array == null) {
            throw new IllegalArgumentException("Intermediate result " + index + " cannot be null");
        }
        nativeOps.setIntermediateResult(
                context, index, array.data().opaqueBuffer(), array.shapeInfoDataBuffer().opaqueBuffer(), array.offset());
    }

    @Override
    public INDArray getIntermediateResult(int index) {
        if (context == null || context.isNull()) {
            throw new IllegalStateException("Vulkan op context is closed");
        }

        LongPointer shapeInfo = nativeOps.intermediateResultShapeInfoAt(index, context);
        if (shapeInfo == null || shapeInfo.isNull()) {
            throw new IllegalStateException("No intermediate result shape at index " + index);
        }

        int shapeInfoLength = Shape.shapeInfoLength((int) shapeInfo.get(0));
        long[] javaShapeInfo = new long[shapeInfoLength];
        shapeInfo.capacity(shapeInfoLength).get(javaShapeInfo, 0, shapeInfoLength);
        DataBuffer shapeBuffer = dataBufferFactory.createLong(javaShapeInfo);

        OpaqueDataBuffer opaqueBuffer = nativeOps.intermediateResultDataAt(index, context);
        int resultDevice = nativeOps.dbDeviceId(opaqueBuffer);
        opaqueBuffer.attachOwner(
                runtime, resultDevice >= 0 ? runtime.deviceDescriptor(resultDevice) : null);
        long length = nativeOps.dbBufferLength(opaqueBuffer);
        Pointer primary = opaqueBuffer.primaryBuffer();
        Pointer special = opaqueBuffer.specialBuffer();
        if (primary != null && !primary.isNull()) {
            primary.capacity(length);
        }
        if (special != null && !special.isNull()) {
            special.capacity(length);
        }

        DataBuffer data = dataBufferFactory.create(
                primary, special, Shape.dataType(javaShapeInfo), Shape.length(javaShapeInfo), null);
        VulkanNDArray result = new VulkanNDArray();
        result.setShapeInfoDataBuffer(shapeBuffer);
        result.setData(data);
        return result;
    }

    @Override
    public void addIntermediateResult(INDArray array) {
        nativeOps.pushIntermediateResult(
                context, array.data().opaqueBuffer(), array.shapeInfoDataBuffer().opaqueBuffer(), array.offset());
    }

    @Override
    public void setRngStates(long rootState, long nodeState) {
        nativeOps.setRandomGeneratorStates(
                nativeOps.getGraphContextRandomGenerator(context), rootState, nodeState);
    }

    @Override
    public Pair<Long, Long> getRngStates() {
        OpaqueRandomGenerator generator = nativeOps.getGraphContextRandomGenerator(context);
        return Pair.makePair(
                nativeOps.getRandomGeneratorRootState(generator),
                nativeOps.getRandomGeneratorNodeState(generator));
    }

    @Override
    public OpaqueContext contextPointer() {
        return context;
    }

    Nd4jVulkan nativeOpsAuthority() {
        return nativeOps;
    }

    VulkanAffinityManager affinityManagerAuthority() {
        return affinityManager;
    }

    VulkanExecutioner executionerAuthority() {
        return executioner;
    }

    @Override
    public void markInplace(boolean inplace) {
        nativeOps.markGraphContextInplace(context, inplace);
    }

    @Override
    public void allowHelpers(boolean allow) {
        nativeOps.ctxAllowHelpers(context, allow);
    }

    @Override
    public void shapeFunctionOverride(boolean override) {
        nativeOps.ctxShapeFunctionOverride(context, override);
    }

    @Override
    public void setExecutionMode(@NonNull ExecutionMode mode) {
        super.setExecutionMode(mode);
        nativeOps.ctxSetExecutionMode(context, mode.ordinal());
    }

    @Override
    public void purge() {
        if (closed) {
            return;
        }
        executioner.commit();
        super.purge();
        if (context != null && !context.isNull()) {
            nativeOps.ctxPurge(context);
        }
    }

    @Override
    public void purgeForReuse() {
        if (closed) {
            return;
        }
        super.purgeForReuse();
        clearArrayReferences();
        if (context != null && !context.isNull()) {
            nativeOps.ctxPurgeNoSync(context);
        }
    }

    @Override
    public long getUniqueId() {
        return id;
    }

    @Override
    public Deallocator deallocator() {
        return new VulkanOpContextDeallocator(this);
    }

    @Override
    public int targetDevice() {
        return deviceId;
    }

    @Override
    public void attachWorkspace(Pointer workspacePointer) {
        if (context != null && workspacePointer != null) {
            nativeOps.attachWorkspaceToContext(context, workspacePointer);
        }
    }

    @Override
    public void detachWorkspace() {
        if (context != null) {
            nativeOps.detachWorkspaceFromContext(context);
        }
    }
}
