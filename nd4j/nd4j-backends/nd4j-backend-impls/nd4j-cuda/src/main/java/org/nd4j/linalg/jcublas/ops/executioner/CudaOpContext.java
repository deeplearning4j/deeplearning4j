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

package org.nd4j.linalg.jcublas.ops.executioner;

import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.*;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.ExecutionMode;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.*;

import java.util.Arrays;
import java.util.List;

/**
 * CUDA wrapper for op Context
 * @author raver119@gmail.com
 */
public class CudaOpContext extends BaseOpContext implements OpContext, Deallocatable {
    // we might want to have configurable
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private volatile OpaqueContext context = nativeOps.createGraphContext(1);
    // Track whether close() was already called to prevent double-free
    private volatile boolean closed = false;
    private final transient long id = Nd4j.getDeallocatorService().nextValue();
    public final static long BASE_CUDA_OP_CONTEXT_OFFSET = RandomUtils.nextLong();
    private long deallocationId;

    private final int deviceId;

    // Keep strong references to INDArrays passed via single-array setters.
    // This prevents GC from collecting them while this OpContext is alive.
    private final java.util.Map<Integer, INDArray> singleInputArrayRefs = new java.util.HashMap<>();
    private final java.util.Map<Integer, INDArray> singleOutputArrayRefs = new java.util.HashMap<>();

    private final java.util.Map<Integer, OpaqueNDArray> inputOpaqueArrayRefs = new java.util.HashMap<>();
    private final java.util.Map<Integer, OpaqueNDArray> outputOpaqueArrayRefs = new java.util.HashMap<>();


    public CudaOpContext() {
        this.deviceId = AtomicAllocator.getInstance().getDeviceId();
        this.deallocationId = Nd4j.getDeallocatorService().pickObject(this);
    }

    @Override
    public void close() {
        // Prevent double-close which would cause double-free
        if (closed) {
            return;
        }

        synchronized (this) {
            // Double-check inside synchronized block - return if already closed
            if (closed) {
                return;
            }

            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            boolean switched = false;
            if (currentDevice != deviceId) {
                DeviceMemoryManager.getInstance().switchDevice(deviceId, "CudaOpContext.close", "device-cleanup");
                switched = true;
            }
            try {
                Nd4j.getExecutioner().commit();
                purge();
                singleInputArrayRefs.clear();
                singleOutputArrayRefs.clear();
                inputOpaqueArrayRefs.clear();
                outputOpaqueArrayRefs.clear();
                Nd4j.getDeallocatorService().getReferenceMap().remove(this.deallocationId);
                if (context != null && !context.isNull()) {
                    nativeOps.deleteGraphContext(context);
                    context.setNull();
                }
            } finally {
                closed = true;
                if (switched) {
                    DeviceMemoryManager.getInstance().switchDevice(currentDevice, "CudaOpContext.close", "restore-device");
                }
            }
        }
    }

    @Override
    public void setIArguments(long... arguments) {
        super.setIArguments(arguments);
        LongPointer iArgs = arguments.length > 0 ? new LongPointer(arguments) : new LongPointer(0);
        nativeOps.setGraphContextIArguments(context, iArgs, arguments.length);
    }

    @Override
    public void setBArguments(boolean... arguments) {
        super.setBArguments(arguments);
        BooleanPointer bArgs = arguments.length > 0 ? new BooleanPointer(arguments) : new BooleanPointer(0);
        nativeOps.setGraphContextBArguments(context, bArgs, arguments.length);
    }

    @Override
    public void setTArguments(double... arguments) {
        super.setTArguments(arguments);
        DoublePointer tArgs = arguments.length > 0 ? new DoublePointer(arguments) : new DoublePointer(0);
        nativeOps.setGraphContextTArguments(context, tArgs, arguments.length);
    }

    @Override
    public void setDArguments(DataType... arguments) {
        super.setDArguments(arguments);
        val args = new int[arguments.length];
        for (int e = 0; e < arguments.length; e++)
            args[e] = arguments[e].toInt();

        IntPointer dArgs = args.length > 0 ? new IntPointer(args) : new IntPointer(0);
        nativeOps.setGraphContextDArguments(context, dArgs, arguments.length);
    }

    @Override
    public void setInputArrays(@NonNull List<INDArray> arrays) {
        int idx = 0;
        for (int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            if (array == null) {
                continue;  // Skip null inputs
            }
            // Note: OpaqueNDArray.fromINDArray() already calls syncToSpecial()
            // so we don't need a separate sync here.
            fastpath_in.put(idx, array);
            singleInputArrayRefs.put(idx, array);
            OpaqueNDArray opaqueArray = OpaqueNDArray.fromINDArray(array);
            inputOpaqueArrayRefs.put(idx, opaqueArray);
            nativeOps.setGraphContextInputArray(context, idx, opaqueArray);
            idx++;
        }
    }

    @Override
    public void setOutputArrays(@NonNull List<INDArray> arrays) {
        // Use the single-array approach like CpuOpContext - bypasses OpaqueNDArrayArr issues
        int idx = 0;
        for (int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            if (array == null) {
                continue;  // Skip null outputs
            }
            fastpath_out.put(idx, array);
            singleOutputArrayRefs.put(idx, array);
            OpaqueNDArray opaqueArray = OpaqueNDArray.fromINDArray(array);
            outputOpaqueArrayRefs.put(idx, opaqueArray);
            nativeOps.setGraphContextOutputArray(context, idx, opaqueArray);
            idx++;
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
    public long id() {
        return id;
    }

    @Override
    public void setIArguments(Pointer arguments, int length) {
        LongPointer iArgs = arguments instanceof LongPointer ?(LongPointer) arguments : new LongPointer(arguments);
        nativeOps.setGraphContextIArguments(context, iArgs,length);

    }

    @Override
    public void setTArguments(Pointer arguments, int length) {
        DoublePointer tArgs = arguments instanceof DoublePointer ?(DoublePointer) arguments : new DoublePointer(arguments);
        nativeOps.setGraphContextTArguments(context, tArgs,length);
    }

    @Override
    public void setDArguments(Pointer arguments, int length) {
        IntPointer dArgs = arguments instanceof IntPointer ?(IntPointer) arguments : new IntPointer(arguments);
        nativeOps.setGraphContextDArguments(context,dArgs,length);
    }

    @Override
    public int numIntermediateResults() {
        return Nd4j.getNativeOps().numIntermediateResults(context);
    }

    @Override
    public void setIntermediateResult(int index, INDArray arr) {
        if(arr == null) {
            throw new IllegalArgumentException("Unable to set intermediate result for index " + index + " with null array");
        }
        Nd4j.getNativeOps().setIntermediateResult(
                context,
                index,
                arr.data().opaqueBuffer(),
                arr.shapeInfoDataBuffer().opaqueBuffer(),
                arr.offset());
    }

    @Override
    public INDArray getIntermediateResult(int index) {
        if (context == null || context.isNull()) {
            throw new IllegalStateException("Cannot get intermediate result: OpContext native context is null");
        }
        LongPointer shapeInfo = nativeOps.intermediateResultShapeInfoAt(index,context);
        if (shapeInfo == null || shapeInfo.isNull()) {
            throw new IllegalStateException("Failed to retrieve intermediate result shape info at index " + index + ": returned null");
        }
        long rank = shapeInfo.get(0);
        shapeInfo.capacity(Shape.shapeInfoLength(rank));
        DataBuffer shapeInfoBuffer = Nd4j.createBuffer(shapeInfo, shapeInfo.capacity(),DataType.LONG);
        long[] convert = shapeInfoBuffer.asLong();
        OpaqueDataBuffer buffer = nativeOps.intermediateResultDataAt(index,context);
        long numElements = nativeOps.dbBufferLength(buffer);
        Pointer pointer = buffer.primaryBuffer();
        pointer.capacity(numElements);
        DataBuffer firstBuffer = Nd4j.createBuffer(pointer,null,
                Shape.length(convert), Shape.dataType(convert));
        INDArray result = Nd4j.createArrayFromShapeBuffer(firstBuffer,shapeInfoBuffer);
        return result;
    }

    @Override
    public void addIntermediateResult(INDArray arr) {
        Nd4j.getNativeOps().pushIntermediateResult(context,
                arr.data().opaqueBuffer(),
                arr.shapeInfoDataBuffer().opaqueBuffer(),
                arr.offset());
    }

    @Override
    public void setBArguments(Pointer arguments, int length) {
        BooleanPointer bArgs = arguments instanceof BooleanPointer ?(BooleanPointer) arguments : new BooleanPointer(arguments);
        nativeOps.setGraphContextBArguments(context, bArgs,length);
    }

    @Override
    public void setSArguments(String... arguments) {
        super.setSArguments(arguments);
        for (int i = 0; i < arguments.length; i++) {
            nativeOps.setGraphContextSArgument(context, arguments[i], i);
        }
    }

    @Override
    public void setRngStates(long rootState, long nodeState) {
        nativeOps.setRandomGeneratorStates(nativeOps.getGraphContextRandomGenerator(context), rootState, nodeState);
    }

    @Override
    public Pair<Long, Long> getRngStates() {
        OpaqueRandomGenerator g = nativeOps.getGraphContextRandomGenerator(context);
        return Pair.makePair(nativeOps.getRandomGeneratorRootState(g), nativeOps.getRandomGeneratorNodeState(g));
    }

    @Override
    public void setInputArray(int index, @NonNull INDArray array) {
        // Note: OpaqueNDArray.fromINDArray() already handles syncToSpecial()
        // for non-constant arrays, so no separate sync needed here.
        singleInputArrayRefs.put(index, array);
        OpaqueNDArray opaqueArray = OpaqueNDArray.fromINDArray(array);
        inputOpaqueArrayRefs.put(index, opaqueArray);
        nativeOps.setGraphContextInputArray(context, index, opaqueArray);
        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        // Store strong reference to INDArray to prevent GC while this OpContext is alive
        singleOutputArrayRefs.put(index, array);
        OpaqueNDArray opaqueArray = OpaqueNDArray.fromINDArray(array);
        outputOpaqueArrayRefs.put(index, opaqueArray);
        nativeOps.setGraphContextOutputArray(context, index, opaqueArray);
        super.setOutputArray(index, array);
    }

    @Override
    public OpaqueContext contextPointer() {
        return context;
    }


    public void setCudaStream(cudaStream_t stream, Pointer reductionPointer, Pointer allocationPointer) {
        nativeOps.setGraphContextCudaContext(context, stream, reductionPointer, allocationPointer);
    }

    @Override
    public void markInplace(boolean reallyInplace) {
        nativeOps.markGraphContextInplace(context, reallyInplace);
    }

    @Override
    public void allowHelpers(boolean reallyAllow) {
        nativeOps.ctxAllowHelpers(context, reallyAllow);
    }

    @Override
    public void shapeFunctionOverride(boolean reallyOverride) {
        nativeOps.ctxShapeFunctionOverride(context, reallyOverride);
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
        Nd4j.getExecutioner().commit();
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
        // Skip the Java commit() - the caller (InferenceSession) does a single
        // commit() after the full execution loop instead of per-op.
        // Use ctxPurgeNoSync to clear the native fast path vectors without
        // cudaDeviceSynchronize() - the kernel has already read the array pointers
        // at launch time, so clearing the vectors is safe without a full device sync.
        super.purgeForReuse();
        singleInputArrayRefs.clear();
        singleOutputArrayRefs.clear();
        inputOpaqueArrayRefs.clear();
        outputOpaqueArrayRefs.clear();
        if (context != null && !context.isNull()) {
            nativeOps.ctxPurgeNoSync(context);
        }
    }

    @Override
    public long getUniqueId() {
        return BASE_CUDA_OP_CONTEXT_OFFSET + id;

    }


    @Override
    public Deallocator deallocator() {
        return new CudaOpContextDeallocator(this);
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
