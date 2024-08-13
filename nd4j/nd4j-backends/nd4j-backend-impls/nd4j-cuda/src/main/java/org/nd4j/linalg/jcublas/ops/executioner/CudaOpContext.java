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

import java.util.List;

/**
 * CUDA wrapper for op Context
 * @author raver119@gmail.com
 */
public class CudaOpContext extends BaseOpContext implements OpContext, Deallocatable {
    // we might want to have configurable
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private OpaqueContext context = nativeOps.createGraphContext(1);
    private final transient long id = Nd4j.getDeallocatorService().nextValue();
    public final static long BASE_CUDA_OP_CONTEXT_OFFSET = RandomUtils.nextLong();
    private long deallocationId;



    public CudaOpContext() {
        this.deallocationId = Nd4j.getDeallocatorService().pickObject(this);
    }

    @Override
    public void close() {
        //nativeOps.ctxPurge(context);

        Nd4j.getDeallocatorService().getReferenceMap().remove(this.deallocationId);

    }

    @Override
    public void setIArguments(long... arguments) {
        if (arguments.length > 0) {
            super.setIArguments(arguments);
            LongPointer iArgs = new LongPointer(arguments);
            nativeOps.setGraphContextIArguments(context, iArgs, arguments.length);
        }
    }

    @Override
    public void setBArguments(boolean... arguments) {
        if (arguments.length > 0) {
            super.setBArguments(arguments);
            BooleanPointer bArgs = new BooleanPointer(arguments);
            nativeOps.setGraphContextBArguments(context, bArgs, arguments.length);
        }
    }

    @Override
    public void setTArguments(double... arguments) {
        if (arguments.length > 0) {
            super.setTArguments(arguments);
            DoublePointer tArgs = new DoublePointer(arguments);
            nativeOps.setGraphContextTArguments(context, tArgs, arguments.length);
        };
    }

    @Override
    public void setDArguments(DataType... arguments) {
        if (arguments.length > 0) {
            super.setDArguments(arguments);
            val args = new int[arguments.length];
            for (int e = 0; e < arguments.length; e++)
                args[e] = arguments[e].toInt();

            IntPointer dArgs =  new IntPointer(args);
            nativeOps.setGraphContextDArguments(context,dArgs, arguments.length);
        }
    }
    @Override
    public void setInputArrays(@NonNull List<INDArray> arrays) {
        OpaqueDataBuffer[] buffers1 = new OpaqueDataBuffer[arrays.size()];
        OpaqueDataBuffer[] shapeInfoBufers2 = new OpaqueDataBuffer[arrays.size()];

        for(int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            buffers1[i] = array.isEmpty() ? null : array.data().opaqueBuffer();
            shapeInfoBufers2[i] = array.shapeInfoDataBuffer().opaqueBuffer();
            fastpath_in.put(i,array.isEmpty() ? null : array);
        }

        PointerPointer<OpaqueDataBuffer> buffers = new PointerPointer<>(buffers1);
        PointerPointer<OpaqueDataBuffer> shapeInfoBuffer = new PointerPointer<>(shapeInfoBufers2);

        nativeOps.setGraphContextInputBuffers(context,arrays.size(),buffers,shapeInfoBuffer,null);
    }

    @Override
    public void setOutputArrays(@NonNull List<INDArray> arrays) {
        OpaqueDataBuffer[] buffers1 = new OpaqueDataBuffer[arrays.size()];
        OpaqueDataBuffer[] shapeInfoBufers2 = new OpaqueDataBuffer[arrays.size()];

        for(int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            buffers1[i] = array.isEmpty() ? null : array.data().opaqueBuffer();
            shapeInfoBufers2[i] = array.shapeInfoDataBuffer().opaqueBuffer();
            fastpath_out.put(i,array);
        }

        PointerPointer<OpaqueDataBuffer> outputBuffers = new PointerPointer<>(buffers1);
        PointerPointer<OpaqueDataBuffer> shapeInfoOutputBuffer = new PointerPointer<>(shapeInfoBufers2);
        nativeOps.setGraphContextOutputBuffers(context,arrays.size(),outputBuffers,shapeInfoOutputBuffer,null);

    }

    @Override
    public void setInputArrays(INDArray... arrays) {
        OpaqueDataBuffer[] buffers1 = new OpaqueDataBuffer[arrays.length];
        OpaqueDataBuffer[] shapeInfoBufers2 = new OpaqueDataBuffer[arrays.length];
        if(!fastpath_in.isEmpty())
            fastpath_in.clear();
        for(int i = 0; i < arrays.length; i++) {
            INDArray array = arrays[i];
            buffers1[i] = array.isEmpty() ? null :  array.data().opaqueBuffer();
            shapeInfoBufers2[i] =  array.shapeInfoDataBuffer().opaqueBuffer();
            fastpath_in.put(i,array);
        }


        PointerPointer<OpaqueDataBuffer> buffers = new PointerPointer<>(buffers1);
        PointerPointer<OpaqueDataBuffer> shapeInfoBuffer = new PointerPointer<>(shapeInfoBufers2);
        nativeOps.setGraphContextInputBuffers(context,arrays.length,buffers,shapeInfoBuffer,null);
    }

    @Override
    public void setOutputArrays(INDArray... arrays) {
        OpaqueDataBuffer[] buffers1 = new OpaqueDataBuffer[arrays.length];
        OpaqueDataBuffer[] shapeInfoBufers2 = new OpaqueDataBuffer[arrays.length];

        for(int i = 0; i < arrays.length; i++) {
            INDArray array = arrays[i];
            buffers1[i] = array.isEmpty() ? null :  array.data().opaqueBuffer();
            shapeInfoBufers2[i] = array.shapeInfoDataBuffer().opaqueBuffer();
            fastpath_out.put(i,array);
        }


        PointerPointer<OpaqueDataBuffer> outputBuffers = new PointerPointer<>(buffers1);

        PointerPointer<OpaqueDataBuffer> shapeInfoOutputBuffer = new PointerPointer<>(shapeInfoBufers2);
        nativeOps.setGraphContextOutputBuffers(context,arrays.length,outputBuffers,shapeInfoOutputBuffer,null);
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
        Nd4j.getNativeOps().setIntermediateResult(context,
                index, arr.data().opaqueBuffer(),
                arr.shapeInfoDataBuffer().opaqueBuffer());
    }

    @Override
    public INDArray getIntermediateResult(int index) {
        LongPointer shapeInfo = nativeOps.intermediateResultShapeInfoAt(index,context);
        long rank = shapeInfo.get(0);
        shapeInfo.capacity(Shape.shapeInfoLength(rank));
        DataBuffer shapeInfoBuffer = Nd4j.createBuffer(shapeInfo, shapeInfo.capacity(),DataType.LONG);
        OpaqueDataBuffer buffer = nativeOps.intermediateResultDataAt(index,context);
        long numElements = nativeOps.dbBufferLength(buffer);
        /**
         * TODO: figure out why the buffer is the wrong length.
         * The shape buffer works but the normal databuffer doesn't.
         */
        Pointer pointer = buffer.primaryBuffer();
        pointer.capacity(numElements);
        DataBuffer firstBuffer = Nd4j.createBuffer(pointer,null,
                Shape.length(shapeInfoBuffer), Shape.dataType(shapeInfoBuffer));
        INDArray result = Nd4j.createArrayFromShapeBuffer(firstBuffer,shapeInfoBuffer);
        return result;
    }

    @Override
    public void addIntermediateResult(INDArray arr) {
        Nd4j.getNativeOps().pushIntermediateResult(context,
                arr.data().opaqueBuffer(),
                arr.shapeInfoDataBuffer().opaqueBuffer());
    }

    @Override
    public void setBArguments(Pointer arguments, int length) {
        BooleanPointer bArgs = arguments instanceof BooleanPointer ?(BooleanPointer) arguments : new BooleanPointer(arguments);
        nativeOps.setGraphContextBArguments(context, bArgs,length);
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
        nativeOps.setGraphContextInputBuffer(context, index, array.isEmpty() ? null : ((BaseCudaDataBuffer) array.data()).getOpaqueDataBuffer(), array.shapeInfoDataBuffer().opaqueBuffer(), array.shapeInfoDataBuffer().opaqueBuffer());

        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        nativeOps.setGraphContextOutputBuffer(context, index, array.isEmpty() ? null : ((BaseCudaDataBuffer) array.data()).getOpaqueDataBuffer(), array.shapeInfoDataBuffer().opaqueBuffer(), array.shapeInfoDataBuffer().opaqueBuffer());

        super.setOutputArray(index, array);
    }

    @Override
    public Pointer contextPointer() {
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
        super.purge();
        nativeOps.ctxPurge(context);
        Nd4j.getDeallocatorService().getReferenceMap().remove(this.deallocationId);


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
        return 0;
    }
}
