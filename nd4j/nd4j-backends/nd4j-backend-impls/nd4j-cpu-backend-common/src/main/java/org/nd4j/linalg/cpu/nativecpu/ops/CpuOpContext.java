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

package org.nd4j.linalg.cpu.nativecpu.ops;

import lombok.NonNull;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.bytedeco.javacpp.*;
import org.nd4j.common.primitives.Pair;
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
import org.nd4j.nativeblas.*;

import java.util.Arrays;
import java.util.List;

public class CpuOpContext extends BaseOpContext implements OpContext, Deallocatable {
    // we might want to have configurable
    private NativeOps nativeOps =Nd4j.getNativeOps();
    private OpaqueContext context = nativeOps.createGraphContext(1);
    private final transient long id = Nd4j.getDeallocatorService().nextValue();
    public final static long BASE_CPU_OP_CONTEXT_OFFSET = RandomUtils.nextLong();


    private transient  long deallocationId;


    public CpuOpContext() {
        this.deallocationId = Nd4j.getDeallocatorService().pickObject(this);

    }

    @Override
    public void close() {
        purge();
        Nd4j.getDeallocatorService().getReferenceMap().remove(this.deallocationId);

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
        nativeOps.setGraphContextDArguments(context, dArgs,length);
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
        LongPointer shapeInfo = nativeOps.intermediateResultShapeInfoAt(index,context);
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
    public long id() {
        return id;
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
    public void setRngStates(long rootState, long nodeState) {
        nativeOps.setRandomGeneratorStates(nativeOps.getGraphContextRandomGenerator(context), rootState, nodeState);
    }

    @Override
    public Pair<Long, Long> getRngStates() {
        OpaqueRandomGenerator g = nativeOps.getGraphContextRandomGenerator(context);
        return Pair.makePair(nativeOps.getRandomGeneratorRootState(g), nativeOps.getRandomGeneratorNodeState(g));
    }

    @Override
    public void setInputArrays(@NonNull List<INDArray> arrays) {
        OpaqueNDArray[] arrs = new OpaqueNDArray[arrays.size()];
        for (int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            arrs[i] = OpaqueNDArray.fromINDArray(array);
            fastpath_in.put(i, array.isEmpty() ? null : array);
        }
        if (!arrays.isEmpty()) {
            OpaqueNDArrayArr arr = new OpaqueNDArrayArr(arrs);
            nativeOps.setGraphContextInputArraysArr(context, arrays.size(), arr);
        }
    }

    @Override
    public void setOutputArrays(@NonNull List<INDArray> arrays) {
        OpaqueNDArray[] arrs = new OpaqueNDArray[arrays.size()];
        for (int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            arrs[i] = OpaqueNDArray.fromINDArray(array);
            fastpath_out.put(i, array.isEmpty() ? null : array);
        }

        if (!arrays.isEmpty()) {
            OpaqueNDArrayArr arr = new OpaqueNDArrayArr(arrs);
            nativeOps.setGraphContextOutputArraysArr(context, arrays.size(), arr);
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
        nativeOps.setGraphContextInputArray(context,index,OpaqueNDArray.fromINDArray(array));
        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        nativeOps.setGraphContextOutputArray(context,index,OpaqueNDArray.fromINDArray(array));
        super.setOutputArray(index, array);
    }

    @Override
    public OpaqueContext contextPointer() {
        return context;
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
    }

    @Override
    public long getUniqueId() {
        return BASE_CPU_OP_CONTEXT_OFFSET + id;
    }

    @Override
    public Deallocator deallocator() {
        return new CpuOpContextDeallocator(this);
    }

    @Override
    public int targetDevice() {
        return 0;
    }

    @Override
    public void transferTArgs() {
        if (fastpath_t.size() > 0) {
            val args = new double[fastpath_t.size()];
            for (int e = 0; e < fastpath_t.size(); e++)
                args[e] = fastpath_t.get(e);

            DoublePointer tArgs =  new DoublePointer(args);
            nativeOps.setGraphContextTArguments(context,  tArgs, fastpath_t.size());
        }
    }

    @Override
    public void transferIArgs() {
        if (fastpath_i.size() > 0) {
            val args = new long[fastpath_i.size()];
            for (int e = 0; e < fastpath_i.size(); e++)
                args[e] = fastpath_i.get(e);

            LongPointer iArgs =  new LongPointer(args);
            nativeOps.setGraphContextIArguments(context, iArgs, fastpath_i.size());
        }
    }

    @Override
    public void transferBArgs() {
        if (fastpath_b.size() > 0) {
            val args = new boolean[fastpath_b.size()];
            for (int e = 0; e < fastpath_b.size(); e++)
                args[e] = fastpath_b.get(e);

            BooleanPointer bArgs =  new BooleanPointer(args);
            nativeOps.setGraphContextBArguments(context, bArgs, fastpath_b.size());
        }
    }

    @Override
    public void transferDArgs() {
        if (fastpath_d.size() > 0) {
            val args = new int[fastpath_d.size()];
            for (int e = 0; e < fastpath_d.size(); e++)
                args[e] = fastpath_d.get(e).toInt();

            IntPointer dArgs =  new IntPointer(args);
            nativeOps.setGraphContextDArguments(context, dArgs, fastpath_d.size());
        }
    }
}
