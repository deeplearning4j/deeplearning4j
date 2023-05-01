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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.ExecutionMode;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.cpu.nativecpu.buffer.BaseCpuDataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.profiler.OpContextTracker;
import org.nd4j.nativeblas.*;

import java.util.ArrayList;
import java.util.List;

public class CpuOpContext extends BaseOpContext implements OpContext, Deallocatable {
    // we might want to have configurable
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private OpaqueContext context = nativeOps.createGraphContext(1);
    private final transient long id = Nd4j.getDeallocatorService().nextValue();
    public final static long BASE_CPU_OP_CONTEXT_OFFSET = RandomUtils.nextLong();


    private transient  long deallocationId;


    public CpuOpContext() {
        this.deallocationId = Nd4j.getDeallocatorService().pickObject(this);
        if(OpContextTracker.getInstance().isEnabled()) {
            OpContextTracker.getInstance().allocateOpContext(this);
        }

    }

    @Override
    public void close() {
        purge();
        if(OpContextTracker.getInstance().isEnabled()) {
            OpContextTracker.getInstance().deallocateContext(this);
            Nd4j.getDeallocatorService().updateDeallocationCount(this.deallocationId);
        }
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
        PointerPointer<OpaqueDataBuffer> buffers = new PointerPointer<>(arrays.size());
        PointerPointer<OpaqueDataBuffer> shapeInfoBuffer = new PointerPointer<>(arrays.size());
        List<DataBuffer> shapeInfoReferences = new ArrayList<>();
        for(int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            OpaqueDataBuffer opaqueDataBuffer = array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer();
            buffers.put(i,opaqueDataBuffer);
            DataBuffer dataBuffer = array.shapeInfoDataBuffer();
            shapeInfoReferences.add(dataBuffer);
            OpaqueDataBuffer shapeBuffer = ((BaseCpuDataBuffer) dataBuffer).getOpaqueDataBuffer();
            shapeInfoBuffer.put(i, shapeBuffer);
            fastpath_in.put(i,array.isEmpty() ? null : array);
            if(OpContextTracker.getInstance().isEnabled()) {
                OpContextTracker.getInstance().associateInput(array,this);
            }
        }

        nativeOps.setGraphContextInputBuffers(context,arrays.size(),buffers,shapeInfoBuffer,null);
    }

    @Override
    public void setOutputArrays(@NonNull List<INDArray> arrays) {
        OpaqueDataBuffer[] buffers1 = new OpaqueDataBuffer[arrays.size()];
        OpaqueDataBuffer[] shapeInfoBufers2 = new OpaqueDataBuffer[arrays.size()];

        for(int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);
            buffers1[i] = array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer();
            shapeInfoBufers2[i] = ((BaseCpuDataBuffer) array.shapeInfoDataBuffer()).getOpaqueDataBuffer();
            fastpath_out.put(i,array);
            if(OpContextTracker.getInstance().isEnabled()) {
                OpContextTracker.getInstance().associateOutput(array,this);
            }
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
            buffers1[i] = array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer();
            shapeInfoBufers2[i] = ((BaseCpuDataBuffer) array.shapeInfoDataBuffer()).getOpaqueDataBuffer();
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
            buffers1[i] = array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer();
            shapeInfoBufers2[i] =((BaseCpuDataBuffer) array.shapeInfoDataBuffer()).getOpaqueDataBuffer();
            fastpath_out.put(i,array);
        }


        PointerPointer<OpaqueDataBuffer> outputBuffers = new PointerPointer<>(buffers1);

        PointerPointer<OpaqueDataBuffer> shapeInfoOutputBuffer = new PointerPointer<>(shapeInfoBufers2);
        nativeOps.setGraphContextOutputBuffers(context,arrays.length,outputBuffers,shapeInfoOutputBuffer,null);
    }

    @Override
    public void setInputArray(int index, @NonNull INDArray array) {
        nativeOps.setGraphContextInputBuffer(context, index,
                array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer(),
                ((BaseCpuDataBuffer) array.shapeInfoDataBuffer()).getOpaqueDataBuffer(),
                null);
        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        nativeOps.setGraphContextOutputBuffer(context, index, array.isEmpty() ? null :
                        ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer(),
                ((BaseCpuDataBuffer) array.shapeInfoDataBuffer()).getOpaqueDataBuffer(), null);

        super.setOutputArray(index, array);
    }

    @Override
    public Pointer contextPointer() {
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
