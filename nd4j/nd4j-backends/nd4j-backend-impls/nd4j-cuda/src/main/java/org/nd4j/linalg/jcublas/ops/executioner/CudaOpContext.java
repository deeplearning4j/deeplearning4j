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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.ExecutionMode;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.profiler.OpContextTracker;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueRandomGenerator;

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
    private transient DoublePointer tArgs;
    private transient BooleanPointer bArgs;
    private transient IntPointer dArgs;
    private transient LongPointer iArgs;


    public CudaOpContext() {
        this.deallocationId = Nd4j.getDeallocatorService().pickObject(this);
        if(OpContextTracker.getInstance().isEnabled()) {
            OpContextTracker.getInstance().allocateOpContext(this);
        }
    }

    @Override
    public void close() {
        nativeOps.ctxPurge(context);

        Nd4j.getDeallocatorService().getReferenceMap().remove(this.deallocationId);
        if(this.iArgs != null) {
            this.iArgs.deallocate();
        }

        if(this.tArgs != null) {
            this.tArgs.deallocate();
        }

        if(this.bArgs != null) {
            this.bArgs.deallocate();
        }

        if(this.dArgs != null) {
            this.dArgs.deallocate();
        }

        context.deallocate();
    }

    @Override
    public long id() {
        return id;
    }

    @Override
    public void setIArguments(Pointer arguments, int length) {
        this.iArgs = arguments instanceof LongPointer ?(LongPointer) arguments : new LongPointer(arguments);
        nativeOps.setGraphContextIArguments(context, this.iArgs,length);

    }

    @Override
    public void setTArguments(Pointer arguments, int length) {
        this.tArgs = arguments instanceof DoublePointer ?(DoublePointer) arguments : new DoublePointer(arguments);
        nativeOps.setGraphContextTArguments(context, this.tArgs,length);
    }

    @Override
    public void setDArguments(Pointer arguments, int length) {
        this.dArgs = arguments instanceof IntPointer ?(IntPointer) arguments : new IntPointer(arguments);
        nativeOps.setGraphContextDArguments(context, this.dArgs,length);
    }

    @Override
    public void setBArguments(Pointer arguments, int length) {
        this.bArgs = arguments instanceof BooleanPointer ?(BooleanPointer) arguments : new BooleanPointer(arguments);
        nativeOps.setGraphContextBArguments(context, this.bArgs,length);
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
