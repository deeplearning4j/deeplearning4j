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
import org.bytedeco.javacpp.*;
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
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueRandomGenerator;

public class CpuOpContext extends BaseOpContext implements OpContext, Deallocatable {
    // we might want to have configurable
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private OpaqueContext context = nativeOps.createGraphContext(1);
    private final transient long id = Nd4j.getDeallocatorService().nextValue();

    private transient DoublePointer tArgs;
    private transient BooleanPointer bArgs;
    private transient IntPointer dArgs;
    private transient LongPointer iArgs;

    public CpuOpContext() {
        Nd4j.getDeallocatorService().pickObject(this);
    }

    @Override
    public void close() {
        // no-op
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
    public boolean hasCachedDArgs() {
        return dArgs != null && !dArgs.isNull();
    }

    @Override
    public boolean hasCachedTArgs() {
        return tArgs != null && !tArgs.isNull();
    }

    @Override
    public boolean hasCachedBArgs() {
        return bArgs != null && !bArgs.isNull();
    }

    @Override
    public boolean hasCachedIArgs() {
        return iArgs != null && !iArgs.isNull();
    }

    @Override
    public void setDArgAt(int index, DataType value) {
        if(dArgs == null || dArgs.isNull())
            throw new IllegalStateException("Please use setDArguments before trying to set at an index.");
        dArgs.put(index,value.toInt());
    }

    @Override
    public void setBArgAt(int index, boolean value) {
        if(bArgs == null || bArgs.isNull())
            throw new IllegalStateException("Please use setBArguments before trying to set at an index.");
        bArgs.put(index,value);
    }

    @Override
    public void setTArgAt(int index, double value) {
        if(tArgs == null || tArgs.isNull())
            throw new IllegalStateException("Please use setTArguments before trying to set at an index.");

        tArgs.put(index,value);
    }

    @Override
    public void setIArgAt(int index, long value) {
        if(iArgs == null || iArgs.isNull())
            throw new IllegalStateException("Please use setIArguments before trying to set at an index.");
        iArgs.put(index,value);
    }

    @Override
    public void setIArguments(long... arguments) {
        if (arguments.length > 0) {
            super.setIArguments(arguments);
            this.iArgs = new LongPointer(arguments);
            nativeOps.setGraphContextIArguments(context, this.iArgs, arguments.length);
        }
    }

    @Override
    public void setBArguments(boolean... arguments) {
        if (arguments.length > 0) {
            super.setBArguments(arguments);
            this.bArgs = new BooleanPointer(arguments);
            nativeOps.setGraphContextBArguments(context, this.bArgs, arguments.length);
        }
    }

    @Override
    public void setTArguments(double... arguments) {
        if (arguments.length > 0) {
            super.setTArguments(arguments);
            this.tArgs = new DoublePointer(arguments);
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

            this.dArgs =  new IntPointer(args);
            nativeOps.setGraphContextDArguments(context, this.dArgs, arguments.length);
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
        nativeOps.setGraphContextInputBuffer(context, index, array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer(), array.shapeInfoDataBuffer().addressPointer(), null);

        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        nativeOps.setGraphContextOutputBuffer(context, index, array.isEmpty() ? null : ((BaseCpuDataBuffer) array.data()).getOpaqueDataBuffer(), array.shapeInfoDataBuffer().addressPointer(), null);

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
    public String getUniqueId() {
        return new String("CTX_" + id);
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

            this.tArgs =  new DoublePointer(args);
            nativeOps.setGraphContextTArguments(context, this.tArgs, fastpath_t.size());
        }
    }

    @Override
    public void transferIArgs() {
        if (fastpath_i.size() > 0) {
            val args = new long[fastpath_i.size()];
            for (int e = 0; e < fastpath_i.size(); e++)
                args[e] = fastpath_i.get(e);

            this.iArgs =  new LongPointer(args);
            nativeOps.setGraphContextIArguments(context, this.iArgs, fastpath_i.size());
        }
    }

    @Override
    public void transferBArgs() {
        if (fastpath_b.size() > 0) {
            val args = new boolean[fastpath_b.size()];
            for (int e = 0; e < fastpath_b.size(); e++)
                args[e] = fastpath_b.get(e);

            this.bArgs =  new BooleanPointer(args);
            nativeOps.setGraphContextBArguments(context, this.bArgs, fastpath_b.size());
        }
    }

    @Override
    public void transferDArgs() {
        if (fastpath_d.size() > 0) {
            val args = new int[fastpath_d.size()];
            for (int e = 0; e < fastpath_d.size(); e++)
                args[e] = fastpath_d.get(e).toInt();

            this.dArgs =  new IntPointer(args);
            nativeOps.setGraphContextDArguments(context, this.dArgs, fastpath_d.size());
        }
    }
}
