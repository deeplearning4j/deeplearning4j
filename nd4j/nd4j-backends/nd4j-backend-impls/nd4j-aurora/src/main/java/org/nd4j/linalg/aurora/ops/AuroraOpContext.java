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


package org.nd4j.linalg.aurora.ops;

import lombok.NonNull;
import lombok.val;

import java.util.Arrays;

import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.ExecutionMode;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.aurora.buffer.BaseAuroraDataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.aurora.Nd4jAuroraOps;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueRandomGenerator;

/**
 * CPU backend Context wrapper
 *
 * @author Adam Gibson
 */
public class AuroraOpContext extends BaseOpContext implements OpContext, Deallocatable {
    // we might want to have configurable
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private OpaqueContext context = nativeOps.createGraphContext(1);
    private final transient long id = Nd4j.getDeallocatorService().nextValue();

    public AuroraOpContext() {
        Nd4j.getDeallocatorService().pickObject(this);
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public void setIArguments(long... arguments) {
        if (arguments.length > 0) {
            super.setIArguments(arguments);
            nativeOps.setGraphContextIArguments(context, new LongPointer(arguments), arguments.length);
        }
    }

    @Override
    public void setBArguments(boolean... arguments) {
        if (arguments.length > 0) {
            super.setBArguments(arguments);
            nativeOps.setGraphContextBArguments(context, new BooleanPointer(arguments), arguments.length);
        }
    }

    @Override
    public void setTArguments(double... arguments) {
        if (arguments.length > 0) {
            super.setTArguments(arguments);
            nativeOps.setGraphContextTArguments(context, new DoublePointer(arguments), arguments.length);
        };
    }

    @Override
    public void setDArguments(DataType... arguments) {
        if (arguments.length > 0) {
            super.setDArguments(arguments);
            val args = new int[arguments.length];
            for (int e = 0; e < arguments.length; e++)
                args[e] = arguments[e].toInt();

            nativeOps.setGraphContextDArguments(context, new IntPointer(args), arguments.length);
        };
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
        //nativeOps.setGraphContextInputArray(context, index, array.isEmpty() ? null : array.data().addressPointer(), array.shapeInfoDataBuffer().addressPointer(), null, null);
        //pointer and addressPointer is the same 
        //but addressPointer calls veo
        //so we will use the pointer instead, but we need set limit<=0 to make sure that Nd4jAuroraOps.call will treat it as input pointer
        Pointer p = new LongPointer(array.shapeInfoDataBuffer().pointer());
        p.limit(0);
        nativeOps.setGraphContextInputBuffer(context, index, array.isEmpty() ? null : ((BaseAuroraDataBuffer) array.data()).getOpaqueDataBuffer(), p, null);

        super.setInputArray(index, array);
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        Pointer p = new LongPointer(array.shapeInfoDataBuffer().pointer());
        p.limit(0);
        nativeOps.setGraphContextOutputBuffer(context, index, array.isEmpty() ? null : ((BaseAuroraDataBuffer) array.data()).getOpaqueDataBuffer(), p, null);

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
        return new AuroraOpContextDeallocator(this);
    }

    @Override
    public int targetDevice() {
        return 0;
    }

    @Override
    public void setArgs(INDArray[] inputArrs, long[] iArgs, DataType[] dArgs, double[] tArgs, boolean[] bArgs) {
        Nd4jAuroraOps ops = (Nd4jAuroraOps) nativeOps;
        int size = inputArrs == null ? 0 : inputArrs.length;
        LongPointer inputPairArr = null;
        LongPointer iArgsPtr = null;
        IntPointer dArgsPtr = null;
        DoublePointer tArgsPtr = null;
        BooleanPointer bArgsPtr = null;
        if (size > 0) {
            inputPairArr = new LongPointer(2 * size);

            int i = 0;
            for (int j=0; j< size; j++) {
                INDArray arr = inputArrs[j];
                long buffer = arr.isEmpty() ? null
                        : ((BaseAuroraDataBuffer) arr.data()).getOpaqueDataBuffer().address();
                long shapeInfo = arr.shapeInfoDataBuffer().pointer().address();
                inputPairArr.put(i, buffer);
                inputPairArr.put(i + 1, shapeInfo);
                i += 2;
                //set internal
                super.setInputArray(j, arr);
            }

        }

        if (iArgs != null && iArgs.length > 0) {
            iArgsPtr = new LongPointer(iArgs.length);
            for (int i = 0; i < iArgs.length; i++) {
                iArgsPtr.put(i, iArgs[i]);
            }
            //this will call internal one, so it will not cause veo calls
            super.setIArguments(iArgs);
        }

        if (dArgs != null && dArgs.length > 0) {
            dArgsPtr = new IntPointer(dArgs.length);
            for (int i = 0; i < dArgs.length; i++) {
                dArgsPtr.put(i, dArgs[i].toInt());
            }
            super.setDArguments(dArgs);
        }

        if (tArgs != null && tArgs.length > 0) {
            tArgsPtr = new DoublePointer(tArgs.length);
            for (int i = 0; i < tArgs.length; i++) {
                tArgsPtr.put(i, tArgs[i]);
            }
            super.setTArguments(tArgs);
        }

        if (bArgs != null && bArgs.length > 0) {
            bArgsPtr = new BooleanPointer(bArgs.length);
            for (int i = 0; i < bArgs.length; i++) {
                bArgsPtr.put(i, bArgs[i]);
            }
            super.setBArguments(bArgs);
        }
        ops.setGraphContextArgs(context, inputPairArr, iArgsPtr, dArgsPtr, tArgsPtr, bArgsPtr);

    }
}
