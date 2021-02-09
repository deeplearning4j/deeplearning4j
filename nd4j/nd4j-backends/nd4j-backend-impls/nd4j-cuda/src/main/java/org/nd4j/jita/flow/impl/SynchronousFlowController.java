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

package org.nd4j.jita.flow.impl;


import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.concurrency.EventsProvider;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.JCublasNDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public class SynchronousFlowController implements FlowController {
    private static Logger log = LoggerFactory.getLogger(SynchronousFlowController.class);
    private volatile Allocator allocator;
    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    protected Configuration configuration = CudaEnvironment.getInstance().getConfiguration();
    @Getter
    protected EventsProvider eventsProvider = new EventsProvider();

    @Override
    public void init(Allocator allocator) {
        this.allocator = allocator;
    }

    /**
     * This method makes sure HOST memory contains latest data from GPU
     *
     * @param point
     */
    @Override
    public void synchronizeToHost(AllocationPoint point) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSyncToPrimary(point.getPtrDataBuffer());
    }

    @Override
    public void synchronizeToDevice(@NonNull AllocationPoint point) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSyncToSpecial(point.getPtrDataBuffer());
    }

    @Override
    public void waitTillFinished(AllocationPoint point) {
        // this should be always null, since synchronization happens in C++ now
        if (point.getLastWriteEvent() != null) {
            point.getLastWriteEvent().synchronize();
        }
    }


    @Override
    public CudaContext prepareActionAllWrite(INDArray... operands) {
        val context = allocator.getDeviceContext();
        val cId = allocator.getDeviceId();

        for (INDArray operand : operands) {
            if (operand == null || operand.isEmpty())
                continue;

            Nd4j.getCompressor().autoDecompress(operand);

            val pointData = allocator.getAllocationPoint(operand);
            val pointShape = allocator.getAllocationPoint(operand.shapeInfoDataBuffer());


            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0) {
                DataBuffer buffer = operand.data().originalDataBuffer() == null ? operand.data()
                                : operand.data().originalDataBuffer();
                allocator.getMemoryHandler().relocateObject(buffer);
            }

            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0) {
                ((JCublasNDArray) operand).setShapeInfoDataBuffer(
                                Nd4j.getConstantHandler().relocateConstantSpace(operand.shapeInfoDataBuffer()));
            }

            prepareDelayedMemory(operand);
            allocator.getAllocationPoint(operand).setCurrentContext(context);
        }
        return context;
    }

    @Override
    public CudaContext prepareAction(INDArray result, INDArray... operands) {
        val context = allocator.getDeviceContext();
        val cId = allocator.getDeviceId();


        if (result != null && !result.isEmpty()) {
            Nd4j.getCompressor().autoDecompress(result);
            prepareDelayedMemory(result);
            val pointData = allocator.getAllocationPoint(result);
            val pointShape = allocator.getAllocationPoint(result.shapeInfoDataBuffer());

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0 && (!CudaEnvironment.getInstance().getConfiguration().isCrossDeviceAccessAllowed() || !NativeOpsHolder.getInstance().getDeviceNativeOps().isP2PAvailable())) {
                DataBuffer buffer = result.data().originalDataBuffer() == null ? result.data()
                                : result.data().originalDataBuffer();
                allocator.getMemoryHandler().relocateObject(buffer);
            }

            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0) {
                ((JCublasNDArray) result).setShapeInfoDataBuffer(Nd4j.getExecutioner().createShapeInfo(result.shape(), result.stride(), result.elementWiseStride(), result.ordering(), result.dataType(), result.isEmpty()));
            }

            allocator.getAllocationPoint(result).setCurrentContext(context);
        }

        if (operands == null)
            return context;

        for (INDArray operand : operands) {
            // empty or String arrays can be skipped
            if (operand == null || operand.isEmpty() || operand.isS())
                continue;

            Nd4j.getCompressor().autoDecompress(operand);

            val pointData = allocator.getAllocationPoint(operand);
            val pointShape = allocator.getAllocationPoint(operand.shapeInfoDataBuffer());
            Nd4j.getAffinityManager().ensureLocation(operand, AffinityManager.Location.DEVICE);

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0 && (!CudaEnvironment.getInstance().getConfiguration().isCrossDeviceAccessAllowed() || !NativeOpsHolder.getInstance().getDeviceNativeOps().isP2PAvailable())) {
                DataBuffer buffer = operand.data().originalDataBuffer() == null ? operand.data()
                                : operand.data().originalDataBuffer();
                allocator.getMemoryHandler().relocateObject(buffer);
            }

            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0) {
                ((JCublasNDArray) operand).setShapeInfoDataBuffer(Nd4j.getExecutioner().createShapeInfo(operand.shape(), operand.stride(), operand.elementWiseStride(), operand.ordering(), operand.dataType(), operand.isEmpty()));
            }

            prepareDelayedMemory(operand);
            allocator.getAllocationPoint(operand).setCurrentContext(context);
        }
        return context;
    }

    @Override
    public void waitTillReleased(AllocationPoint point) {
        waitTillFinished(point);

        if (point.getLastReadEvent() != null)
            point.getLastReadEvent().synchronize();
    }

    @Override
    public void registerAction(CudaContext context, AllocationPoint result, AllocationPoint... operands) {
        // this method is irrelevant now, everything happens in C++ now
        /*
        eventsProvider.storeEvent(result.getLastWriteEvent());
        result.setLastWriteEvent(eventsProvider.getEvent());
        result.getLastWriteEvent().register(context.getOldStream());


        for (AllocationPoint operand : operands) {
            eventsProvider.storeEvent(operand.getLastReadEvent());
            operand.setLastReadEvent(eventsProvider.getEvent());
            operand.getLastReadEvent().register(context.getOldStream());
        }
        //   context.syncOldStream();
        */
    }

    @Override
    public void registerActionAllWrite(CudaContext context, INDArray... operands) {
        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            val pointOperand = allocator.getAllocationPoint(operand);
            pointOperand.tickDeviceWrite();
        }
    }

    public void registerAction(CudaContext context, INDArray result, INDArray... operands) {
        if (result == null || result.isEmpty())
            return;

        val point = allocator.getAllocationPoint(result);
        point.tickDeviceWrite();

        for (INDArray operand : operands) {
            if (operand == null || operand.isEmpty())
                continue;

            val pointOperand = allocator.getAllocationPoint(operand);
            pointOperand.tickDeviceRead();
        }
    }

    @Override
    public CudaContext prepareAction(AllocationPoint result, AllocationPoint... operands) {
        val context = allocator.getDeviceContext();

        if (result != null) {
            result.setCurrentContext(context);
        }

        for (AllocationPoint operand : operands) {
            if (operand == null)
                continue;

            operand.setCurrentContext(context);
        }

        return context;
    }

    @Override
    public void commitTransfer(cudaStream_t streamUsed) {
        streamUsed.synchronize();
    }

    protected void prepareDelayedMemory(INDArray array) {
        if (configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED) {
            val pointData = allocator.getAllocationPoint(array.shapeInfoDataBuffer());
            val pointShape = allocator.getAllocationPoint(array.shapeInfoDataBuffer());

            if (pointData.getAllocationStatus() != AllocationStatus.DEVICE)
                prepareDelayedMemory(array.data());

            if (pointShape.getAllocationStatus() == AllocationStatus.HOST) {
                val oShape = array.shapeInfoDataBuffer();
                val nShape = Nd4j.getConstantHandler().relocateConstantSpace(oShape);

                if (nShape == oShape)
                    Nd4j.getConstantHandler().moveToConstantSpace(nShape);
                ((JCublasNDArray) array).setShapeInfoDataBuffer(nShape);
            }
        }
    }

    protected void prepareDelayedMemory(DataBuffer buffer) {

        allocator.getMemoryHandler().promoteObject(buffer);
    }
}
