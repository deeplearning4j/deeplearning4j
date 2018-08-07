/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.jita.flow.impl;


import lombok.Getter;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.concurrency.EventsProvider;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.JCublasNDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.profiler.OpProfiler;
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

        if (!point.isActualOnHostSide()) {
            CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

            if (!point.isConstant())
                waitTillFinished(point);

            //  log.info("Synchronization started... " + point.getShape());

            // if this piece of memory is device-dependant, we'll also issue copyback once
            if (point.getAllocationStatus() == AllocationStatus.DEVICE && !point.isActualOnHostSide()) {

                long perfD = PerformanceTracker.getInstance().helperStartTransaction();

                if (nativeOps.memcpyAsync(point.getHostPointer(), point.getDevicePointer(), AllocationUtils.getRequiredMemory(point.getShape()), CudaConstants.cudaMemcpyDeviceToHost, context.getSpecialStream()) == 0)
                    throw new IllegalStateException("MemcpyAsync failed: " + point.getShape());

                commitTransfer(context.getSpecialStream());

                PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), perfD, point.getNumberOfBytes(), MemcpyDirection.DEVICE_TO_HOST);
            } // else log.info("Not [DEVICE] memory, skipping...");


            // updating host read timer
            point.tickHostRead();
            //log.info("After sync... isActualOnHostSide: {}", point.isActualOnHostSide());
        } // else log.info("Point is actual on host side! " + point.getShape());
    }

    @Override
    public void synchronizeToDevice(AllocationPoint point) {
        if (point.isConstant())
            return;

        if (!point.isActualOnDeviceSide()) {


            if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
                CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

                long perfD = PerformanceTracker.getInstance().helperStartTransaction();

                if (nativeOps.memcpyAsync(point.getDevicePointer(), point.getHostPointer(),
                        AllocationUtils.getRequiredMemory(point.getShape()),
                        CudaConstants.cudaMemcpyHostToDevice, context.getSpecialStream()) == 0)
                    throw new IllegalStateException("MemcpyAsync failed: " + point.getShape());

                commitTransfer(context.getSpecialStream());
                point.tickDeviceRead();

                PerformanceTracker.getInstance().helperRegisterTransaction(point.getDeviceId(), perfD, point.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);
            }
        }
    }

    @Override
    public void waitTillFinished(AllocationPoint point) {
        /*CudaContext context = point.getCurrentContext(); //(CudaContext) allocator.getDeviceContext().getContext();
        if (context == null)
            context = (CudaContext) allocator.getDeviceContext().getContext();
        context.syncOldStream();
        */

        if (point.getLastWriteEvent() != null) {
            point.getLastWriteEvent().synchronize();
        }
    }


    @Override
    public CudaContext prepareActionAllWrite(INDArray... operands) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        int cId = allocator.getDeviceId();

        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            Nd4j.getCompressor().autoDecompress(operand);

            AllocationPoint pointData = allocator.getAllocationPoint(operand);
            AllocationPoint pointShape = allocator.getAllocationPoint(operand.shapeInfoDataBuffer());

            pointData.acquireLock();

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
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        int cId = allocator.getDeviceId();


        if (result != null) {
            Nd4j.getCompressor().autoDecompress(result);
            prepareDelayedMemory(result);
            AllocationPoint pointData = allocator.getAllocationPoint(result);
            AllocationPoint pointShape = allocator.getAllocationPoint(result.shapeInfoDataBuffer());

            pointData.acquireLock();


            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0 && (!CudaEnvironment.getInstance().getConfiguration().isCrossDeviceAccessAllowed() || !NativeOpsHolder.getInstance().getDeviceNativeOps().isP2PAvailable())) {
                DataBuffer buffer = result.data().originalDataBuffer() == null ? result.data()
                                : result.data().originalDataBuffer();
                allocator.getMemoryHandler().relocateObject(buffer);
            }

            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0) {
                ((JCublasNDArray) result).setShapeInfoDataBuffer(
                                Nd4j.getConstantHandler().relocateConstantSpace(result.shapeInfoDataBuffer()));
            }

            allocator.getAllocationPoint(result).setCurrentContext(context);
        }

        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            Nd4j.getCompressor().autoDecompress(operand);

            AllocationPoint pointData = allocator.getAllocationPoint(operand);
            AllocationPoint pointShape = allocator.getAllocationPoint(operand.shapeInfoDataBuffer());

            pointData.acquireLock();

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0 && (!CudaEnvironment.getInstance().getConfiguration().isCrossDeviceAccessAllowed() || !NativeOpsHolder.getInstance().getDeviceNativeOps().isP2PAvailable())) {
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
    public void waitTillReleased(AllocationPoint point) {
        waitTillFinished(point);

        if (point.getLastReadEvent() != null)
            point.getLastReadEvent().synchronize();
    }

    @Override
    public void registerAction(CudaContext context, AllocationPoint result, AllocationPoint... operands) {


        eventsProvider.storeEvent(result.getLastWriteEvent());
        result.setLastWriteEvent(eventsProvider.getEvent());
        result.getLastWriteEvent().register(context.getOldStream());
        result.releaseLock();


        for (AllocationPoint operand : operands) {
            eventsProvider.storeEvent(operand.getLastReadEvent());
            operand.setLastReadEvent(eventsProvider.getEvent());
            operand.getLastReadEvent().register(context.getOldStream());
            operand.releaseLock();
        }
        //   context.syncOldStream();
    }

    @Override
    public void registerActionAllWrite(CudaContext context, INDArray... operands) {
        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            AllocationPoint pointOperand = allocator.getAllocationPoint(operand);
            pointOperand.tickDeviceWrite();
            eventsProvider.storeEvent(pointOperand.getLastWriteEvent());
            pointOperand.setLastWriteEvent(eventsProvider.getEvent());
            pointOperand.getLastWriteEvent().register(context.getOldStream());
            pointOperand.releaseLock();
        }
    }

    public void registerAction(CudaContext context, INDArray result, INDArray... operands) {
        if (result == null)
            return;
        AllocationPoint point = allocator.getAllocationPoint(result);
        point.tickDeviceWrite();
        eventsProvider.storeEvent(point.getLastWriteEvent());
        point.setLastWriteEvent(eventsProvider.getEvent());
        point.getLastWriteEvent().register(context.getOldStream());
        point.releaseLock();

        for (INDArray operand : operands) {
            if (operand == null)
                continue;

            AllocationPoint pointOperand = allocator.getAllocationPoint(operand);
            pointOperand.releaseLock();
            eventsProvider.storeEvent(pointOperand.getLastReadEvent());
            pointOperand.setLastReadEvent(eventsProvider.getEvent());
            pointOperand.getLastReadEvent().register(context.getOldStream());
        }
    }

    @Override
    public CudaContext prepareAction(AllocationPoint result, AllocationPoint... operands) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        if (result != null) {
            result.acquireLock();
            result.setCurrentContext(context);
        }

        for (AllocationPoint operand : operands) {
            if (operand == null)
                continue;
            operand.acquireLock();
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
            AllocationPoint pointData = allocator.getAllocationPoint(array.shapeInfoDataBuffer());
            AllocationPoint pointShape = allocator.getAllocationPoint(array.shapeInfoDataBuffer());

            if (pointData.getAllocationStatus() != AllocationStatus.DEVICE)
                prepareDelayedMemory(array.data());

            if (pointShape.getAllocationStatus() == AllocationStatus.HOST) {
                DataBuffer oShape = array.shapeInfoDataBuffer();
                DataBuffer nShape = Nd4j.getConstantHandler().relocateConstantSpace(oShape);
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
