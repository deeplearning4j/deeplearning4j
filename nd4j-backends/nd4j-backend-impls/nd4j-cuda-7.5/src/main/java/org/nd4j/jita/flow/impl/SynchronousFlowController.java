package org.nd4j.jita.flow.impl;


import org.apache.commons.lang3.concurrent.BasicThreadFactory;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.handler.MemoryHandler;
import org.nd4j.jita.memory.MemoryProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
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

    @Override
    public void init(Allocator allocator) {
        this.allocator = allocator;
    }

    @Override
    public void synchronizeToHost(AllocationPoint point) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        if (!point.isActualOnHostSide()) {

            if (!point.isConstant())
                waitTillFinished(point);

          //  log.info("Synchronization started... " + point.getShape());

            // if this piece of memory is device-dependant, we'll also issue copyback once
            if (point.getAllocationStatus() == AllocationStatus.DEVICE && !point.isActualOnHostSide()) {

                if (nativeOps.memcpyAsync(point.getHostPointer(), point.getDevicePointer(), AllocationUtils.getRequiredMemory(point.getShape()), CudaConstants.cudaMemcpyDeviceToHost, context.getSpecialStream()) == 0)
                    throw new IllegalStateException("MemcpyAsync failed");

                commitTransfer(context.getSpecialStream());
            }// else log.info("Not [DEVICE] memory, skipping...");


            // updating host read timer
            point.tickHostRead();
            //log.info("After sync... isActualOnHostSide: {}", point.isActualOnHostSide());
        }// else log.info("Point is actual on host side! " + point.getShape());
    }

    @Override
    public void waitTillFinished(AllocationPoint point) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        context.syncOldStream();
    }

    public void registerAction(CudaContext context, INDArray result, INDArray... operands) {
        if (result == null) return;
        AllocationPoint point = allocator.getAllocationPoint(result);
        point.tickDeviceWrite();
    }

    @Override
    public CudaContext prepareAction(INDArray result, INDArray... operands) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
        int cId = allocator.getDeviceId();
  //      StringBuilder builder = new StringBuilder();
  //      builder.append("threadId: ").append(Thread.currentThread().getId())
  //              .append("; cId: ").append(cId);


        if (result != null) {
            prepareDelayedMemory(result);
            AllocationPoint pointData = allocator.getAllocationPoint(result.data());
            AllocationPoint pointShape = allocator.getAllocationPoint(result.shapeInfoDataBuffer());

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0) {
          //      log.info("currentDevice: {}, pointDevice: {}, pointer: {}", cId, pointData.getDeviceId(), pointData.getPointers().getDevicePointer().address());
                allocator.getMemoryHandler().relocateObject(result.data());

                //allocator.getMemoryHandler().relocateObject(result.shapeInfoDataBuffer());
            }

            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0) {
                ((JCublasNDArray) result).setShapeInfoDataBuffer(Nd4j.getConstantHandler().relocateConstantSpace(result.shapeInfoDataBuffer()));
            }


/*
            pointData.addThreadToTrace(Thread.currentThread().getId());

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0)
                throw new RuntimeException("R data cId: [" +cId + "] != dId: ["+ pointData.getDeviceId() +"]; "  + pointData.getThreadsTrace().toString());

            AllocationPoint pointShape = allocator.getAllocationPoint(result.shapeInfoDataBuffer());
            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0)
                throw new RuntimeException("R shape cId: [" +cId + "] != dId: ["+ pointShape.getDeviceId() +"]");
*/
            allocator.getAllocationPoint(result).setCurrentContext(context);
        }

        for (INDArray operand: operands) {
            if (operand == null) continue;

            AllocationPoint pointData = allocator.getAllocationPoint(operand.data());
            AllocationPoint pointShape = allocator.getAllocationPoint(operand.shapeInfoDataBuffer());

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0) {
//                log.info("currentDevice: {}, pointDevice: {}, pointer: {}", cId, pointData.getDeviceId(), pointData.getPointers().getDevicePointer().address());

                allocator.getMemoryHandler().relocateObject(operand.data());

                //allocator.getMemoryHandler().relocateObject(operand.shapeInfoDataBuffer());
            }

            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0) {
                ((JCublasNDArray) operand).setShapeInfoDataBuffer(Nd4j.getConstantHandler().relocateConstantSpace(operand.shapeInfoDataBuffer()));
            }

/*

            pointData.addThreadToTrace(Thread.currentThread().getId());

            if (pointData.getDeviceId() != cId && pointData.getDeviceId() >= 0)
                throw new RuntimeException("O data cId: [" +cId + "] != dId: ["+ pointData.getDeviceId() +"]; Size: " + AllocationUtils.getRequiredMemory(pointData.getShape()) + "; " + pointData.getThreadsTrace().toString());

            AllocationPoint pointShape = allocator.getAllocationPoint(operand.shapeInfoDataBuffer());
            if (pointShape.getDeviceId() != cId && pointShape.getDeviceId() >= 0)
                throw new RuntimeException("O shape cId: [" +cId + "] != dId: ["+ pointShape.getDeviceId() +"]");
*/
            prepareDelayedMemory(operand);
            allocator.getAllocationPoint(operand).setCurrentContext(context);

      //      builder.append("; O_dId: ").append(pointData.getDeviceId());
      //      builder.append("; O_sdId: ").append(pointShape.getDeviceId());
      //      builder.append(", O_DPTR: ").append(pointData.getPointers().getDevicePointer().address());
      //      builder.append(", O_SPTR: ").append(pointShape.getPointers().getDevicePointer().address());
        }

//        log.info(builder.toString());

        return context;
    }

    @Override
    public void waitTillReleased(AllocationPoint point) {
        waitTillFinished(point);
    }

    @Override
    public void registerAction(CudaContext context, AllocationPoint result, AllocationPoint... operands) {
        context.syncOldStream();
    }

    @Override
    public CudaContext prepareAction(AllocationPoint result, AllocationPoint... operands) {
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();
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
