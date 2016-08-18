package org.nd4j.linalg.jcublas.ops.executioner;

import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.meta.LinearMetaOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * mGRID implementation for OpExecutioner interface
 *
 * PLEASE NOTE: WORK IN PROGRESS, DO NOT EVER USE THIS EXECUTIONER IN PRODUCTION
 * @author raver119@gmail.com
 */
public class GridExecutioner extends JCudaExecutioner {

    // general queues
    private List<Deque<OpDescriptor>> deviceQueues = new ArrayList<>();

    // last op
    private ThreadLocal<Op> lastOp = new ThreadLocal<>();

    @Override
    public Op exec(Op op) {
        /*
            We pass this op to GridProcessor through check for possible MetaOp concatenation
         */
        if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            exec(acc, new int[]{Integer.MAX_VALUE});
        }

        return validateAsMetaOp(op, null);
    }

    /**
     * This method adds op into GridOp queue
     *
     * @param op
     * @param dimension
     * @return
     */
    protected Op pushToGrid(Op op, int... dimension) {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        deviceQueues.get(deviceId).add(new OpDescriptor(op, dimension));

        return op;
    }

    protected Op validateAsMetaOp(Op op, int... dimension) {
        /*
            We have multiple options here:
                1) Op has no relation to lastOp
                2) Op has SOME relation to lastOp

                So we either should append this op to future GridOp, or form MetaOp
         */


        Op last = lastOp.get();
        if (!isMatchingMetaOp(op)) {
            /*
                If we can't form MetaOp with new Op here, we should move lastOp to GridOp queue, and update lastOp with current Op
             */
            lastOp.set(op);

            if (last != null)
                pushToGrid(last, dimension);
        } else {
            /*
                If we can form new MetaOp, we should do that right now.
             */
            lastOp.remove();

            MetaOp metaOp = new LinearMetaOp(last, op);
            pushToGrid(metaOp, null);
        }

        return op;
    }

    protected boolean isMatchingMetaOp(Op op, int... dimension) {
        Op last = lastOp.get();
        if (last == null) {
            return false;
        } else {
            // check for linear access ops
            if (last instanceof ScalarOp || last instanceof TransformOp) {
                if (op instanceof ScalarOp || op instanceof  TransformOp) {
                    return isMatchingZX(last, op);
                }
            }
        }

        return false;
    }

    protected boolean isMatchingZX(Op opA, Op opB) {
        if (opA.y() == null && opB.y() == null)
            if (opA.z() == opB.x())
                return true;

        return false;
    }

    protected boolean isMatchingZXY(Op opA, Op opB) {
        if (opA.y() == null)
            if (opA.z() == opB.x() || opA.z() == opB.y())
                return true;

        return false;
    }

    /**
     * This method returns Op as set of required pointers for it
     * @param op
     * @param dimensions
     * @return
     */
    protected GridPointers pointerizeOp(Op op, int... dimensions) {
        GridPointers pointers = new GridPointers(op, dimensions);

        AtomicAllocator allocator = AtomicAllocator.getInstance();

        // FIXME: do not leave it as is
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        pointers.setX(allocator.getPointer(op.x(), context));
        pointers.setXShapeInfo(allocator.getPointer(op.x().shapeInfoDataBuffer(), context));
        pointers.setZ(allocator.getPointer(op.z(), context));
        pointers.setZShapeInfo(allocator.getPointer(op.z().shapeInfoDataBuffer(), context));

        if (op.y() != null) {
            pointers.setY(allocator.getPointer(op.y(), context));
            pointers.setYShapeInfo(allocator.getPointer(op.y().shapeInfoDataBuffer(), context));
        }

        if (dimensions != null && dimensions.length > 0) {
            DataBuffer dimensionBuffer = Nd4j.getConstantHandler().getConstantBuffer(dimensions);
            pointers.setDimensions(allocator.getPointer(dimensionBuffer, context));
            pointers.setDimensionsLength(dimensions.length);
        }


        // we build TADs
        if (dimensions != null && dimensions.length > 0) {
            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimensions);

            Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);
            Pointer devTadOffsets = tadBuffers.getSecond() == null ? null :AtomicAllocator.getInstance().getPointer(tadBuffers.getSecond(), context);

            pointers.setTadShape(devTadShapeInfo);
            pointers.setTadOffsets(devTadOffsets);
        }


        return pointers;
    }

    /**
     * This method returns Op queue lengths for current device
     *
     * @return
     */
    protected int getQueueLength() {
        return getQueueLength(Nd4j.getAffinityManager().getDeviceForCurrentThread());
    }

    /**
     * This method returns Op queue length for specified device
     *
     * @param deviceId
     * @return
     */
    protected int getQueueLength(int deviceId) {
        if (deviceId >= deviceQueues.size() || deviceQueues.get(deviceId) == null)
            deviceQueues.add(deviceId, new ConcurrentLinkedDeque<OpDescriptor>());

        return deviceQueues.get(deviceId).size();
    }

    /**
     * This method bundless all ops available in queue into single GridOp
     * @return
     */
    protected GridOp buildGrid() {
        return null;
    }

    protected void buildZ(IndexAccumulation op, int... dimension) {
        Arrays.sort(dimension);

        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }

        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};


        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[]{1, retShape[0]};
            else
                retShape = new int[]{retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[]{1, 1};
        }

/*
        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.noOp();
*/

        INDArray ret = null;
        if (op.zeroDouble() > -0.01f && op.zeroDouble() < 0.01f) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }
        op.setZ(ret);
    }

    protected void buildZ(Accumulation op, int... dimension) {
        Arrays.sort(dimension);

        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }

        //do op along all dimensions
        if (dimension.length == op.x().rank())
            dimension = new int[]{Integer.MAX_VALUE};


        int[] retShape = Shape.wholeArrayDimension(dimension) ? new int[] {1,1} : ArrayUtil.removeIndex(op.x().shape(), dimension);
        //ensure vector is proper shape
        if (retShape.length == 1) {
            if (dimension[0] == 0)
                retShape = new int[]{1, retShape[0]};
            else
                retShape = new int[]{retShape[0], 1};
        } else if (retShape.length == 0) {
            retShape = new int[]{1, 1};
        }

/*
        if(op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape))
            return op.noOp();
*/

        INDArray ret = null;
        if (op.zeroDouble() > -0.01f && op.zeroDouble() < 0.01f) {
            ret= Nd4j.zeros(retShape);
        } else {
            ret = Nd4j.valueArrayOf(retShape, op.zeroDouble());
        }
        op.setZ(ret);
    }

    @Override
    public Op exec(Op op, int... dimension) {
        return super.exec(op, dimension);
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        buildZ(op, dimension);

        return op.z();
    }

    @Override
    public INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension) {
        return super.exec(accumulation, biasCorrected, dimension);
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        buildZ(op, dimension);

        return op.z();
    }

    @Override
    public INDArray exec(BroadcastOp broadcast, int... dimension) {
        return super.exec(broadcast, dimension);
    }

    @Override
    public void exec(MetaOp op) {
        // TODO: to be implemented
    }

    @Override
    public void exec(GridOp op) {
        // TODO: to be implemented
    }
}
