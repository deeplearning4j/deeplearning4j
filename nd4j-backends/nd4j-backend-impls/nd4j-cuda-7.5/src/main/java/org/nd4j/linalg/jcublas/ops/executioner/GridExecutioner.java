package org.nd4j.linalg.jcublas.ops.executioner;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.meta.LinearMetaOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

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

    protected GridPointers pointerizeOp(Op op, int... dimensions) {
        GridPointers pointers = new GridPointers(op, dimensions);

        return pointers;
    }

    @Override
    public Op exec(Op op, int... dimension) {
        return super.exec(op, dimension);
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        return super.exec(op, dimension);
    }

    @Override
    public INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension) {
        return super.exec(accumulation, biasCorrected, dimension);
    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        return super.exec(op, dimension);
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
