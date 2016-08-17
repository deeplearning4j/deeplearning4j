package org.nd4j.linalg.jcublas.ops.executioner;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
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
public class GridExecutioner extends DefaultOpExecutioner {

    // general queues
    private List<Deque<Op>> deviceQueues = new ArrayList<>();

    // last op
    private ThreadLocal<Op> lastOp = new ThreadLocal<>();

    @Override
    public Op exec(Op op) {
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        if (!isMatchingMetaOp(op))
            lastOp.set(op);
        else {
            Op last = lastOp.get();
            lastOp.remove();

            MetaOp metaOp = new LinearMetaOp(last, op);
            exec(metaOp);
        }

        return op;
    }

    protected boolean isMatchingMetaOp(Op op) {
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
