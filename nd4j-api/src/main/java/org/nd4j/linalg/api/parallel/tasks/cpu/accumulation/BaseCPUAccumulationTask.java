package org.nd4j.linalg.api.parallel.tasks.cpu.accumulation;

import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;

public abstract class BaseCPUAccumulationTask extends BaseCPUTask<Double> {

    protected final Accumulation op;
    protected int offsetY;
    protected int incrY;
    protected final boolean outerTask;

    /**Constructor for operating on subset of NDArray
     */
    public BaseCPUAccumulationTask(Accumulation op, int threshold, int n, int offsetX, int offsetY, int incrX, int incrY,
                                   boolean outerTask) {
        super(threshold,n,offsetX,0,incrX,0);
        this.op = op;
        this.offsetY = offsetY;
        this.incrY = incrY;
        this.outerTask = outerTask;
    }

    /**Constructor for doing task on entire NDArray
     */
    public BaseCPUAccumulationTask(Accumulation op, int threshold, boolean outerTask) {
        super(op,threshold);
        this.op = op;
        this.offsetY = (op.y() != null ? op.y().offset() : 0);
        this.incrY = (op.y() != null ? op.y().elementWiseStride() : 0);
        this.outerTask = outerTask;
    }

    /** Constructor for doing a 1d tensor first */
    public BaseCPUAccumulationTask(Accumulation op, int threshold, int tadIdx, int tadDim, boolean outerTask){
        super(threshold,tadIdx,tadDim);
        this.op = op;
        this.outerTask = outerTask;
    }
}
