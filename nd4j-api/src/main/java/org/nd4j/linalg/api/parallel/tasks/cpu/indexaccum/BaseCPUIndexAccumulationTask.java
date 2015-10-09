package org.nd4j.linalg.api.parallel.tasks.cpu.indexaccum;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;

public abstract class BaseCPUIndexAccumulationTask extends BaseCPUTask<Pair<Double,Integer>> {

    protected final IndexAccumulation op;
    protected final int elementOffset;
    protected final boolean outerTask;


    /**Constructor for operating on subset of NDArray
     */
    public BaseCPUIndexAccumulationTask(IndexAccumulation op, int threshold, int n, int offsetX, int offsetY,
                                        int incrX, int incrY, int elementOffset, boolean outerTask) {
        super(threshold,n,offsetX,offsetY,0,incrX,incrY,0);
        this.op = op;
        this.elementOffset = elementOffset;
        this.outerTask = outerTask;
    }

    /**Constructor for doing task on entire NDArray
     */
    public BaseCPUIndexAccumulationTask(IndexAccumulation op, int threshold, boolean outerTask) {
        super(op,threshold);
        this.op = op;
        this.offsetY = (op.y() != null ? op.y().offset() : 0);
        this.incrY = (op.y() != null ? op.y().elementWiseStride() : 0);
        this.elementOffset = 0;
        this.outerTask = outerTask;
    }

    /** Constructor for doing a 1d tensor first */
    public BaseCPUIndexAccumulationTask(IndexAccumulation op, int threshold, int tadIdx, int tadDim, boolean outerTask){
        super(op,threshold,tadIdx,tadDim);
        this.op = op;
        this.outerTask = outerTask;
        this.elementOffset = tadIdx * op.x().size(tadDim);
    }
}
