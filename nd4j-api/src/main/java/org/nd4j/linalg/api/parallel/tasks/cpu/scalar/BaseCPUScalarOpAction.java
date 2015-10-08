package org.nd4j.linalg.api.parallel.tasks.cpu.scalar;

import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUTask;

public abstract class BaseCPUScalarOpAction extends BaseCPUAction {
    protected final ScalarOp op;

    /**Constructor for operating on subset of NDArray
     */
    public BaseCPUScalarOpAction(ScalarOp op, int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ) {
        super(threshold,n,offsetX,offsetZ,incrX,incrZ);
        this.op = op;
    }

    /**Constructor for doing task on entire NDArray
     */
    public BaseCPUScalarOpAction(ScalarOp op, int threshold) {
        super(op,threshold);
        this.op = op;
    }

    /** Constructor for doing a 1d tensor first */
    public BaseCPUScalarOpAction(ScalarOp op, int threshold, int tadIdx, int tadDim){
        super(threshold, tadIdx, tadDim);
        this.op = op;
    }
}
