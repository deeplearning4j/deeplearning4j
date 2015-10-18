package org.nd4j.linalg.api.parallel.tasks.cpu.transform;

import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.parallel.tasks.cpu.BaseCPUAction;

public abstract class BaseCPUTransformOpAction extends BaseCPUAction {

    protected final TransformOp op;

    /**
     * Constructor for operating on subset of NDArray
     */
    public BaseCPUTransformOpAction(TransformOp op, int threshold, int n, int offsetX, int offsetY, int offsetZ,
                                    int incrX, int incrY, int incrZ) {
        super(threshold, n, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
        this.op = op;
    }

    /**
     * Constructor for doing task on entire NDArray
     */
    public BaseCPUTransformOpAction(TransformOp op, int threshold) {
        super(op, threshold);
        this.op = op;
        this.offsetY = (op.y() != null ? op.y().offset() : 0);
        this.incrY = (op.y() != null ? op.y().elementWiseStride() : 0);
    }

    /**
     * Constructor for doing a 1d tensor first
     */
    public BaseCPUTransformOpAction(TransformOp op, int threshold, int tadIdx, int tadDim) {
        super(op, threshold, tadIdx, tadDim);
        this.op = op;
    }
}
