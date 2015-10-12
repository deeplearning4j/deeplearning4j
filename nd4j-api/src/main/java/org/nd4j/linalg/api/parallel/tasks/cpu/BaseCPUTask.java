package org.nd4j.linalg.api.parallel.tasks.cpu;

import org.nd4j.linalg.api.ops.Op;

public abstract class BaseCPUTask<V> extends AbstractCPUTask<V> {

    public BaseCPUTask(int threshold, int n, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(threshold, n, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public BaseCPUTask(Op op, int threshold) {
        super(op, threshold);
    }

    public BaseCPUTask(Op op, int threshold, int tadIdx, int tadDim) {
        super(op, threshold, tadIdx, tadDim);
    }
}
