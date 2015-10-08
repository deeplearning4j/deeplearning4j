package org.nd4j.linalg.api.parallel.tasks.cpu;

import org.nd4j.linalg.api.ops.Op;

public abstract class BaseCPUTask<V> extends AbstractCPUTask<V> {

    public BaseCPUTask( int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ ){
        super(threshold, n, offsetX, offsetZ, incrX, incrZ );
    }

    public BaseCPUTask( Op op, int threshold ){
        super(op,threshold);
    }

    public BaseCPUTask(int threshold, int tadIdx, int tadDim){
        super(threshold,tadIdx,tadDim);
    }
}
