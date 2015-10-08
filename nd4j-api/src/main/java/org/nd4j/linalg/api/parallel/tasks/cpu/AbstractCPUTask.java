package org.nd4j.linalg.api.parallel.tasks.cpu;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.parallel.tasks.BaseTask;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.util.List;
import java.util.concurrent.Future;

public abstract class AbstractCPUTask<V> extends BaseTask<V> {
    protected final int threshold;
    protected int n;
    protected int offsetX;
    protected int offsetZ;
    protected int incrX;
    protected int incrZ;

    protected final boolean doTensorFirst;
    protected int tadIdx;
    protected int tadDim;

    protected Future<V> future;
    protected List<Task<V>> subTasks;

    public AbstractCPUTask(int threshold, int n, int offsetX, int offsetZ, int incrX, int incrZ){
        this.threshold = threshold;
        this.n = n;
        this.offsetX = offsetX;
        this.offsetZ = offsetZ;
        this.incrX = incrX;
        this.incrZ = incrZ;
        doTensorFirst = false;
    }

    public AbstractCPUTask(Op op, int threshold){
        this.threshold = threshold;
        this.n = op.x().length();
        this.offsetX = op.x().offset();
        this.offsetZ = op.z().offset();
        this.incrX = op.x().elementWiseStride();
        this.incrZ = op.z().elementWiseStride();
        this.doTensorFirst = false;
    }

    public AbstractCPUTask(int threshold, int tadIdx, int tadDim){
        this.threshold = threshold;
        this.doTensorFirst = true;
        this.tadIdx = tadIdx;
        this.tadDim = tadDim;
    }
}
