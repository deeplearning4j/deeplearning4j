package org.nd4j.linalg.api.parallel.tasks.cpu;

import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.parallel.tasks.Task;

import java.util.List;

public abstract class BaseCPUAction extends AbstractCPUTask<Void> {

    protected List<Task<Void>> subTasks;

    public BaseCPUAction(int threshold, int n, int offsetX, int offsetY, int offsetZ, int incrX, int incrY, int incrZ) {
        super(threshold, n, offsetX, offsetY, offsetZ, incrX, incrY, incrZ);
    }

    public BaseCPUAction(Op op, int threshold) {
        super(op, threshold);
    }

    public BaseCPUAction(Op op, int threshold, int tadIdx, int tadDim) {
        super(op, threshold, tadIdx, tadDim);
    }

    @Override
    public Void blockUntilComplete() {
        if (future == null ) {
            //invokeAsync hasn't been called?
            invokeAsync();
        }
        try {
            future.get();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if (subTasks != null) {
            for (Task<?> t : subTasks) {
                t.blockUntilComplete();
            }
        }
        return null;
    }
}
