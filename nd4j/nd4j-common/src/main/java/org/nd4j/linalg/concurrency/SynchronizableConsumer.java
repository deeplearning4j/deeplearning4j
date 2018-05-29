package org.nd4j.linalg.concurrency;

public interface SynchronizableConsumer {
    void setVariableBarrier(VariableBarrier barrier);

    VariableBarrier getVariableBarrier();
}
