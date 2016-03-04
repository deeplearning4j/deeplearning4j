package org.nd4j.linalg.api.parallel.tasks;


public abstract class BaseTask<V> implements Task<V> {

    @Override
    public V invokeBlocking(){
        invokeAsync();
        return blockUntilComplete();
    }
}
