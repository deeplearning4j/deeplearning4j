package org.nd4j.linalg.api.parallel.tasks;


import java.util.concurrent.Callable;

public interface Task<V> extends Callable<V> {

    /** Schedule for execution, and block until completion */
    V invokeBlocking();

    /** Schedule for asyncronous execution; returns immediately */
    void invokeAsync();

    /** Assuming invokeAsync() has been called, block until the execution completes */
    V blockUntilComplete();

    /** Compute the result immediately, in the current thread */
    V call();

}
