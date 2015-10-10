package org.nd4j.linalg.api.parallel.tasks;


import java.util.concurrent.Callable;

/**A Task implements/defines the method of execution of an Op
 * An Op defines what is to be done
 * A Task defines how that op is executed (for example, it might be split up and parallelized)
 * A task also has methods for blockind and non-blocking (asynchronous) execution.
 * @param <V> The return type of the Task, Void if no return type
 * @author Alex Black
 */
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
