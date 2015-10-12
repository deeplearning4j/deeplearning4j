package org.nd4j.linalg.api.parallel.tasks;

import java.util.concurrent.Future;

public interface TaskExecutor {

    /** Schedule a task for asynchronous execution */
    <V> Future<V> executeAsync(Task<V> task);

}
