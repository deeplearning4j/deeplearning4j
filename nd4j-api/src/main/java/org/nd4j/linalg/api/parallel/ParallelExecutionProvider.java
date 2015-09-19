package org.nd4j.linalg.api.parallel;


/**
 * Created by agibsonccc on 9/19/15.
 */
public interface ParallelExecutionProvider {
    String EXECUTOR_SERVICE_PROVIDER = "org.nd4j.linalg.api.parallel.executorserviceprovider";

    ParallelExecutioner getService();

}
