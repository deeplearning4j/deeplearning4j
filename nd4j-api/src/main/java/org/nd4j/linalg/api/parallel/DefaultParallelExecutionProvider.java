package org.nd4j.linalg.api.parallel;



/**
 * Created by agibsonccc on 9/19/15.
 */
public class DefaultParallelExecutionProvider implements ParallelExecutionProvider {
    @Override
    public ParallelExecutioner getService() {
        return new DefaultParallelExecutioner();
    }
}
