package org.nd4j.linalg.jcublas.parallel;

import org.nd4j.linalg.api.parallel.ParallelExecutionProvider;
import org.nd4j.linalg.api.parallel.ParallelExecutioner;

/**
 * Created by agibsonccc on 10/3/15.
 */
public class GpuParallelExecutionProvider implements ParallelExecutionProvider {
    @Override
    public ParallelExecutioner getService() {
        return new GpuParallelExecutioner();
    }
}
