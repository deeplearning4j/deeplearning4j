package org.nd4j.linalg.benchmark.api;

import org.nd4j.linalg.factory.Nd4jBackend;

/**
 *
 * Run performance benchmark
 * with a given backend.
 *
 * @author Adam Gibson
 */
public interface BenchMarkPerformer {


    /**
     * Number of times to run
     * @return the umber of times to run
     */
    int nTimes();

    /**
     * The average time for the benchmark (in milliseconds)
     * @return the average time for the benchmark
     */
    long averageTime();

    /**
     * Run a given backend
     * @param backend the backend to run
     */
    long run(Nd4jBackend backend);



}
