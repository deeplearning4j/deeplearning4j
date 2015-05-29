package org.nd4j.linalg.benchmark.scalar;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class ScalarBenchmarkPerformer extends BaseBenchmarkPerformer {

    public ScalarBenchmarkPerformer(int nTimes) {
        super(new ScalarOpRunner(),nTimes);
    }



}
