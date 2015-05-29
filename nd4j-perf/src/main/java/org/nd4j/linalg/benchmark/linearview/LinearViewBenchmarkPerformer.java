package org.nd4j.linalg.benchmark.linearview;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class LinearViewBenchmarkPerformer extends BaseBenchmarkPerformer {

    public LinearViewBenchmarkPerformer(int nTimes) {
        super(new LinearViewOpRunner(),nTimes);
    }



}
