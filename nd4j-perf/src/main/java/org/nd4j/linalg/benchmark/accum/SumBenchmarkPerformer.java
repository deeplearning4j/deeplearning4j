package org.nd4j.linalg.benchmark.accum;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class SumBenchmarkPerformer extends BaseBenchmarkPerformer {

    public SumBenchmarkPerformer(int nTimes) {
        super(new SumOpRunner(),nTimes);
    }



}
