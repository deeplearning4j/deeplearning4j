package org.nd4j.linalg.benchmark.transform;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class TransformBenchmarkPerformer extends BaseBenchmarkPerformer {

    public TransformBenchmarkPerformer(int nTimes) {
        super(new TransformOpRunner(),nTimes);
    }



}
