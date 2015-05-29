package org.nd4j.linalg.benchmark.addirowvector;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class AddiRowVectorBenchmarkPerformer extends BaseBenchmarkPerformer {

    public AddiRowVectorBenchmarkPerformer(int nTimes) {
        super(new AddiRowVectorOpRunner(),nTimes);
    }



}
