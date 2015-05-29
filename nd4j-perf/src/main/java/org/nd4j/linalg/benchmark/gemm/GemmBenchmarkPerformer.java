package org.nd4j.linalg.benchmark.gemm;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class GemmBenchmarkPerformer extends BaseBenchmarkPerformer {

    public GemmBenchmarkPerformer(int nTimes) {
        super(new GemmOpRunner(),nTimes);
    }



}
