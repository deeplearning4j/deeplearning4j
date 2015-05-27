package org.nd4j.linalg.benchmark.linearview.getput;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class GetPutScalarLinearViewBenchmarkPerformer extends BaseBenchmarkPerformer {
    public GetPutScalarLinearViewBenchmarkPerformer(int nTimes) {
        super(new GetPutScalarLinearViewOpRunner(),nTimes);
    }
}
