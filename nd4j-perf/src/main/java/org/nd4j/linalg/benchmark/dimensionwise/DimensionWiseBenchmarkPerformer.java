package org.nd4j.linalg.benchmark.dimensionwise;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class DimensionWiseBenchmarkPerformer extends BaseBenchmarkPerformer {

    public DimensionWiseBenchmarkPerformer(int nTimes) {
        super(new DimensionWiseOpRunner(),nTimes);
    }



}
