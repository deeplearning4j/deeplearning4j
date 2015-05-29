package org.nd4j.linalg.benchmark.convolution;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class ConvolutionBenchmarkPerformer extends BaseBenchmarkPerformer {

    public ConvolutionBenchmarkPerformer(int nTimes) {
        super(new ConvolutionOpRunner(),nTimes);
    }



}
