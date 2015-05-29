package org.nd4j.linalg.benchmark.fft;

import org.nd4j.linalg.benchmark.api.BaseBenchmarkPerformer;

/**
 * @author Adam Gibson
 */
public class FFTBenchmarkPerformer extends BaseBenchmarkPerformer {

    public FFTBenchmarkPerformer(int nTimes) {
        super(new FFTOpRunner(),nTimes);
    }



}
