package org.nd4j.linalg.benchmark.fft;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;

/**
 * @author Adam Gibson
 */
public class FFTOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(1000);
    INDArray dup = arr.dup();


    @Override
    public void runOp() {
        FFT.fft(arr);
    }
}
