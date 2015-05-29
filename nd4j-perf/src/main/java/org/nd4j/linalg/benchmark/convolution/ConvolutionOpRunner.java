package org.nd4j.linalg.benchmark.convolution;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;

/**
 * @author Adam Gibson
 */
public class ConvolutionOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(1000);


    @Override
    public void runOp() {
        Convolution.conv2d(arr,arr, Convolution.Type.FULL);
    }
}
