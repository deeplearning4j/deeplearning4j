package org.deeplearning4j.nn.layers.convolution.preprocessor;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Used for feeding the output of a conv net in to a 2d classifier.
 * Takes the output shape of the convolution in 4d and reshapes it to a 2d
 * by using the first output as the batch size, and taking the columns via the prod operator
 * for the rest. The shape of the output is inferred by the passed in layer.
 *
 * @author Adam Gibson
 */
public class ConvolutionPostProcessor implements OutputPreProcessor {
    private int[] shape;

    public ConvolutionPostProcessor(int[] shape) {
        this.shape = shape;
    }

    public ConvolutionPostProcessor() {}

    @Override
    public INDArray preProcess(INDArray output) {
        if(shape == null) {
            int[] otherOutputs = new int[3];
            int[] outputShape = output.shape();
            for(int i = 0;i < otherOutputs.length; i++) {
                otherOutputs[i] = outputShape[i + 1];

            }
            shape = new int[] {output.shape()[0], ArrayUtil.prod(otherOutputs)};

        }
        return output.reshape(shape);
    }
}
