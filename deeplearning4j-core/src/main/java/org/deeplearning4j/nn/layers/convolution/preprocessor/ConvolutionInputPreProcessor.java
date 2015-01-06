package org.deeplearning4j.nn.layers.convolution.preprocessor;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A convolution input pre processor.
 * When passing things in to a convolutional net, a 4d tensor is expected of shape:
 * batch size,1,rows,cols
 *
 * For a typical flattened dataset of images which are of:
 * batch size x rows * cols in size, this gives the equivalent transformation for a convolutional layer of:
 *
 * batch size (inferred from matrix) x 1 x rows x columns
 *
 * Note that for any output passed in, the number of columns of the passed in feature matrix must be equal to
 * rows * cols passed in to the pre processor.
 *
 * @author Adam Gibson
 */
public class ConvolutionInputPreProcessor implements OutputPreProcessor {
    private int rows,cols;

    public ConvolutionInputPreProcessor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    @Override
    public INDArray preProcess(INDArray output) {
        if(output.columns() != rows * cols)
            throw new IllegalArgumentException("Output columns must be equal to rows " + rows + " x columns " + cols);

        return output.reshape(output.rows(),1,rows,cols);
    }
}
