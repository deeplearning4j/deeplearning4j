/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.convolution.preprocessor;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.OutputPostProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * Used for feeding the output of a conv net in to a 2d classifier.
 * Takes the output shape of the convolution in 4d and reshapes it to a 2d
 * by using the first output as the batch size, and taking the columns via the prod operator
 * for the rest. The shape of the output is inferred by the passed in layer.
 *
 * @author Adam Gibson
 */
@Deprecated
public class ConvolutionOutputPostProcessor implements OutputPostProcessor {
    private int rows,cols,channels = 1;
    private int[] shape;

    public ConvolutionOutputPostProcessor(int[] shape) {
        this.shape = shape;
    }

    public ConvolutionOutputPostProcessor(int rows, int cols, int channels) {
        this.rows = rows;
        this.cols = cols;
        this.channels = channels;
    }

    public ConvolutionOutputPostProcessor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    public ConvolutionOutputPostProcessor(){}

    @Override
    public INDArray process(INDArray output) {
        if(shape == null || ArrayUtil.prod(shape) != output.length()) {
            int[] otherOutputs = null;
            if(output.shape().length == 2) {
                return output;
            } else if(output.shape().length == 4) {
                otherOutputs = new int[3];
            }
            else if(output.shape().length == 3) {
                otherOutputs = new int[2];
            }
            int[] outputShape = output.shape();
            System.arraycopy(outputShape, 1, otherOutputs, 0, otherOutputs.length);
            shape = new int[] {output.shape()[0], ArrayUtil.prod(otherOutputs)};
        }

        return output.reshape(shape);
    }

    @Override
    public INDArray backprop(INDArray input){

        if (input.shape().length == 4)
            return input;
        if (input.columns() != rows * cols)
            throw new IllegalArgumentException("Output columns must be equal to rows " + rows + " x columns " + cols + " but was instead " + Arrays.toString(input.shape()));
        return input.reshape(input.size(0), channels, rows, cols);
    }

}
