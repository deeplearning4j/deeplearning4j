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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * A convolution input pre processor.
 * When passing things in to a convolutional net,
 * a 4d tensor is expected of shape:
 * batch size,channels,rows,cols
 *
 * For a typical flattened dataset of images which are of:
 * batch size x rows * cols in size, this gives the
 * equivalent transformation
 * for a convolutional layer of:
 *
 * batch size (inferred from matrix) x channels x rows x columns
 *
 * Note that for any output passed in,
 * the number of columns of the passed in feature matrix
 * must be equal to
 * rows * cols passed in to the pre processor.
 *
 * @author Adam Gibson
 */
@Deprecated
public class ConvolutionInputPreProcessor implements InputPreProcessor {
    private int rows,cols,channels = 1;
    private int[] shape;

    /**
     * Reshape to a channels x rows x columns tensor
     * @param rows the rows
     * @param cols the columns
     * @param channels the channels
     */
    public ConvolutionInputPreProcessor(int rows, int cols, int channels) {
        this.rows = rows;
        this.cols = cols;
        this.channels = channels;
    }

    public ConvolutionInputPreProcessor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    public ConvolutionInputPreProcessor(int[] shape) {
        this.shape = shape;
    }

    public ConvolutionInputPreProcessor(){}

    @Override
    public INDArray preProcess(INDArray input) {
        if(input.shape().length == 4)
            return input;
        if(input.columns() != rows * cols)
            throw new IllegalArgumentException("Output columns must be equal to rows " + rows + " x columns " + cols + " but was instead " + Arrays.toString(input.shape()));

        return input.reshape(input.size(0),channels,rows,cols);
    }

    @Override
    public INDArray backprop(INDArray output){
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
            int outputShape = output.shape()[0];
            System.arraycopy(output.shape(), 1, otherOutputs, 0, otherOutputs.length);
            shape = new int[] {outputShape, ArrayUtil.prod(otherOutputs)};

        }

        return output.reshape(shape);
    }

}
