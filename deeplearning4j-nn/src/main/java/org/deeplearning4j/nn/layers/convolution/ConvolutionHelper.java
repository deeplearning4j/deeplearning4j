/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Helper for the convolution layer.
 *
 * @author saudet
 */
public interface ConvolutionHelper {
    Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray weights, INDArray delta, int[] kernel,
                    int[] strides, int[] pad, INDArray biasGradView, INDArray weightGradView, IActivation afn,
                    AlgoMode mode, ConvolutionMode convolutionMode);

    INDArray preOutput(INDArray input, INDArray weights, INDArray bias, int[] kernel, int[] strides, int[] pad,
                    AlgoMode mode, ConvolutionMode convolutionMode);

    INDArray activate(INDArray z, IActivation afn);
}
