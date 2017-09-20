/*-
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

package org.deeplearning4j.nn.layers.convolution.upsampling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseUpsamplingLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * 1D Upsampling layer.
 * <p>
 * Used for upsampling a 1D convolution. Currently derived from 2D version.
 * For forward and backward pass we add a dummy dimension, apply the 2D version
 * and strip the extra dimension again. Eventually, we will want to migrate to a
 * proper 1D version without this overhead.
 *
 * @author Max Pumperla
 */
@Slf4j
public class Upsampling1D extends Upsampling2D {


    public Upsampling1D(NeuralNetConfiguration conf) {
        super(conf);
    }


    @Override
    public Gradients backpropGradient(Gradients gradients) {
        INDArray origInput = this.input.get(0);
        INDArray epsilon = gradients.get(0);

        int size = ((BaseUpsamplingLayer) layerConf()).getSize();
        epsilon = epsilon.reshape(epsilon.size(0), epsilon.size(1), epsilon.size(2), 1);
        // we replicate the error term times "size" so that backprop works properly on it
        epsilon = epsilon.repeat(3, size);

        this.input.set(0, origInput.reshape(origInput.size(0), origInput.size(1), origInput.size(2), 1));

        Gradients gradientEpsNext = super.backpropGradient(gradients);
        INDArray epsNext = gradientEpsNext.get(0);

        epsNext = epsNext.slice(0, 3);
        this.input.set(0, origInput);

        // Since we aggregate the gradient across "size" slices, we need to normalize afterwards.
        gradientEpsNext.set(0, epsNext.divi(size));
        return gradientEpsNext;
    }


    public INDArray preOutput(boolean training) {
        return preOutput(training, false);
    }

    public INDArray preOutput(boolean training, boolean forBackprop) {
        INDArray input = this.input.get(0);
        INDArray originalInput = input;
        this.input.set(0, input.reshape(input.size(0), input.size(1), input.size(2), 1));

        INDArray preOutput = super.preOutput(training, forBackprop);

        this.input.set(0, originalInput);
        preOutput = preOutput.slice(0, 3);

        return preOutput;
    }


}
