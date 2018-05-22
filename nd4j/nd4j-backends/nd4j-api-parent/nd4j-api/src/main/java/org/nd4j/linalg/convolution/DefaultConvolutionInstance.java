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
 *
 */

package org.nd4j.linalg.convolution;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Default convolution instance (FFT based)
 *
 * @author Adam Gibson
 */
public class DefaultConvolutionInstance extends BaseConvolution {

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kernel to op with
     * @param type   the opType of convolution
     * @param axes   the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    @Override
    public IComplexNDArray convn(IComplexNDArray input, IComplexNDArray kernel, Convolution.Type type, int[] axes) {
        throw new UnsupportedOperationException();
    }

    /**
     * ND Convolution
     *
     * @param input  the input to op
     * @param kernel the kernel to op with
     * @param type   the opType of convolution
     * @param axes   the axes to do the convolution along
     * @return the convolution of the given input and kernel
     */
    @Override
    public INDArray convn(INDArray input, INDArray kernel, Convolution.Type type, int[] axes) {
        throw new UnsupportedOperationException();
    }

}
