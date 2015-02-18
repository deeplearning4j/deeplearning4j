/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.activation;

import com.google.common.base.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

import java.io.Serializable;

/**
 * An activation function for a hidden layer for neural networks
 *
 * @author Adam Gibson
 */
public interface ActivationFunction extends Function<INDArray, INDArray>, Serializable {


    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    public ElementWiseOpFactory transformFactory();

    /**
     * Name of the function
     *
     * @return the name of the function
     */
    public String type();


    /**
     * Applies the derivative of this function
     *
     * @param input the input to apply it to
     * @return the derivative of this function applied to
     * the input
     */
    public INDArray applyDerivative(INDArray input);
}
