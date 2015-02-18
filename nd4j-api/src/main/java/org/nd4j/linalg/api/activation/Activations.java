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

/**
 * Activation Functions for neural nets
 *
 * @author Adam Gibson
 */
public class Activations {

    private static ActivationFunction EXP = new Exp();
    private static ActivationFunction LINEAR = new Linear();
    private static ActivationFunction TANH = new Tanh();
    private static ActivationFunction HARD_TANH = new HardTanh();
    private static ActivationFunction SIGMOID = new Sigmoid();
    private static ActivationFunction SOFTMAX = new SoftMax();
    private static ActivationFunction SOFTMAX_ROWS = new SoftMax(true);

    private static ActivationFunction ROUNDED_LINEAR = new RoundedLinear();
    private static ActivationFunction RECTIFIEDLINEAR = new RoundedLinear();
    private static ActivationFunction MAX_OUT = new MaxOut();


    /**
     * Max out activation: max of input(i,j)
     *
     * @return
     */
    public static ActivationFunction maxOut() {
        return MAX_OUT;
    }

    /**
     * Softmax with row wise features
     *
     * @return the softmax with row wise features
     */
    public static ActivationFunction softMaxRows() {
        return SOFTMAX_ROWS;
    }

    /**
     * Rectified linear, the output: rounded
     *
     * @return the rounded lienar activation function
     */
    public static ActivationFunction rectifiedLinear() {
        return RECTIFIEDLINEAR;
    }

    /**
     * Rounded linear, the output: rounded
     *
     * @return the rounded lienar activation function
     */
    public static ActivationFunction roundedLinear() {
        return ROUNDED_LINEAR;
    }

    /**
     * The e^x function
     *
     * @return the e^x activation function
     */
    public static ActivationFunction exp() {
        return EXP;
    }

    /**
     * Linear activation function, just returns the input as is
     *
     * @return the linear activation function
     */
    public static ActivationFunction linear() {
        return LINEAR;
    }

    /**
     * Tanh function
     *
     * @return
     */
    public static ActivationFunction tanh() {
        return TANH;
    }

    /**
     * Sigmoid function
     *
     * @return
     */
    public static ActivationFunction sigmoid() {
        return SIGMOID;
    }

    /**
     * Hard Tanh is tanh constraining input to -1 to 1
     *
     * @return the hard tanh function
     */
    public static ActivationFunction hardTanh() {
        return HARD_TANH;
    }


    /**
     * Soft max function used for multinomial classification
     *
     * @return
     */
    public static ActivationFunction softmax() {
        return SOFTMAX;
    }
}
