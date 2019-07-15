/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.activations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarSet;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.same.OldIdentity;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.scalar.Step;
import org.nd4j.linalg.api.ops.impl.transforms.strict.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.same.Cube;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative;

/**
 * This enum is the factory for the activation function.
 *
 * Created by susaneraly on 12/8/16.
 */
public enum Activation {
    CUBE, ELU, HARDSIGMOID, HARDTANH, IDENTITY, LEAKYRELU, RATIONALTANH, RELU, RELU6,
    RRELU, SIGMOID, SOFTMAX, SOFTPLUS, SOFTSIGN, TANH, RECTIFIEDTANH, SELU, SWISH,
    THRESHOLDEDRELU, GELU;

    /**
     * Creates an instance of the activation function
     *
     * @return an instance of the activation function
     */
    public IActivation getActivationFunction() {
        switch (this) {
            case CUBE:
                return new ActivationCube();
            case ELU:
                return new ActivationELU();
            case HARDSIGMOID:
                return new ActivationHardSigmoid();
            case HARDTANH:
                return new ActivationHardTanH();
            case IDENTITY:
                return new ActivationIdentity();
            case LEAKYRELU:
                return new ActivationLReLU();
            case RATIONALTANH:
                return new ActivationRationalTanh();
            case RECTIFIEDTANH:
                return new ActivationRectifiedTanh();
            case RELU:
                return new ActivationReLU();
            case RELU6:
                return new ActivationReLU6();
            case SELU:
                return new ActivationSELU();
            case SWISH:
                return new ActivationSwish();
            case RRELU:
                return new ActivationRReLU();
            case SIGMOID:
                return new ActivationSigmoid();
            case SOFTMAX:
                return new ActivationSoftmax();
            case SOFTPLUS:
                return new ActivationSoftPlus();
            case SOFTSIGN:
                return new ActivationSoftSign();
            case TANH:
                return new ActivationTanH();
            case THRESHOLDEDRELU:
                return new ActivationThresholdedReLU();
            case GELU:
                return new ActivationGELU();
            default:
                throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
        }
    }

    /**
     * Returns the activation function enum value
     *
     * @param name the case-insensitive opName of the activation function
     * @return the activation function enum value
     */
    public static Activation fromString(String name) {
        return Activation.valueOf(name.toUpperCase());
    }

    /**
     * Get the Activation as a SameDiff variable
     *
     * @param sd    SameDiff instance
     * @param input Input variable to apply the activation function to
     * @return SDVariable: output after applying the activation function
     * @see #asSameDiff(SameDiff, SDVariable)
     */
    public SDVariable asSameDiff(SameDiff sd, SDVariable input) {
        return asSameDiff(null, sd, input);
    }

    /**
     * Get the Activation as a SameDiff variable
     *
     * @param variableName Variable name
     * @param sd           SameDiff instance
     * @param input        Input variable to apply the activation function to
     * @return SDVariable: output after applying the activation function
     */
    public SDVariable asSameDiff(String variableName, SameDiff sd, SDVariable input) {
        switch (this) {
            case CUBE:
                return sd.math().pow(variableName, input, 3.0);
            case ELU:
                return sd.nn().elu(variableName, input);
            case HARDTANH:
                return sd.nn().hardTanh(variableName, input);
            case IDENTITY:
                return sd.identity(variableName, input);
            case LEAKYRELU:
                return sd.nn().leakyRelu(variableName, input, 0.0);
            case RELU:
                return sd.nn().relu(variableName, input, 0.0);
            case SIGMOID:
                return sd.nn().sigmoid(variableName, input);
            case SOFTMAX:
                return sd.nn().softmax(variableName, input);
            case SOFTPLUS:
                return sd.nn().softplus(variableName, input);
            case SOFTSIGN:
                return sd.nn().softsign(variableName, input);
            case TANH:
                return sd.math().tanh(variableName, input);
            case GELU:
                return sd.nn().gelu(variableName, input);
            case HARDSIGMOID:
            case RATIONALTANH:
            case RRELU:
            case RECTIFIEDTANH:
            case SELU:
            case SWISH:
            default:
                throw new UnsupportedOperationException("Activation function not yet supported: " + this);
        }
    }
}
