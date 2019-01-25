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
    THRESHOLDEDRELU;

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
                return sd.pow(variableName, input, 3.0);
            case ELU:
                return sd.elu(variableName, input);
            case HARDTANH:
                return sd.hardTanh(variableName, input);
            case IDENTITY:
                return sd.identity(variableName, input);
            case LEAKYRELU:
                return sd.leakyRelu(variableName, input, 0.0);
            case RELU:
                return sd.relu(variableName, input, 0.0);
            case SIGMOID:
                return sd.sigmoid(variableName, input);
            case SOFTMAX:
                return sd.softmax(variableName, input);
            case SOFTPLUS:
                return sd.softplus(variableName, input);
            case SOFTSIGN:
                return sd.softsign(variableName, input);
            case TANH:
                return sd.tanh(variableName, input);
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

    /**
     * Get the Activation function as an ND4J Transform, applied on either the input or a copy of the input
     *
     * @param in  Input to apply the activation function op to
     * @param dup If true: duplicate the array before applying the transform. If false: don't duplicate
     * @return The transform op (execute using {@code Nd4j.getExecutioner().exec(op)}
     */
    public Op asTransform(INDArray in, boolean dup) {
        if (dup) {
            in = in.dup();
        }
        switch (this) {
            case CUBE:
                return new Cube(in);
            case ELU:
                return new ELU(in);
            case HARDSIGMOID:
                return new HardSigmoid(in);
            case HARDTANH:
                return new HardTanh(in);
            case IDENTITY:
                return new OldIdentity(in);
            case LEAKYRELU:
                return new LeakyReLU(in);
            case RATIONALTANH:
                return new RationalTanh(in);
            case RELU:
                return new RectifiedLinear(in);
            case SIGMOID:
                return new Sigmoid(in);
            case SOFTMAX:
                return new OldSoftMax(in);
            case SOFTPLUS:
                return new SoftPlus(in);
            case SOFTSIGN:
                return new SoftSign(in);
            case TANH:
                return new Tanh(in);
            case RECTIFIEDTANH:
                return new RectifiedTanh(in);
            case SELU:
                return new SELU(in);
            case SWISH:
                return new Swish(in);
            case RRELU:
            default:
                throw new UnsupportedOperationException("Not supported via this method: " + this);
        }
    }

    /**
     * Get the Activation function <i>derivative</i> (i.e., dOut/dIn) as an ND4J Transform, applied on either the input
     * or a copy of the input
     *
     * @param in  Input to apply the activation function derivative op to
     * @param dup If true: duplicate the array before applying the transform. If false: don't duplicate
     * @return The op (execute using {@code Nd4j.getExecutioner().exec(op)}
     */
    public Op asTransformDerivative(INDArray in, boolean dup) {
        if (dup) {
            in = in.dup();
        }
        switch (this) {
            case CUBE:
                return new CubeDerivative(in);
            case ELU:
                return new ELUDerivative(in);
            case HARDSIGMOID:
                return new HardSigmoidDerivative(in);
            case HARDTANH:
                return new HardTanhDerivative(in);
            case LEAKYRELU:
                return new LeakyReLUDerivative(in);
            case RATIONALTANH:
                return new RationalTanhDerivative(in);
            case SIGMOID:
                return new SigmoidDerivative(in);
            case SOFTPLUS:
                return new Sigmoid(in);
            case SOFTSIGN:
                return new SoftSignDerivative(in);
            case TANH:
                return new TanhDerivative(in);
            case RECTIFIEDTANH:
                return new RectifiedTanhDerivative(in);
            case SELU:
                return new SELUDerivative(in);
            case SWISH:
                return new SwishDerivative(in);
            case SOFTMAX:
                return new SoftMaxDerivative(in);
            case IDENTITY:
                return new ScalarSet(in, 1.0);
            case RELU:
                return new Step(in);
            case RRELU:
            default:
                throw new UnsupportedOperationException("Not supported via this method: " + this);
        }
    }
}
