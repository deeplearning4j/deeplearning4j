/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.transforms.Pad;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import static org.nd4j.autodiff.samediff.ops.SDValidation.validateFloatingPoint;

/**
 * SameDiff general neural network operations<br>
 * Accessible via {@link SameDiff#math()}<br>
 * See also {@link SDCNN} (accessible via {@link SameDiff#cnn()} for convolutional neural network ops.<br>
 * See also {@link SDRNN} (accessible via {@link SameDiff#rnn()} for recurrent neural network ops.<br>
 *
 * @author Alex Black
 */
public class SDNN extends SDOps {
    public SDNN(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * Batch norm operation.
     *
     * @see #batchNorm(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, double, int...)
     */
    public SDVariable batchNorm(SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta, double epsilon, int... axis) {
        return batchNorm(null, input, mean, variance, gamma, beta, true, true, epsilon, axis);
    }

    /**
     * Batch normalization with optional application of gamma/beta args.
     * See {@link #batchNorm(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, double, int...)}
     */
    public SDVariable batchNorm(String name, SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta, boolean applyGamma, boolean applyBeta, double epsilon, int... axis) {
        validateFloatingPoint("batchNorm", "input", input);
        validateFloatingPoint("batchNorm", "mean", mean);
        validateFloatingPoint("batchNorm", "variance", variance);
        validateFloatingPoint("batchNorm", "gamma", gamma);
        validateFloatingPoint("batchNorm", "beta", beta);
        SDVariable res = f().batchNorm(input, mean, variance, gamma, beta, applyGamma, applyBeta, epsilon, axis);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Neural network batch normalization operation.<br>
     * For details, see <a href="https://arxiv.org/abs/1502.03167">https://arxiv.org/abs/1502.03167</a>
     *
     * @param name     Name of the output variable
     * @param input    Input variable.
     * @param mean     Mean value. For 1d axis, this should match input.size(axis)
     * @param variance Variance value. For 1d axis, this should match input.size(axis)
     * @param gamma    Gamma value. For 1d axis, this should match input.size(axis)
     * @param beta     Beta value. For 1d axis, this should match input.size(axis)
     * @param epsilon  Epsilon constant for numerical stability (to avoid division by 0)
     * @param axis     For 2d CNN activations: 1 for NCHW format activations, or 3 for NHWC format activations.<br>
     *                 For 3d CNN activations: 1 for NCDHW format, 4 for NDHWC<br>
     *                 For 1d/RNN activations: 1 for NCW format, 2 for NWC
     * @return Output variable for batch normalization
     */
    public SDVariable batchNorm(String name, SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta, double epsilon, int... axis) {
        return batchNorm(name, input, mean, variance, gamma, beta, true, true, epsilon, axis);
    }

    /**
     * @see #biasAdd(String, SDVariable, SDVariable, boolean)
     */
    public SDVariable biasAdd(SDVariable input, SDVariable bias, boolean nchw) {
        return biasAdd(null, input, bias, nchw);
    }

    /**
     * Bias addition operation: a special case of addition, typically used with CNN 4D activations and a 1D bias vector
     *
     * @param name  Name of the output variable
     * @param input 4d input variable
     * @param bias  1d bias
     * @param nchw  The format - nchw=true means [minibatch, channels, height, width] format; nchw=false - [minibatch, height, width, channels].
     *              Unused for 2d inputs
     * @return Output variable
     */
    public SDVariable biasAdd(String name, SDVariable input, SDVariable bias, boolean nchw) {
        validateFloatingPoint("biasAdd", "input", input);
        validateFloatingPoint("biasAdd", "bias", bias);
        SDVariable ret = f().biasAdd(input, bias, nchw);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @param input                  Input
     * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
     * @return
     */
    public SDVariable dropout(SDVariable input, double inputRetainProbability) {
        return dropout(null, input, inputRetainProbability);
    }

    /**
     * @param input                  Input
     * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
     * @return
     */
    public SDVariable dropout(String name, SDVariable input, double inputRetainProbability) {
        validateFloatingPoint("dropout", input);
        SDVariable res = f().dropout(input, inputRetainProbability);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Element-wise exponential linear unit (ELU) function:<br>
     * out = x if x > 0<br>
     * out = a * (exp(x) - 1) if x <= 0<br>
     * with constant a = 1.0
     * <p>
     * See: <a href="https://arxiv.org/abs/1511.07289">https://arxiv.org/abs/1511.07289</a>
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable elu(SDVariable x) {
        return elu(null, x);
    }

    /**
     * Element-wise exponential linear unit (ELU) function:<br>
     * out = x if x > 0<br>
     * out = a * (exp(x) - 1) if x <= 0<br>
     * with constant a = 1.0
     * <p>
     * See: <a href="https://arxiv.org/abs/1511.07289">https://arxiv.org/abs/1511.07289</a>
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable elu(String name, SDVariable x) {
        validateFloatingPoint("elu", x);
        SDVariable result = f().elu(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * GELU activation function - Gaussian Error Linear Units<br>
     * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
     * This method uses the sigmoid approximation
     *
     * @param x Input
     * @return Output variable - GELU applied to the input
     */
    public SDVariable gelu(SDVariable x) {
        return gelu(null, x);
    }

    /**
     * GELU activation function - Gaussian Error Linear Units<br>
     * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
     * This method uses the sigmoid approximation
     *
     * @param name Name of the output variable. May be null.
     * @param x    Input
     * @return Output variable - GELU applied to the input
     */
    public SDVariable gelu(String name, SDVariable x) {
        validateFloatingPoint("gelu", x);
        SDVariable ret = f().gelu(x, false);    //Defaults to si
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise hard sigmoid function:<br>
     * out[i] = 0 if in[i] <= -2.5<br>
     * out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5<br>
     * out[i] = 1 if in[i] >= 2.5<br>
     *
     * @param in Input variable
     * @return Output variable
     */
    public SDVariable hardSigmoid(SDVariable in) {
        return hardSigmoid(null, in);
    }

    /**
     * Element-wise hard sigmoid function:<br>
     * out[i] = 0 if in[i] <= -2.5<br>
     * out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5<br>
     * out[i] = 1 if in[i] >= 2.5<br>
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @return Output variable
     */
    public SDVariable hardSigmoid(String name, SDVariable in) {
        validateFloatingPoint("hard sigmoid", in);
        SDVariable ret = f().hardSigmoid(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise hard tanh function:<br>
     * out[i] = -1 if in[i] <= -1<br>
     * out[1] = in[i] if -1 < in[i] < 1<br>
     * out[i] = 1 if in[i] >= 1<br>
     *
     * @param in Input variable
     * @return Output variable
     */
    public SDVariable hardTanh(SDVariable in) {
        return hardTanh(null, in);
    }

    /**
     * Element-wise hard tanh function:<br>
     * out[i] = -1 if in[i] <= -1<br>
     * out[1] = in[i] if -1 < in[i] < 1<br>
     * out[i] = 1 if in[i] >= 1<br>
     *
     * @param name Output variable name
     * @param in   Input variable
     * @return Output variable
     */
    public SDVariable hardTanh(String name, SDVariable in) {
        validateFloatingPoint("hard Tanh", in);
        SDVariable result = f().hardTanh(in);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Derivative (dOut/dIn) of the element-wise hard Tanh function - {@link #hardTanh(SDVariable)}
     *
     * @param x Input
     * @return Output variable
     */
    public SDVariable hardTanhDerivative(SDVariable x) {
        return hardTanhDerivative(null, x);
    }

    /**
     * Derivative (dOut/dIn) of the element-wise hard Tanh function - {@link #hardTanh(SDVariable)}
     *
     * @param name Output variable name
     * @param x    Input
     * @return Output variable
     */
    public SDVariable hardTanhDerivative(String name, SDVariable x) {
        validateFloatingPoint("hard Tanh derivative", x);
        SDVariable result = f().hardTanhDerivative(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise leaky ReLU function:<br>
     * out = x if x >= 0.0<br>
     * out = alpha * x if x < cutoff<br>
     * Alpha value is most commonly set to 0.01
     *
     * @param x     Input variable
     * @param alpha Cutoff - usually 0.0
     * @return Output variable
     */
    public SDVariable leakyRelu(SDVariable x, double alpha) {
        return leakyRelu(null, x, alpha);
    }

    /**
     * Element-wise leaky ReLU function:<br>
     * out = x if x >= 0.0<br>
     * out = alpha * x if x < cutoff<br>
     * Alpha value is most commonly set to 0.01
     *
     * @param x     Input variable
     * @param alpha Cutoff - usually 0.0
     * @return Output variable
     */
    public SDVariable leakyRelu(String name, SDVariable x, double alpha) {
        validateFloatingPoint("leaky ReLU", x);
        SDVariable result = f().leakyRelu(x, alpha);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Leaky ReLU derivative: dOut/dIn given input.<br>
     * See {@link #leakyRelu(String, SDVariable, double)}
     *
     * @param x     Input variable
     * @param alpha Alpha value
     * @return Output variable
     */
    public SDVariable leakyReluDerivative(String name, SDVariable x, double alpha) {
        validateFloatingPoint("leaky ReLU derivative", x);
        SDVariable result = f().leakyReluDerivative(x, alpha);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #linear(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable linear(SDVariable input, SDVariable weights, SDVariable bias) {
        return linear(null, input, weights, bias);
    }

    /**
     * Linear layer operation: out = mmul(in,w) + bias<br>
     * Note that bias array is optional
     *
     * @param name    Name of the output variable
     * @param input   Input data
     * @param weights Weights variable
     * @param bias    Optional bias variable (may be null)
     * @return Output variable
     */
    public SDVariable linear(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        validateFloatingPoint("linear", "input", input);
        validateFloatingPoint("linear", "weights", weights);
        validateFloatingPoint("linear", "bias", bias);
        SDVariable res = f().xwPlusB(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))
     *
     * @param x Input Variable
     * @return Output variable
     */
    public SDVariable logSigmoid(SDVariable x) {
        return logSigmoid(null, x);
    }

    /**
     * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))
     *
     * @param name Name of the output variable
     * @param x    Input Variable
     * @return Output variable
     */
    public SDVariable logSigmoid(String name, SDVariable x) {
        validateFloatingPoint("log sigmoid", x);
        SDVariable ret = f().logSigmoid(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Log softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable logSoftmax(SDVariable x) {
        return logSoftmax(null, x);
    }

    /**
     * Log softmax activation
     *
     * @param name Variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable logSoftmax(String name, SDVariable x) {
        validateFloatingPoint("log softmax", x);
        SDVariable ret = f().logSoftmax(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Log softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable logSoftmax(SDVariable x, int dimension) {
        return logSoftmax(null, x, dimension);
    }

    /**
     * Log softmax activation
     *
     * @param name Variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable logSoftmax(String name, SDVariable x, int dimension) {
        validateFloatingPoint("log softmax", x);
        SDVariable ret = f().logSoftmax(x, dimension);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise rectified linear function with specified cutoff:<br>
     * out[i] = in[i] if in[i] >= cutoff
     * out[i] = 0 otherwise
     *
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu(SDVariable x, double cutoff) {
        return relu(null, x, cutoff);
    }

    /**
     * Element-wise rectified linear function with specified cutoff:<br>
     * out[i] = in[i] if in[i] >= cutoff
     * out[i] = 0 otherwise
     *
     * @param name   Output variable name
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu(String name, SDVariable x, double cutoff) {
        validateFloatingPoint("ReLU", x);
        SDVariable result = f().relu(x, cutoff);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise "rectified linear 6" function with specified cutoff:<br>
     * out[i] = min(max(in, cutoff), 6)
     *
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu6(SDVariable x, double cutoff) {
        return relu6(null, x, cutoff);
    }

    /**
     * Element-wise "rectified linear 6" function with specified cutoff:<br>
     * out[i] = min(max(in, cutoff), 6)
     *
     * @param name   Output variable name
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu6(String name, SDVariable x, double cutoff) {
        validateFloatingPoint("ReLU6", x);
        SDVariable result = f().relu6(x, cutoff);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #reluLayer(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
        return reluLayer(null, input, weights, bias);
    }

    /**
     * ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)<br>
     * Note that bias array is optional
     *
     * @param name    Name of the output variable
     * @param input   Input data
     * @param weights Weights variable
     * @param bias    Optional bias variable (may be null)
     * @return Output variable
     */
    public SDVariable reluLayer(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        validateFloatingPoint("reluLayer", "input", input);
        validateFloatingPoint("reluLayer", "weights", weights);
        validateFloatingPoint("reluLayer", "bias", bias);
        SDVariable res = f().reluLayer(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * See {@link #prelu(String, SDVariable, SDVariable, int...)}.
     */
    public SDVariable prelu(@NonNull SDVariable input, @NonNull SDVariable alpha, @NonNull int... sharedAxes){
        return f().prelu(input, alpha, sharedAxes);
    }

    /**
     * PReLU (Parameterized Rectified Linear Unit) operation.  Like LeakyReLU with a learnable alpha:<br>
     * out[i] = in[i] if in[i] >= 0<br>
     * out[i] = in[i] * alpha[i] otherwise<br>
     *
     * sharedAxes allows you to share learnable parameters along axes.
     * For example, if the input has shape [batchSize, channels, height, width]
     * and you want each channel to have its own cutoff, use sharedAxes = [2, 3] and an
     * alpha with shape [channels].
     *
     * @param name    Name of the output variable
     * @param input   Input data
     * @param alpha   The cutoff variable.  Note that the batch dimension (the 0th, whether it is batch or not) should not be part of alpha.
     * @param sharedAxes Which axes to share cutoff parameters along.
     * @return Output variable
     */
    public SDVariable prelu(String name, @NonNull SDVariable input, @NonNull SDVariable alpha, @NonNull int... sharedAxes){
        SDVariable res = f().prelu(input, alpha, sharedAxes);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a>
     * <br>
     * out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0<br>
     * Uses default lcale and alpha values.
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable selu(SDVariable x) {
        return selu(null, x);
    }

    /**
     * Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a>
     * <br>
     * out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0<br>
     * Uses default lcale and alpha values.
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable selu(String name, SDVariable x) {
        validateFloatingPoint("selu", x);
        SDVariable ret = f().selu(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))
     *
     * @param x Input Variable
     * @return Output variable
     */
    public SDVariable sigmoid(SDVariable x) {
        return sigmoid(null, x);
    }

    /**
     * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))
     *
     * @param name Output variable name
     * @param x    Input Variable
     * @return Output variable
     */
    public SDVariable sigmoid(String name, SDVariable x) {
        validateFloatingPoint("sigmoid", x);
        SDVariable result = f().sigmoid(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut
     *
     * @param x   Input Variable
     * @param wrt Gradient at the output - dL/dOut. Must have same shape as the input
     * @return Output variable
     */
    public SDVariable sigmoidDerivative(SDVariable x, SDVariable wrt) {
        return sigmoidDerivative(null, x, wrt);
    }

    /**
     * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut
     *
     * @param name Output variable name
     * @param x    Input Variable
     * @param wrt  Gradient at the output - dL/dOut. Must have same shape as the input
     * @return Output variable
     */
    public SDVariable sigmoidDerivative(String name, SDVariable x, SDVariable wrt) {
        validateFloatingPoint("sigmoidDerivative", x);
        SDVariable result = f().sigmoidDerivative(x, wrt);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Softmax activation on dimension 1.
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softmax(SDVariable x) {
        return softmax(null, x);
    }

    /**
     * Softmax activation on dimension 1.
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softmax(String name, SDVariable x) {
        validateFloatingPoint("softmax", x);
        SDVariable result = f().softmax(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softmax(SDVariable x, int dimension) {
        return softmax(null, x, dimension);
    }

    /**
     * Softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softmax(String name, SDVariable x, int dimension) {
        validateFloatingPoint("softmax", x);
        SDVariable result = f().softmax(x, dimension);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param x
     * @return
     */
    public SDVariable softmaxDerivative(String name, SDVariable x, SDVariable wrt) {
        return softmaxDerivative(name, x, wrt, null);
    }

    public SDVariable softmaxDerivative(String name, SDVariable x, SDVariable wrt, Integer dimension) {
        validateFloatingPoint("softmaxDerivative", x);
        SDVariable result = f().softmaxDerivative(x, wrt, dimension);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise softplus function: out = log(exp(x) + 1)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softplus(SDVariable x) {
        return softplus(null, x);
    }

    /**
     * Element-wise softplus function: out = log(exp(x) + 1)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable softplus(String name, SDVariable x) {
        validateFloatingPoint("softplus", x);
        SDVariable result = f().softplus(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise softsign function: out = x / (abs(x) + 1)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softsign(SDVariable x) {
        return softsign(null, x);
    }

    /**
     * Element-wise softsign function: out = x / (abs(x) + 1)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable softsign(String name, SDVariable x) {
        validateFloatingPoint("softsign", x);
        SDVariable result = f().softsign(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise derivative (dOut/dIn) of the softsign function {@link #softsign(SDVariable)}
     *
     * @param x Input variable
     * @return Output varible
     */
    public SDVariable softsignDerivative(SDVariable x) {
        return softsignDerivative(null, x);
    }

    /**
     * Element-wise derivative (dOut/dIn) of the softsign function {@link #softsign(SDVariable)}
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output varible
     */
    public SDVariable softsignDerivative(String name, SDVariable x) {
        validateFloatingPoint("softsignDerivative", x);
        SDVariable result = f().softsignDerivative(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
     * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a>
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable swish(SDVariable x) {
        return swish(null, x);
    }

    /**
     * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
     * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a>
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable swish(String name, SDVariable x) {
        validateFloatingPoint("swish", x);
        SDVariable ret = f().swish(x);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable tanh(String name, SDVariable x) {
        return sd.math().tanh(name, x);
    }

    public SDVariable tanh(SDVariable x) {
        return sd.math().tanh(x);
    }

    /**
     * Apply Layer Normalization
     *
     * y = gain * standardize(x) + bias
     *
     * @return Output variable
     */
    public SDVariable layerNorm(SDVariable input, SDVariable gain, SDVariable bias, boolean channelsFirst, int... dimensions) {
        return layerNorm(null, input, gain, bias, channelsFirst, dimensions);
    }

    /**
     * Apply Layer Normalization
     *
     * y = gain * standardize(x) + bias
     *
     * @param name Name of the output variable
     * @param input Input variable
     * @param gain gain
     * @param bias bias
     * @param channelsFirst For 2D input - unused. True for NCHW (minibatch, channels, height, width), false for NHWC data
     * @param dimensions Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs
     * @return Output variable
     */
    public SDVariable layerNorm(String name, SDVariable input, SDVariable gain, SDVariable bias, boolean channelsFirst, int... dimensions) {
        validateFloatingPoint("layerNorm", "input", input);
        validateFloatingPoint("layerNorm", "gain", gain);
        validateFloatingPoint("layerNorm", "bias", bias);
        SDVariable result = f().layerNorm(input, gain, bias, channelsFirst, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Apply Layer Normalization without bias
     *
     * y = gain * standardize(x)
     *
     * @return Output variable
     */
    public SDVariable layerNorm(SDVariable input, SDVariable gain, boolean channelsFirst, int... dimensions) {
        return layerNorm((String)null, input, gain, channelsFirst, dimensions);
    }

    /**
     * Apply Layer Normalization
     *
     * y = gain * standardize(x)
     *
     * @param name Name of the output variable
     * @param input Input variable
     * @param gain gain
     * @return Output variable
     */
    public SDVariable layerNorm(String name, SDVariable input, SDVariable gain, boolean channelsFirst, int... dimensions) {
        validateFloatingPoint("layerNorm", "input", input);
        validateFloatingPoint("layerNorm", "gain", gain);
        SDVariable result = f().layerNorm(input, gain, channelsFirst, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * See {@link #pad(SDVariable, SDVariable, double)}
     */
    public SDVariable pad(SDVariable input, int[][] padding, double constant){
        return pad(input, sd.constant(Nd4j.createFromArray(padding)), constant);
    }

    /**
     * Perform padding on the given array, where padded values are the specified constant.<br>
     * Example:<br>
     * Input array:<br>
     * [1, 2]<br>
     * [3, 4]<br>
     * Padding array:<br>
     * [2, 0]<br>
     * [1, 1]<br>
     * Contant = 0<br>
     * Result:<br>
     * [0, 0, 0, 0]<br>
     * [0, 0, 0, 0]<br>
     * [0, 1, 2, 0]<br>
     * [0, 3, 4, 0]<br>
     * <br>
     *
     *
     * @param input    Input array to pad
     * @param padding  Padding array
     * @param constant Constant to use for padded values
     * @return Padded array
     */
    public SDVariable pad(SDVariable input, SDVariable padding, double constant){
        return pad(null, input, padding, Pad.Mode.CONSTANT, constant);
    }

    /**
     * As per {@link #pad(SDVariable, SDVariable, double)} but also supports multiple {@link Pad.Mode} modes.<br>
     * Example:
     * Input array:<br>
     * [1, 2]<br>
     * [3, 4]<br>
     * [5, 6]<br>
     * Padding array:<br>
     * [2, 0]<br>
     * [1, 1]<br>
     * Contant = 0<br>
     * Result: CONSTANT mode<br>
     * [0, 0, 0, 0]<br>
     * [0, 0, 0, 0]<br>
     * [0, 1, 2, 0]<br>
     * [0, 3, 4, 0]<br>
     * [0, 5, 6, 0]<br>
     * <br>
     * Result: SYMMETRIC mode<br>
     * [3, 3, 4, 4]<br>
     * [1, 1, 2, 2]<br>
     * [1, 1, 2, 2]<br>
     * [3, 3, 4, 4]<br>
     * [5, 5, 6, 6]<br>
     * <br>
     * Result: REFLECT:<br>
     * [6, 5, 6, 0]<br>
     * [2, 3, 4, 3]<br>
     * [2, 1, 2, 1]<br>
     * [4, 3, 4, 3]<br>
     * [6, 5, 6, 5]<br>
     * <br>
     * @param outputName
     * @param input
     * @param padding
     * @param mode
     * @param constant
     * @return
     */
    public SDVariable pad(String outputName, SDVariable input, SDVariable padding, Pad.Mode mode, double constant){
        SDVariable out = f().pad(input, padding, mode, constant);
        return updateVariableNameAndReference(out, outputName);
    }

    /**
     * This operation performs dot product attention on the given timeseries input with the given queries
     * @see #dotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     */
    public SDVariable dotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable mask, boolean scaled){
        return dotProductAttention(null, queries, keys, values, mask, scaled);
    }

    /**
     * This operation performs dot product attention on the given timeseries input with the given queries
     * @see #dotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     */
    public SDVariable dotProductAttention(String name, SDVariable queries, SDVariable keys, SDVariable values, SDVariable mask, boolean scaled){
        final SDVariable result = f().dotProductAttention(queries, keys, values, mask, scaled);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * This operation performs dot product attention on the given timeseries input with the given queries
     * @see #dotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     */
    public List<SDVariable> dotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable mask, boolean scaled, boolean withWeights){
        return dotProductAttention(null, queries, keys, values, mask, scaled, withWeights);
    }


    /**
     * This operation performs dot product attention on the given timeseries input with the given queries
     * out = sum(similarity(k_i, q) * v_i)
     *
     * similarity(k, q) = softmax(k * q) where x * q is the dot product of x and q
     *
     * Optionally with normalization step:
     * similarity(k, q) = softmax(k * q / sqrt(size(q))
     *
     * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, p. 4, eq. 1)
     *
     * Note: This supports multiple queries at once, if only one query is available the queries vector still has to
     * be 3D but can have queryCount = 1
     *
     * Note: keys and values usually is the same array. If you want to use it as the same array, simply pass it for
     * both.
     *
     * Note: Queries, keys and values must either be all rank 3 or all rank 4 arrays. Mixing them doesn't work. The
     * output rank will depend on the input rank.
     *
     * @param queries input 3D array "queries" of shape [batchSize, featureKeys, queryCount]
     *                or 4D array of shape [batchSize, numHeads, featureKeys, queryCount]
     * @param keys input 3D array "keys" of shape [batchSize, featureKeys, timesteps]
     *             or 4D array of shape [batchSize, numHeads, featureKeys, timesteps]
     * @param values input 3D array "values" of shape [batchSize, featureValues, timesteps]
     *               or 4D array of shape [batchSize, numHeads, featureValues, timesteps]
     * @param mask OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]
     * @param scaled normalization, false -> do not apply normalization, true -> apply normalization
     * @param withWeights return attention weights as well, false -> only one output, true -> two outputs
     *
     * Output Arrays:
     * @return [ Attention result arrays of shape [batchSize, featureValues, queryCount] or [batchSize, numHeads, featureValues, queryCount],
     *           (optionally) Attention Weights of shape [batchSize, timesteps, queryCount] or [batchSize, numHeads, timesteps, queryCount]]
     */
    public List<SDVariable> dotProductAttention(String name, SDVariable queries, SDVariable keys, SDVariable values, SDVariable mask, boolean scaled, boolean withWeights){
        List<SDVariable> result = f().dotProductAttention(queries, keys, values, mask, scaled, withWeights);
        if(withWeights){
            return Collections.singletonList(updateVariableNameAndReference(result.get(0), name));
        }else{
            return Arrays.asList(
                    updateVariableNameAndReference(result.get(0), name),
                    updateVariableNameAndReference(result.get(1), name+":weights")
            );
        }
    }

    /**
     * This performs multi-headed dot product attention on the given timeseries input
     * @see #multiHeadDotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     */
    public SDVariable multiHeadDotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable mask, boolean scaled){
        return multiHeadDotProductAttention(null, queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled);
    }

    /**
     * This performs multi-headed dot product attention on the given timeseries input
     * @see #multiHeadDotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     */
    public SDVariable multiHeadDotProductAttention(String name, SDVariable queries, SDVariable keys, SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable mask, boolean scaled){
        final SDVariable result = f().multiHeadDotProductAttention(queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * This performs multi-headed dot product attention on the given timeseries input
     * @see #multiHeadDotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     */
    public List<SDVariable> multiHeadDotProductAttention(SDVariable queries, SDVariable keys, SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable mask, boolean scaled, boolean withWeights){
        return multiHeadDotProductAttention(null, queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, withWeights);
    }


    /**
     * This performs multi-headed dot product attention on the given timeseries input
     * out = concat(head_1, head_2, ..., head_n) * Wo
     * head_i = dot_product_attention(Wq_i*q, Wk_i*k, Wv_i*v)
     *
     * Optionally with normalization when calculating the attention for each head.
     *
     * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")
     *
     * This makes use of dot_product_attention OP support for rank 4 inputs.
     * @see #dotProductAttention(String, SDVariable, SDVariable, SDVariable, SDVariable, boolean, boolean)
     *
     * @param queries input 3D array "queries" of shape [batchSize, featureKeys, queryCount]
     * @param keys input 3D array "keys" of shape [batchSize, featureKeys, timesteps]
     * @param values input 3D array "values" of shape [batchSize, featureValues, timesteps]
     * @param Wq input query projection weights of shape [numHeads, projectedKeys, featureKeys]
     * @param Wk input key projection weights of shape [numHeads, projectedKeys, featureKeys]
     * @param Wv: input value projection weights of shape [numHeads, projectedValues, featureValues]
     * @param Wo: output projection weights of shape [numHeads * projectedValues, outSize]
     * @param mask OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]
     * @param scaled normalization, false -> do not apply normalization, true -> apply normalization
     * @param withWeights return attention weights as well, false -> only one output, true -> two outputs
     *
     * Output Arrays:
     * @return [ Attention result arrays of shape [batchSize, outSize, queryCount]
     *           (optionally) Attention Weights of shape [batchSize, numHeads, timesteps, queryCount]
     */
    public List<SDVariable> multiHeadDotProductAttention(String name, SDVariable queries, SDVariable keys, SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo, SDVariable mask, boolean scaled, boolean withWeights){
        List<SDVariable> result = f().multiHeadDotProductAttention(queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, withWeights);
        if(withWeights){
            return Collections.singletonList(updateVariableNameAndReference(result.get(0), name));
        }else{
            return Arrays.asList(
                    updateVariableNameAndReference(result.get(0), name),
                    updateVariableNameAndReference(result.get(1), name+":weights")
            );
        }
    }

    /**
     * Max pooling on the input and outputs both max values and indices
     *
     * @param name  Name of the output variable
     * @param x input array
     * @return output array and argmax array
     */
    public SDVariable[] maxPoolWithArgmax(String[] names, SDVariable x, Pooling2DConfig pooling2DConfig) {
        SDVariable[] res = f().maxPoolWithArgmaxs(x, pooling2DConfig);
        return sd.updateVariableNamesAndReferences(res, names);
    }

    /**
     * Batch normalization
     *
     * @param name  Name of the output variable
     * @param x 4D array
     * @param scale vector for scaling factor of normalized x
     * @param offset vector to shift to the normalized x
     * @param dataFormat integer scalar - data format
     * @param isTraining boolean scalar - is training mode
     * @return y: 4D array
     *         batch_mean: vector
     *         batch_var: vector
     */
    public SDVariable[] fusedBatchNorm(String[] names, SDVariable x, SDVariable scale, SDVariable offset,
                                       SDVariable dataFormat, SDVariable isTraining) {
        SDVariable[] res = f().fusedBatchNorm(x,scale,offset,dataFormat,isTraining);
        return sd.updateVariableNamesAndReferences(res, names);
    }
}
