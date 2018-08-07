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

package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Exponential reconstruction distribution.<br>
 * Supports data in range [0,infinity)<br>
 * <p>
 * Parameterization used here: network models distribution parameter gamma, where gamma = log(lambda), with gamma \in (-inf, inf)
 * <p>
 * This means that an input from the decoder of gamma = 0 gives lambda = 1
 * which corresponds to a mean value for the expontial distribution of 1/lambda = 1
 * <p>
 * Regarding the choice of activation function: the parameterization above supports gamma in the range (-infinity,infinity)
 * therefore a symmetric activation function such as "identity" or perhaps "tanh" is preferred.
 *
 * @author Alex Black
 */
@Data
public class ExponentialReconstructionDistribution implements ReconstructionDistribution {

    private final IActivation activationFn;

    public ExponentialReconstructionDistribution() {
        this("identity");
    }

    /**
     * @deprecated Use {@link #ExponentialReconstructionDistribution(Activation)}
     */
    @Deprecated
    public ExponentialReconstructionDistribution(String activationFn) {
        this(Activation.fromString(activationFn).getActivationFunction());
    }

    public ExponentialReconstructionDistribution(Activation activation) {
        this(activation.getActivationFunction());
    }

    public ExponentialReconstructionDistribution(IActivation activationFn) {
        this.activationFn = activationFn;
    }

    @Override
    public boolean hasLossFunction() {
        return false;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        return dataSize;
    }

    @Override
    public double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {
        //p(x) = lambda * exp( -lambda * x)
        //logp(x) = log(lambda) - lambda * x = gamma - lambda * x

        INDArray gamma = preOutDistributionParams.dup();
        activationFn.getActivation(gamma, false);

        INDArray lambda = Transforms.exp(gamma, true);
        double negLogProbSum = -lambda.muli(x).rsubi(gamma).sumNumber().doubleValue();
        if (average) {
            return negLogProbSum / x.size(0);
        } else {
            return negLogProbSum;
        }

    }

    @Override
    public INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams) {

        INDArray gamma = preOutDistributionParams.dup();
        activationFn.getActivation(gamma, false);

        INDArray lambda = Transforms.exp(gamma, true);
        return lambda.muli(x).rsubi(gamma).sum(1).negi();
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        //p(x) = lambda * exp( -lambda * x)
        //logp(x) = log(lambda) - lambda * x = gamma - lambda * x
        //dlogp(x)/dgamma = 1 - lambda * x      (or negative of this for d(-logp(x))/dgamma

        INDArray gamma = activationFn.getActivation(preOutDistributionParams.dup(), true);

        INDArray lambda = Transforms.exp(gamma, true);
        INDArray dLdx = x.mul(lambda).subi(1.0);

        //dL/dz
        return activationFn.backprop(preOutDistributionParams.dup(), dLdx).getFirst();
    }

    @Override
    public INDArray generateRandom(INDArray preOutDistributionParams) {
        INDArray gamma = activationFn.getActivation(preOutDistributionParams.dup(), false);

        INDArray lambda = Transforms.exp(gamma, true);

        //Inverse cumulative distribution function: -log(1-p)/lambda

        INDArray u = Nd4j.rand(preOutDistributionParams.shape());

        //Note here: if u ~ U(0,1) then 1-u ~ U(0,1)
        return Transforms.log(u, false).divi(lambda).negi();
    }

    @Override
    public INDArray generateAtMean(INDArray preOutDistributionParams) {
        //Input: gamma = log(lambda)    ->  lambda = exp(gamma)
        //Mean for exponential distribution: 1/lambda

        INDArray gamma = activationFn.getActivation(preOutDistributionParams.dup(), false);

        INDArray lambda = Transforms.exp(gamma, true);
        return lambda.rdivi(1.0); //mean = 1.0 / lambda
    }

    @Override
    public String toString() {
        return "ExponentialReconstructionDistribution(afn=" + activationFn + ")";
    }
}
