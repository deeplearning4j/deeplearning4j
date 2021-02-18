/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.layers.variational;

import lombok.Data;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class LossFunctionWrapper implements ReconstructionDistribution {

    private final IActivation activationFn;
    private final ILossFunction lossFunction;

    public LossFunctionWrapper(@JsonProperty("activationFn") IActivation activationFn,
                    @JsonProperty("lossFunction") ILossFunction lossFunction) {
        this.activationFn = activationFn;
        this.lossFunction = lossFunction;
    }

    public LossFunctionWrapper(Activation activation, ILossFunction lossFunction) {
        this(activation.getActivationFunction(), lossFunction);
    }

    @Override
    public boolean hasLossFunction() {
        return true;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        return dataSize;
    }

    @Override
    public double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {

        //NOTE: The returned value here is NOT negative log probability, but it (the loss function value)
        // is equivalent, in terms of being something we want to minimize...

        return lossFunction.computeScore(x, preOutDistributionParams, activationFn, null, average);
    }

    @Override
    public INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams) {
        return lossFunction.computeScoreArray(x, preOutDistributionParams, activationFn, null);
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        return lossFunction.computeGradient(x, preOutDistributionParams, activationFn, null);
    }

    @Override
    public INDArray generateRandom(INDArray preOutDistributionParams) {
        //Loss functions: not probabilistic -> deterministic output
        return generateAtMean(preOutDistributionParams);
    }

    @Override
    public INDArray generateAtMean(INDArray preOutDistributionParams) {
        //Loss functions: not probabilistic -> not random
        INDArray out = preOutDistributionParams.dup();
        return activationFn.getActivation(out, true);
    }

    @Override
    public String toString() {
        return "LossFunctionWrapper(afn=" + activationFn + "," + lossFunction + ")";
    }
}
