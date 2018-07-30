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
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * CompositeReconstructionDistribution is a reconstruction distribution built from multiple other ReconstructionDistribution
 * instances.<br>
 * The typical use is to combine for example continuous and binary data in the same model, or to combine different
 * distributions for continuous variables. In either case, this class allows users to model (for example) the first 10 values
 * as continuous/Gaussian (with a {@link GaussianReconstructionDistribution}, the next 10 values as binary/Bernoulli (with
 * a {@link BernoulliReconstructionDistribution})
 *
 * @author Alex Black
 */
@Data
public class CompositeReconstructionDistribution implements ReconstructionDistribution {

    private final int[] distributionSizes;
    private final ReconstructionDistribution[] reconstructionDistributions;
    private final int totalSize;

    public CompositeReconstructionDistribution(@JsonProperty("distributionSizes") int[] distributionSizes,
                    @JsonProperty("reconstructionDistributions") ReconstructionDistribution[] reconstructionDistributions,
                    @JsonProperty("totalSize") int totalSize) {
        this.distributionSizes = distributionSizes;
        this.reconstructionDistributions = reconstructionDistributions;
        this.totalSize = totalSize;
    }

    private CompositeReconstructionDistribution(Builder builder) {
        distributionSizes = new int[builder.distributionSizes.size()];
        reconstructionDistributions = new ReconstructionDistribution[distributionSizes.length];
        int sizeCount = 0;
        for (int i = 0; i < distributionSizes.length; i++) {
            distributionSizes[i] = builder.distributionSizes.get(i);
            reconstructionDistributions[i] = builder.reconstructionDistributions.get(i);
            sizeCount += distributionSizes[i];
        }
        totalSize = sizeCount;
    }

    public INDArray computeLossFunctionScoreArray(INDArray data, INDArray reconstruction) {
        if (!hasLossFunction()) {
            throw new IllegalStateException("Cannot compute score array unless hasLossFunction() == true");
        }

        //Sum the scores from each loss function...
        int inputSoFar = 0;
        int paramsSoFar = 0;
        INDArray reconstructionScores = null;
        for (int i = 0; i < distributionSizes.length; i++) {
            int thisInputSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisInputSize);


            INDArray dataSubset =
                            data.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSoFar, inputSoFar + thisInputSize));
            INDArray reconstructionSubset = reconstruction.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            if (i == 0) {
                reconstructionScores = getScoreArray(reconstructionDistributions[i], dataSubset, reconstructionSubset);
            } else {
                reconstructionScores
                                .addi(getScoreArray(reconstructionDistributions[i], dataSubset, reconstructionSubset));
            }

            inputSoFar += thisInputSize;
            paramsSoFar += thisParamsSize;
        }

        return reconstructionScores;
    }

    private INDArray getScoreArray(ReconstructionDistribution reconstructionDistribution, INDArray dataSubset,
                    INDArray reconstructionSubset) {
        if (reconstructionDistribution instanceof LossFunctionWrapper) {
            ILossFunction lossFunction = ((LossFunctionWrapper) reconstructionDistribution).getLossFunction();
            //Re: the activation identity here - the reconstruction array already has the activation function applied,
            // so we don't want to apply it again. i.e., we are passing the output, not the pre-output.
            return lossFunction.computeScoreArray(dataSubset, reconstructionSubset, new ActivationIdentity(), null);
        } else if (reconstructionDistribution instanceof CompositeReconstructionDistribution) {
            return ((CompositeReconstructionDistribution) reconstructionDistribution)
                            .computeLossFunctionScoreArray(dataSubset, reconstructionSubset);
        } else {
            throw new UnsupportedOperationException("Cannot calculate composite reconstruction distribution");
        }
    }

    @Override
    public boolean hasLossFunction() {
        for (ReconstructionDistribution rd : reconstructionDistributions) {
            if (!rd.hasLossFunction())
                return false;
        }
        return true;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        if (dataSize != totalSize) {
            throw new IllegalStateException("Invalid input size: Got input size " + dataSize
                            + " for data, but expected input" + " size for all distributions is " + totalSize
                            + ". Distribution sizes: " + Arrays.toString(distributionSizes));
        }

        int sum = 0;
        for (int i = 0; i < distributionSizes.length; i++) {
            sum += reconstructionDistributions[i].distributionInputSize(distributionSizes[i]);
        }

        return sum;
    }

    @Override
    public double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {

        int inputSoFar = 0;
        int paramsSoFar = 0;
        double logProbSum = 0.0;
        for (int i = 0; i < distributionSizes.length; i++) {
            int thisInputSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisInputSize);


            INDArray inputSubset =
                            x.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSoFar, inputSoFar + thisInputSize));
            INDArray paramsSubset = preOutDistributionParams.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            logProbSum += reconstructionDistributions[i].negLogProbability(inputSubset, paramsSubset, average);

            inputSoFar += thisInputSize;
            paramsSoFar += thisParamsSize;
        }

        return logProbSum;
    }

    @Override
    public INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams) {

        int inputSoFar = 0;
        int paramsSoFar = 0;
        INDArray exampleLogProbSum = null;
        for (int i = 0; i < distributionSizes.length; i++) {
            int thisInputSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisInputSize);


            INDArray inputSubset =
                            x.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSoFar, inputSoFar + thisInputSize));
            INDArray paramsSubset = preOutDistributionParams.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            if (i == 0) {
                exampleLogProbSum = reconstructionDistributions[i].exampleNegLogProbability(inputSubset, paramsSubset);
            } else {
                exampleLogProbSum.addi(
                                reconstructionDistributions[i].exampleNegLogProbability(inputSubset, paramsSubset));
            }

            inputSoFar += thisInputSize;
            paramsSoFar += thisParamsSize;
        }

        return exampleLogProbSum;
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        int inputSoFar = 0;
        int paramsSoFar = 0;
        INDArray gradient = Nd4j.createUninitialized(preOutDistributionParams.shape());
        for (int i = 0; i < distributionSizes.length; i++) {
            int thisInputSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisInputSize);


            INDArray inputSubset =
                            x.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSoFar, inputSoFar + thisInputSize));
            INDArray paramsSubset = preOutDistributionParams.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            INDArray grad = reconstructionDistributions[i].gradient(inputSubset, paramsSubset);
            gradient.put(new INDArrayIndex[] {NDArrayIndex.all(),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize)}, grad);

            inputSoFar += thisInputSize;
            paramsSoFar += thisParamsSize;
        }

        return gradient;
    }

    @Override
    public INDArray generateRandom(INDArray preOutDistributionParams) {
        return randomSample(preOutDistributionParams, false);
    }

    @Override
    public INDArray generateAtMean(INDArray preOutDistributionParams) {
        return randomSample(preOutDistributionParams, true);
    }

    private INDArray randomSample(INDArray preOutDistributionParams, boolean isMean) {
        int inputSoFar = 0;
        int paramsSoFar = 0;
        INDArray out = Nd4j.createUninitialized(new long[] {preOutDistributionParams.size(0), totalSize});
        for (int i = 0; i < distributionSizes.length; i++) {
            int thisDataSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisDataSize);

            INDArray paramsSubset = preOutDistributionParams.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            INDArray thisRandomSample;
            if (isMean) {
                thisRandomSample = reconstructionDistributions[i].generateAtMean(paramsSubset);
            } else {
                thisRandomSample = reconstructionDistributions[i].generateRandom(paramsSubset);
            }

            out.put(new INDArrayIndex[] {NDArrayIndex.all(),
                            NDArrayIndex.interval(inputSoFar, inputSoFar + thisDataSize)}, thisRandomSample);

            inputSoFar += thisDataSize;
            paramsSoFar += thisParamsSize;
        }

        return out;
    }

    public static class Builder {

        private List<Integer> distributionSizes = new ArrayList<>();
        private List<ReconstructionDistribution> reconstructionDistributions = new ArrayList<>();

        /**
         * Add another distribution to the composite distribution. This will add the distribution for the next 'distributionSize'
         * values, after any previously added.
         * For example, calling addDistribution(10, X) once will result in values 0 to 9 (inclusive) being modelled
         * by the specified distribution X. Calling addDistribution(10, Y) after that will result in values 10 to 19 (inclusive)
         * being modelled by distribution Y.
         *
         * @param distributionSize    Number of values to model with the specified distribution
         * @param distribution        Distribution to model data with
         */
        public Builder addDistribution(int distributionSize, ReconstructionDistribution distribution) {
            distributionSizes.add(distributionSize);
            reconstructionDistributions.add(distribution);
            return this;
        }

        public CompositeReconstructionDistribution build() {
            return new CompositeReconstructionDistribution(this);
        }
    }
}
