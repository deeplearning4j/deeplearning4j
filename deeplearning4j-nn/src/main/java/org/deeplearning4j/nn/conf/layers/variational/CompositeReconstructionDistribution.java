package org.deeplearning4j.nn.conf.layers.variational;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
public class CompositeReconstructionDistribution implements ReconstructionDistribution {

    private final int[] distributionSizes;
    private final ReconstructionDistribution[] reconstructionDistributions;
    private final int totalSize;

    private CompositeReconstructionDistribution(Builder builder){
        distributionSizes = new int[builder.distributionSizes.size()];
        reconstructionDistributions = new ReconstructionDistribution[distributionSizes.length];
        int sizeCount = 0;
        for( int i=0; i< distributionSizes.length; i++ ){
            distributionSizes[i] = builder.distributionSizes.get(i);
            reconstructionDistributions[i] = builder.reconstructionDistributions.get(i);
            sizeCount += distributionSizes[i];
        }
        totalSize = sizeCount;
    }

    @Override
    public int distributionInputSize(int dataSize) {
        if(dataSize != totalSize){
            throw new IllegalStateException("Invalid input size: Got input size " + dataSize + " for data, but expected input" +
                    " size for all distributions is " + totalSize + ". Distribution sizes: " + Arrays.toString(distributionSizes));
        }

        int sum = 0;
        for( int i=0; i<distributionSizes.length; i++ ){
            sum += reconstructionDistributions[i].distributionInputSize(distributionSizes[i]);
        }

        return sum;
    }

    @Override
    public double logProbability(INDArray x, INDArray preOutDistributionParams, boolean average) {

        int inputSoFar = 0;
        int paramsSoFar = 0;
        double logProbSum = 0.0;
        for( int i=0; i<distributionSizes.length; i++ ){
            int thisInputSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisInputSize);


            INDArray inputSubset = x.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSoFar, inputSoFar + thisInputSize));
            INDArray paramsSubset = preOutDistributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            logProbSum += reconstructionDistributions[i].logProbability(inputSubset, paramsSubset, average);

            inputSoFar += thisInputSize;
            paramsSoFar += thisParamsSize;
        }

        return logProbSum;
    }

    @Override
    public INDArray gradient(INDArray x, INDArray preOutDistributionParams) {
        int inputSoFar = 0;
        int paramsSoFar = 0;
        INDArray gradient = Nd4j.createUninitialized(preOutDistributionParams.shape());
        for( int i=0; i<distributionSizes.length; i++ ){
            int thisInputSize = distributionSizes[i];
            int thisParamsSize = reconstructionDistributions[i].distributionInputSize(thisInputSize);


            INDArray inputSubset = x.get(NDArrayIndex.all(), NDArrayIndex.interval(inputSoFar, inputSoFar + thisInputSize));
            INDArray paramsSubset = preOutDistributionParams.get(NDArrayIndex.all(), NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize));

            INDArray grad = reconstructionDistributions[i].gradient(inputSubset, paramsSubset);
            gradient.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(paramsSoFar, paramsSoFar + thisParamsSize)}, grad);

            inputSoFar += thisInputSize;
            paramsSoFar += thisParamsSize;
        }

        return gradient;
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
        public Builder addDistribution(int distributionSize, ReconstructionDistribution distribution){
            distributionSizes.add(distributionSize);
            reconstructionDistributions.add(distribution);
            return this;
        }

        public CompositeReconstructionDistribution build(){
            return new CompositeReconstructionDistribution(this);
        }
    }
}
