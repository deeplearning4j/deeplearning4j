package org.deeplearning4j.nn.conf.layers.variational;

import org.deeplearning4j.nn.conf.serde.legacyformat.LegacyReconstructionDistributionDeserializerHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * The ReconstructionDistribution is used with variational autoencoders {@link VariationalAutoencoder}
 * to specify the form of the distribution p(data|x). For example, real-valued data could be modelled
 * by a {@link GaussianReconstructionDistribution}, whereas binary data could be modelled by a {@link BernoulliReconstructionDistribution}.<br>
 * <p>
 * To model multiple types of data in the one data vector, use {@link CompositeReconstructionDistribution}.
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyReconstructionDistributionDeserializerHelper.class)
public interface ReconstructionDistribution extends Serializable {

    /**
     * Does this reconstruction distribution has a standard neural network loss function (such as mean squared error,
     * which is deterministic) or is it a standard VAE with a probabilistic reconstruction distribution?
     * @return true if the reconstruction distribution has a loss function only (and no probabilistic reconstruction distribution)
     */
    boolean hasLossFunction();

    /**
     * Get the number of distribution parameters for the given input data size.
     * For example, a Gaussian distribution has 2 parameters value (mean and log(variance)) for each data value,
     * whereas a Bernoulli distribution has only 1 parameter value (probability) for each data value.
     *
     * @param dataSize Size of the data. i.e., nIn value
     * @return Number of distribution parameters for the given reconstruction distribution
     */
    int distributionInputSize(int dataSize);

    /**
     * Calculate the negative log probability (summed or averaged over each example in the minibatch)
     *
     * @param x                        Data to be modelled (reconstructions)
     * @param preOutDistributionParams Distribution parameters used by this reconstruction distribution (for example,
     *                                 mean and log variance values for Gaussian)
     * @param average                  Whether the log probability should be averaged over the minibatch, or simply summed.
     * @return Average or sum of negative log probability of the reconstruction given the distribution parameters
     */
    double negLogProbability(INDArray x, INDArray preOutDistributionParams, boolean average);

    /**
     * Calculate the negative log probability for each example individually
     *
     * @param x                        Data to be modelled (reconstructions)
     * @param preOutDistributionParams Distribution parameters used by this reconstruction distribution (for example,
     *                                 mean and log variance values for Gaussian) - before applying activation function
     * @return Negative log probability of the reconstruction given the distribution parameters, for each example individually.
     * Column vector, shape [numExamples, 1]
     */
    INDArray exampleNegLogProbability(INDArray x, INDArray preOutDistributionParams);

    /**
     * Calculate the gradient of the negative log probability with respect to the preOutDistributionParams
     *
     * @param x                        Data
     * @param preOutDistributionParams Distribution parameters used by this reconstruction distribution (for example,
     *                                 mean and log variance values for Gaussian) - before applying activation function
     * @return Gradient with respect to the preOutDistributionParams
     */
    INDArray gradient(INDArray x, INDArray preOutDistributionParams);

    /**
     * Randomly sample from P(x|z) using the specified distribution parameters
     *
     * @param preOutDistributionParams Distribution parameters used by this reconstruction distribution (for example,
     *                                 mean and log variance values for Gaussian) - before applying activation function
     * @return A random sample of x given the distribution parameters
     */
    INDArray generateRandom(INDArray preOutDistributionParams);

    /**
     * Generate a sample from P(x|z), where x = E[P(x|z)]
     * i.e., return the mean value for the distribution
     *
     * @param preOutDistributionParams Distribution parameters used by this reconstruction distribution (for example,
     *                                 mean and log variance values for Gaussian) - before applying activation function
     * @return A deterministic sample of x (mean/expected value) given the distribution parameters
     */
    INDArray generateAtMean(INDArray preOutDistributionParams);
}
