package org.deeplearning4j.nn.conf.layers.variational;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
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
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {
        @JsonSubTypes.Type(value = GaussianReconstructionDistribution.class, name = "Gaussian"),
        @JsonSubTypes.Type(value = BernoulliReconstructionDistribution.class, name = "Bernoulli"),
        @JsonSubTypes.Type(value = CompositeReconstructionDistribution.class, name = "Composite")
})
public interface ReconstructionDistribution extends Serializable {

    /**
     * Get the number of distribution parameters for the given input data size.
     * For example, a Gaussian distribution has 2 parameters value (mean and log(variance)) for each data value,
     * whereas a Bernoulli distribution has only 1 parameter value (probability) for each data value.
     * @param dataSize    Size of the data. i.e., nIn value
     * @return Number of distribution parameters for the given reconstruction distribution
     */
    int distributionInputSize(int dataSize);

    /**
     * Calculate the log probability
     * @param x                           Data to be modelled (reconstructions)
     * @param preOutDistributionParams    Distribution parameters used by this reconstruction distribution (for example,
     *                                    mean and log variance values for Gaussian)
     * @param average                     Whether the log probability should be averaged over the minibatch, or simply summed.
     * @return                            Log probability of the reconstruction given the distribution parameters
     */
    double logProbability(INDArray x, INDArray preOutDistributionParams, boolean average);

    INDArray gradient(INDArray x, INDArray preOutDistributionParams);

}
