package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.OrthogonalDistribution;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Sample weights from a provided {@link Distribution}<br>
 * Note that Distribution is not extendable as it is interpreted through
 * {@link Distributions#createDistribution(Distribution)}. This class basically exists for legacy reasons.
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitDistribution implements IWeightInit {

    private final Distribution distribution;

    public WeightInitDistribution(@JsonProperty("distribution") Distribution distribution) {
        if(distribution == null) {
            // Would fail later below otherwise
            throw new IllegalArgumentException("Must set distribution!");
        }
        this.distribution = distribution;
    }

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        //org.nd4j.linalg.api.rng.distribution.Distribution not serializable
        org.nd4j.linalg.api.rng.distribution.Distribution dist = Distributions.createDistribution(distribution);
        if (dist instanceof OrthogonalDistribution) {
            dist.sample(paramView.reshape(order, shape));
        } else {
            dist.sample(paramView);
        }
        return paramView.reshape(order, shape);
    }
}
