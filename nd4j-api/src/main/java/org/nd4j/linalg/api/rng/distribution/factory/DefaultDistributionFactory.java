package org.nd4j.linalg.api.rng.distribution.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.BinomialDistribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;

/**
 * Default distribution factory
 *
 * @author Adam Gibson
 */
public class DefaultDistributionFactory implements DistributionFactory {
    @Override
    public Distribution createBinomial(int n, INDArray p) {
        return new BinomialDistribution(n, p);
    }

    @Override
    public Distribution createBinomial(int n, double p) {
        return new BinomialDistribution(n, p);
    }

    @Override
    public Distribution createNormal(INDArray mean, double std) {
        return new NormalDistribution(mean, std);
    }

    @Override
    public Distribution createNormal(double mean, double std) {
        return new NormalDistribution(mean, std);
    }

    @Override
    public Distribution createUniform(double min, double max) {
        return new UniformDistribution(min, max);
    }
}
