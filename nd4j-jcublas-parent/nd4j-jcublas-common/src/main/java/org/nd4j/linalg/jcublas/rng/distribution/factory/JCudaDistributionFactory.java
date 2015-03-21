package org.nd4j.linalg.jcublas.rng.distribution.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.factory.DistributionFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.rng.JcudaRandom;
import org.nd4j.linalg.jcublas.rng.distribution.BinomialDistribution;
import org.nd4j.linalg.jcublas.rng.distribution.NormalDistribution;
import org.nd4j.linalg.jcublas.rng.distribution.UniformDistribution;

/**
 * JCuda distribution factory
 *
 * @author Adam Gibson
 */
public class JCudaDistributionFactory implements DistributionFactory {
    @Override
    public Distribution createBinomial(int n, INDArray p) {
        return new BinomialDistribution((JcudaRandom) Nd4j.getRandom(), p, n);
    }

    @Override
    public Distribution createBinomial(int n, double p) {
        return new BinomialDistribution((JcudaRandom) Nd4j.getRandom(), n, p);
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
        return new UniformDistribution((JcudaRandom) Nd4j.getRandom(), min, max);
    }
}
