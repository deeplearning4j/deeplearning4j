package org.deeplearning4j.nn.conf.distribution;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Static method for instantiating an nd4j distribution from a configuration object.
 *
 */
public class Distributions {
    public static org.nd4j.linalg.api.rng.distribution.Distribution createDistribution(
            Distribution dist) {
        if(dist instanceof NormalDistribution) {
            NormalDistribution nd = (NormalDistribution) dist;
            return Nd4j.getDistributions().createNormal(nd.getMean(), nd.getStd());
        }
        if(dist instanceof UniformDistribution) {
            UniformDistribution ud = (UniformDistribution) dist;
            return Nd4j.getDistributions().createUniform(ud.getLower(), ud.getUpper());
        }
        if(dist instanceof BinomialDistribution) {
            BinomialDistribution bd = (BinomialDistribution) dist;
            return Nd4j.getDistributions().createBinomial(bd.getNumberOfTrials(), bd.getProbabilityOfSuccess());
        }
        throw new RuntimeException("unknown distribution type: " + dist.getClass());
    }
}
