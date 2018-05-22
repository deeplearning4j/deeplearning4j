/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.distribution;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Static method for instantiating an nd4j distribution from a configuration object.
 *
 */
public class Distributions {
    private Distributions() {}

    public static org.nd4j.linalg.api.rng.distribution.Distribution createDistribution(Distribution dist) {
        if (dist == null)
            return null;
        if (dist instanceof NormalDistribution) {
            NormalDistribution nd = (NormalDistribution) dist;
            return Nd4j.getDistributions().createNormal(nd.getMean(), nd.getStd());
        }
        if (dist instanceof GaussianDistribution) {
            GaussianDistribution nd = (GaussianDistribution) dist;
            return Nd4j.getDistributions().createNormal(nd.getMean(), nd.getStd());
        }
        if (dist instanceof UniformDistribution) {
            UniformDistribution ud = (UniformDistribution) dist;
            return Nd4j.getDistributions().createUniform(ud.getLower(), ud.getUpper());
        }
        if (dist instanceof BinomialDistribution) {
            BinomialDistribution bd = (BinomialDistribution) dist;
            return Nd4j.getDistributions().createBinomial(bd.getNumberOfTrials(), bd.getProbabilityOfSuccess());
        }
        if (dist instanceof LogNormalDistribution) {
            LogNormalDistribution lnd = (LogNormalDistribution) dist;
            return Nd4j.getDistributions().createLogNormal(lnd.getMean(), lnd.getStd());
        }
        if (dist instanceof TruncatedNormalDistribution) {
            TruncatedNormalDistribution tnd = (TruncatedNormalDistribution) dist;
            return Nd4j.getDistributions().createTruncatedNormal(tnd.getMean(), tnd.getStd());
        }
        if (dist instanceof OrthogonalDistribution) {
            OrthogonalDistribution od = (OrthogonalDistribution) dist;
            return Nd4j.getDistributions().createOrthogonal(od.getGain());
        }
        if (dist instanceof ConstantDistribution) {
            ConstantDistribution od = (ConstantDistribution) dist;
            return Nd4j.getDistributions().createConstant(od.getValue());
        }
        throw new RuntimeException("unknown distribution type: " + dist.getClass());
    }
}
