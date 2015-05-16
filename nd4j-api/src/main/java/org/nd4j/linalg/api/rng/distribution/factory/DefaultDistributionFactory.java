/*
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
 *
 */

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
