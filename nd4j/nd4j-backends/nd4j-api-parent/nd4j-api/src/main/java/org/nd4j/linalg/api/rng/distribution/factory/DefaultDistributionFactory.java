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

package org.nd4j.linalg.api.rng.distribution.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.*;

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
    public Distribution createLogNormal(double mean, double std) {
        return new LogNormalDistribution(mean, std);
    }

    @Override
    public Distribution createTruncatedNormal(double mean, double std) {
        return new TruncatedNormalDistribution(mean, std);
    }

    @Override
    public Distribution createOrthogonal(double gain) {
        return new OrthogonalDistribution(gain);
    }

    @Override
    public Distribution createConstant(double value) {
        return new ConstantDistribution(value);
    }

    @Override
    public Distribution createUniform(double min, double max) {
        return new UniformDistribution(min, max);
    }
}
