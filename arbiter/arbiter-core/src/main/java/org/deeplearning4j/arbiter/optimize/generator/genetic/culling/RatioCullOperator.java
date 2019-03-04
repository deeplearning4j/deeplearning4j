/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the terms of the Apache License, Version 2.0
 * which is available at https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.generator.genetic.culling;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.nd4j.base.Preconditions;

import java.util.List;

/**
 * An abstract base for cull operators that culls back the population to a ratio of its maximum size.
 *
 * @author Alexandre Boulanger
 */
public abstract class RatioCullOperator implements CullOperator {
    private static final double DEFAULT_CULL_RATIO = 1.0 / 3.0;
    protected int culledSize;
    protected List<Chromosome> population;
    protected final double cullRatio;

    /**
     * @param cullRatio The ratio of the maximum population size to be culled.<br>
     * For example, a ratio of 1/3 on a population with a maximum size of 30 will cull back a given population to 20.
     */
    public RatioCullOperator(double cullRatio) {
        Preconditions.checkState(cullRatio >= 0.0 && cullRatio <= 1.0, "Cull ratio must be between 0.0 and 1.0, got %s",
                        cullRatio);

        this.cullRatio = cullRatio;
    }

    /**
     * The default cull ratio is 1/3
     */
    public RatioCullOperator() {
        this(DEFAULT_CULL_RATIO);
    }

    /**
     * Will be called by the population model once created.
     */
    public void initializeInstance(PopulationModel populationModel) {
        this.population = populationModel.getPopulation();
        culledSize = (int) (populationModel.getPopulationSize() * (1.0 - cullRatio) + 0.5);
    }

    /**
     * @return The target population size after culling.
     */
    @Override
    public int getCulledSize() {
        return culledSize;
    }

}
