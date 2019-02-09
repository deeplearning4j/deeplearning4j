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

package org.deeplearning4j.arbiter.optimize.generator.genetic.population;

import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;

import java.util.ArrayList;
import java.util.List;

/**
 * A population initializer that build an empty population.
 *
 * @author Alexandre Boulanger
 */
public class EmptyPopulationInitializer implements PopulationInitializer {

    /**
     * Initialize an empty population
     *
     * @param size The maximum size of the population.
     * @return The initialized population.
     */
    @Override
    public List<Chromosome> getInitializedPopulation(int size) {
        return new ArrayList<>(size);
    }
}
