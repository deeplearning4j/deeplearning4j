/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.arbiter.optimize.generator.genetic.culling;

/**
 * An elitist cull operator that discards the chromosomes with the worst fitness while keeping the best ones.
 *
 * @author Alexandre Boulanger
 */
public class LeastFitCullOperator extends RatioCullOperator {

    /**
     * The default cull ratio is 1/3.
     */
    public LeastFitCullOperator() {
        super();
    }

    /**
     * @param cullRatio The ratio of the maximum population size to be culled.<br>
     * For example, a ratio of 1/3 on a population with a maximum size of 30 will cull back a given population to 20.
     */
    public LeastFitCullOperator(double cullRatio) {
        super(cullRatio);
    }

    /**
     * Will discard the chromosomes with the worst fitness until the population size fall back at the culled size.
     */
    @Override
    public void cullPopulation() {
        while (population.size() > culledSize) {
            population.remove(population.size() - 1);
        }
    }
}
