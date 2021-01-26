/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.arbiter.optimize.generator.genetic;

import lombok.Data;

/**
 * Candidates are stored as Chromosome in the population model
 *
 * @author Alexandre Boulanger
 */
@Data
public class Chromosome {
    /**
     * The fitness score of the genes.
     */
    protected final double fitness;

    /**
     * The genes.
     */
    protected final double[] genes;

    public Chromosome(double[] genes, double fitness) {
        this.genes = genes;
        this.fitness = fitness;
    }
}
