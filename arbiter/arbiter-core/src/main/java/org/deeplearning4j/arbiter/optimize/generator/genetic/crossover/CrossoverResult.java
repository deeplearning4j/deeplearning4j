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

package org.deeplearning4j.arbiter.optimize.generator.genetic.crossover;

import lombok.Data;

/**
 * Returned by a crossover operator
 * 
 * @author Alexandre Boulanger
 */
@Data
public class CrossoverResult {
    /**
    * If false, there was no crossover and the operator simply returned the genes of a random parent.
    * If true, the genes are the result of a crossover.
    */
    private final boolean isModified;

    /**
    * The genes returned by the operator.
    */
    private final double[] genes;

    public CrossoverResult(boolean isModified, double[] genes) {
        this.isModified = isModified;
        this.genes = genes;
    }
}
