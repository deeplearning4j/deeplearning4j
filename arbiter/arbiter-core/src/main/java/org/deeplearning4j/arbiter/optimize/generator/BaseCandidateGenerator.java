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

package org.deeplearning4j.arbiter.optimize.generator;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.util.LeafUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * BaseCandidateGenerator: abstract class upon which {@link RandomSearchGenerator}
 * and {@link GridSearchCandidateGenerator}
 * are built.
 *
 * @param <T> Type of candidates to generate
 */
@Data
@EqualsAndHashCode(exclude = {"rng", "candidateCounter"})
public abstract class BaseCandidateGenerator<T> implements CandidateGenerator {
    protected ParameterSpace<T> parameterSpace;
    protected AtomicInteger candidateCounter = new AtomicInteger(0);
    protected SynchronizedRandomGenerator rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());
    protected Map<String, Object> dataParameters;
    protected boolean initDone = false;

    public BaseCandidateGenerator(ParameterSpace<T> parameterSpace, Map<String, Object> dataParameters,
                                  boolean initDone) {
        this.parameterSpace = parameterSpace;
        this.dataParameters = dataParameters;
        this.initDone = initDone;
    }

    protected void initialize() {
        if(!initDone) {
            //First: collect leaf parameter spaces objects and remove duplicates
            List<ParameterSpace> noDuplicatesList = LeafUtils.getUniqueObjects(parameterSpace.collectLeaves());

            //Second: assign each a number
            int i = 0;
            for (ParameterSpace ps : noDuplicatesList) {
                int np = ps.numParameters();
                if (np == 1) {
                    ps.setIndices(i++);
                } else {
                    int[] values = new int[np];
                    for (int j = 0; j < np; j++)
                        values[j] = i++;
                    ps.setIndices(values);
                }
            }
            initDone = true;
        }
    }

    @Override
    public ParameterSpace<T> getParameterSpace() {
        return parameterSpace;
    }

    @Override
    public void reportResults(OptimizationResult result) {
        //No op
    }

    @Override
    public void setRngSeed(long rngSeed) {
        rng.setSeed(rngSeed);
    }
}
