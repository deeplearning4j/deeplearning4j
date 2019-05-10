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

import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * RandomSearchGenerator: generates candidates at random.<br>
 * Note: if a probability distribution is provided for continuous hyperparameters,
 * this will be taken into account
 * when generating candidates. This allows the search to be weighted more towards
 * certain values according to a probability
 * density. For example: generate samples for learning rate according to log uniform distribution
 *
 * @author Alex Black
 */
@Slf4j
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"numValuesPerParam", "totalNumCandidates", "order", "candidateCounter", "rng", "candidate"})
public class RandomSearchGenerator extends BaseCandidateGenerator {

    @JsonCreator
    public RandomSearchGenerator(@JsonProperty("parameterSpace") ParameterSpace<?> parameterSpace,
                    @JsonProperty("dataParameters") Map<String, Object> dataParameters,
                                 @JsonProperty("initDone") boolean initDone) {
        super(parameterSpace, dataParameters, initDone);
        initialize();
    }

    public RandomSearchGenerator(ParameterSpace<?> parameterSpace, Map<String,Object> dataParameters){
        this(parameterSpace, dataParameters, false);
    }

    public RandomSearchGenerator(ParameterSpace<?> parameterSpace){
        this(parameterSpace, null, false);
    }


    @Override
    public boolean hasMoreCandidates() {
        return true;
    }

    @Override
    public Candidate getCandidate() {
        double[] randomValues = new double[parameterSpace.numParameters()];
        for (int i = 0; i < randomValues.length; i++)
            randomValues[i] = rng.nextDouble();

        Object value = null;
        Exception e = null;
        try {
            value = parameterSpace.getValue(randomValues);
        } catch (Exception e2) {
            log.warn("Error getting configuration for candidate", e2);
            e = e2;
        }

        return new Candidate(value, candidateCounter.getAndIncrement(), randomValues, dataParameters, e);
    }

    @Override
    public Class<?> getCandidateType() {
        return null;
    }

    @Override
    public String toString() {
        return "RandomSearchGenerator";
    }
}
