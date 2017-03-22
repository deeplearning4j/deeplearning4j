/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.optimize.api;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * An optimization result represents the results of an optimization run, including the canditate configuration, the
 * trained model, the score for that model, and index of the model
 *
 * @param <C> Type for the model configuration
 * @param <M> Type of the trained model
 * @param <A> Type for any additional evaluation
 * @author Alex Black
 */
@Data
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
public class OptimizationResult<C, M, A> implements Serializable {
    @JsonProperty
    private Candidate<C> candidate;
    @JsonProperty
    private M result;
    @JsonProperty
    private Double score;
    @JsonProperty
    private int index;
    @JsonProperty
    private A modelSpecificResults;

    public OptimizationResult(Candidate<C> candidate, M result, Double score, int index, A modelSpecificResults) {
        this.candidate = candidate;
        this.result = result;
        this.score = score;
        this.index = index;
        this.modelSpecificResults = modelSpecificResults;
    }


}
