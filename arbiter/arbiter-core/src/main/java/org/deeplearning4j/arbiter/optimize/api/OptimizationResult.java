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

package org.deeplearning4j.arbiter.optimize.api;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.runner.CandidateInfo;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * An optimization result represents the results of an optimization run, including the canditate configuration, the
 * trained model, the score for that model, and index of the model
 *
 * @author Alex Black
 */
@Data
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonIgnoreProperties({"resultReference"})
public class OptimizationResult implements Serializable {
    @JsonProperty
    private Candidate candidate;
    @JsonProperty
    private Double score;
    @JsonProperty
    private int index;
    @JsonProperty
    private Object modelSpecificResults;
    @JsonProperty
    private CandidateInfo candidateInfo;
    private ResultReference resultReference;


    public OptimizationResult(Candidate candidate, Double score, int index, Object modelSpecificResults,
                    CandidateInfo candidateInfo, ResultReference resultReference) {
        this.candidate = candidate;
        this.score = score;
        this.index = index;
        this.modelSpecificResults = modelSpecificResults;
        this.candidateInfo = candidateInfo;
        this.resultReference = resultReference;
    }
}
