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
package org.deeplearning4j.arbiter.optimize.api.score;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.candidategenerator.GridSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.Map;

/**
 * ScoreFunction defines the objective of hyperparameter optimization.
 * Specifically, it is used to calculate a score for a given model, relative to the data set provided
 * in the configuration.
 *
 * @param <M> Type of model
 * @param <D> Type of data used
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
public interface ScoreFunction<M, D> extends Serializable {

    /**
     * Calculate and return the score, for the given model and data provider
     *
     * @param model          Model to score
     * @param dataProvider   Data provider - data to use
     * @param dataParameters Parameters for data
     * @return Calculated score
     */
    double score(M model, DataProvider<D> dataProvider, Map<String, Object> dataParameters);

    /**
     * Should this score function be minimized or maximized?
     *
     * @return true if score should be minimized, false if score should be maximized
     */
    boolean minimize();
}
