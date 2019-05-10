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

package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/** ScoreCalculator interface is used to calculate a score for a neural network.
 * For example, the loss function, test set accuracy, F1, or some other (possibly custom) metric.
 * @param <T> Type of model. For example, {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or {@link org.deeplearning4j.nn.graph.ComputationGraph}
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonSubTypes(value = {
                @JsonSubTypes.Type(value = DataSetLossCalculator.class, name = "BestScoreEpochTerminationCondition"),
                @JsonSubTypes.Type(value = DataSetLossCalculatorCG.class, name = "MaxEpochsTerminationCondition"),

})
public interface ScoreCalculator<T extends Model> extends Serializable {

    /** Calculate the score for the given MultiLayerNetwork */
    double calculateScore(T network);

    /**
     * @return If true: the score should be minimized. If false: the score should be maximized.
     */
    boolean minimizeScore();
}
