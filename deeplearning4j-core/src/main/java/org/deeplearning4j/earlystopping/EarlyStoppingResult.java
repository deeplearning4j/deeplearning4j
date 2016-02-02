/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.earlystopping;

import lombok.Data;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.Serializable;
import java.util.Map;

/** EarlyStoppingResult: contains the results of the early stopping training, such as:
 * - Why the training was terminated
 * - Score vs. epoch
 * - Epoch that the best model was found
 * - Score of the best model
 * - The best model (MultiLayerNetwork) itself
 * @param <T> Type of model. For example, {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork} or {@link org.deeplearning4j.nn.graph.ComputationGraph}
 * @author Alex Black
 */
@Data
public class EarlyStoppingResult<T extends Model> implements Serializable {
    public enum TerminationReason {Error, IterationTerminationCondition, EpochTerminationCondition};

    private TerminationReason terminationReason;
    private String terminationDetails;
    private Map<Integer,Double> scoreVsEpoch;
    private int bestModelEpoch;
    private double bestModelScore;
    private int totalEpochs;
    private T bestModel;

    public EarlyStoppingResult(TerminationReason terminationReason, String terminationDetails, Map<Integer,Double> scoreVsEpoch,
                               int bestModelEpoch, double bestModelScore, int totalEpochs, T bestModel ){
        this.terminationReason = terminationReason;
        this.terminationDetails = terminationDetails;
        this.scoreVsEpoch = scoreVsEpoch;
        this.bestModelEpoch = bestModelEpoch;
        this.bestModelScore = bestModelScore;
        this.totalEpochs = totalEpochs;
        this.bestModel = bestModel;
    }

    @Override
    public String toString(){
        return "EarlyStoppingResult(terminationReason=" + terminationReason + ",details="+terminationDetails +
                ",bestModelEpoch="+bestModelEpoch + ",bestModelScore="+bestModelScore + ",totalEpochs="+totalEpochs + ")";

    }

}
