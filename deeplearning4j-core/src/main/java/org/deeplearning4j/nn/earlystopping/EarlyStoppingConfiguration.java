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

package org.deeplearning4j.nn.earlystopping;

import lombok.Data;
import org.deeplearning4j.nn.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.nn.earlystopping.termination.IterationTerminationCondition;

import java.util.ArrayList;
import java.util.List;

@Data
public class EarlyStoppingConfiguration {

    private EarlyStoppingModelSaver modelSaver;
    private List<EpochTerminationCondition> epochTerminationConditions;
    private List<IterationTerminationCondition> iterationTerminationConditions;
    private boolean saveLastModel;
    private int evaluateEveryNEpochs;

    private EarlyStoppingConfiguration( Builder builder ){
        this.modelSaver = builder.modelSaver;
        this.epochTerminationConditions = builder.epochTerminationConditions;
        this.iterationTerminationConditions = builder.iterationTerminationConditions;
        this.saveLastModel = builder.saveLastModel;
        this.evaluateEveryNEpochs = builder.evaluateEveryNEpochs;
    }


    public static class Builder {

        private EarlyStoppingModelSaver modelSaver;
        private List<EpochTerminationCondition> epochTerminationConditions = new ArrayList<>();
        private List<IterationTerminationCondition> iterationTerminationConditions = new ArrayList<>();
        private boolean saveLastModel = false;
        private int evaluateEveryNEpochs = 1;

        public Builder modelSaver( EarlyStoppingModelSaver modelSaver ){
            this.modelSaver = modelSaver;
            return this;
        }

        public Builder epochTerminationConditions(EpochTerminationCondition... terminationConditions){
            epochTerminationConditions.clear();
            for(EpochTerminationCondition c : terminationConditions) epochTerminationConditions.add(c);
            return this;
        }

        public Builder epochTerminationConditions(List<EpochTerminationCondition> terminationConditions){
            this.epochTerminationConditions = terminationConditions;
            return this;
        }

        /** Save the last model? If true: save the most recent model at each epoch, in addition to the best
         * model (whenever the best model improves). If false: only save the best model.
         */
        public Builder saveLastModel(boolean saveLastModel){
            this.saveLastModel = saveLastModel;
            return this;
        }

        /** How frequently should evaluations be conducted (in terms of epochs)? Defaults to every (1) epochs. */
        public Builder evaluateEveryNEpochs(int everyNEpochs){
            this.evaluateEveryNEpochs = everyNEpochs;
            return this;
        }

        public EarlyStoppingConfiguration build(){
            return new EarlyStoppingConfiguration(this);
        }

    }

}
