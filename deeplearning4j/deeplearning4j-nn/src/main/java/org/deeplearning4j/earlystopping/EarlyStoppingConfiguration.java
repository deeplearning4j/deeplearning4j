/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.earlystopping;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.IterationTerminationCondition;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.common.function.Supplier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Data
@NoArgsConstructor
public class EarlyStoppingConfiguration<T extends Model> implements Serializable {

    private EarlyStoppingModelSaver<T> modelSaver;
    private List<EpochTerminationCondition> epochTerminationConditions;
    private List<IterationTerminationCondition> iterationTerminationConditions;
    private boolean saveLastModel;
    private int evaluateEveryNEpochs;
    private ScoreCalculator<T> scoreCalculator;
    private Supplier<ScoreCalculator> scoreCalculatorSupplier;

    private EarlyStoppingConfiguration(Builder<T> builder) {
        this.modelSaver = builder.modelSaver;
        this.epochTerminationConditions = builder.epochTerminationConditions;
        this.iterationTerminationConditions = builder.iterationTerminationConditions;
        this.saveLastModel = builder.saveLastModel;
        this.evaluateEveryNEpochs = builder.evaluateEveryNEpochs;
        this.scoreCalculator = builder.scoreCalculator;
        this.scoreCalculatorSupplier = builder.scoreCalculatorSupplier;
    }

    public ScoreCalculator<T> getScoreCalculator(){
        if(scoreCalculatorSupplier != null){
            return scoreCalculatorSupplier.get();
        }
        return scoreCalculator;
    }


    public void validate() {
        if(scoreCalculator == null && scoreCalculatorSupplier == null) {
            throw new DL4JInvalidConfigException("A score calculator or score calculator supplier must be defined.");
        }

        if(modelSaver == null) {
            throw new DL4JInvalidConfigException("A model saver must be defined");
        }

        boolean hasTermination = false;
        if(iterationTerminationConditions != null && !iterationTerminationConditions.isEmpty()) {
            hasTermination = true;
        }

        else if(epochTerminationConditions != null && !epochTerminationConditions.isEmpty()) {
            hasTermination = true;
        }

        if(!hasTermination) {
            throw new DL4JInvalidConfigException("No termination conditions defined.");
        }
    }


    public static class Builder<T extends Model> {

        private EarlyStoppingModelSaver<T> modelSaver = new InMemoryModelSaver<>();
        private List<EpochTerminationCondition> epochTerminationConditions = new ArrayList<>();
        private List<IterationTerminationCondition> iterationTerminationConditions = new ArrayList<>();
        private boolean saveLastModel = false;
        private int evaluateEveryNEpochs = 1;
        private ScoreCalculator<T> scoreCalculator;
        private Supplier<ScoreCalculator> scoreCalculatorSupplier;


        /** How should models be saved? (Default: in memory)*/
        public Builder<T> modelSaver(EarlyStoppingModelSaver<T> modelSaver) {
            this.modelSaver = modelSaver;
            return this;
        }

        /** Termination conditions to be evaluated every N epochs, with N set by evaluateEveryNEpochs option */
        public Builder<T> epochTerminationConditions(EpochTerminationCondition... terminationConditions) {
            epochTerminationConditions.clear();
            Collections.addAll(epochTerminationConditions, terminationConditions);
            return this;
        }

        /** Termination conditions to be evaluated every N epochs, with N set by evaluateEveryNEpochs option */
        public Builder<T> epochTerminationConditions(List<EpochTerminationCondition> terminationConditions) {
            this.epochTerminationConditions = terminationConditions;
            return this;
        }

        /** Termination conditions to be evaluated every iteration (minibatch)*/
        public Builder<T> iterationTerminationConditions(IterationTerminationCondition... terminationConditions) {
            iterationTerminationConditions.clear();
            Collections.addAll(iterationTerminationConditions, terminationConditions);
            return this;
        }

        /** Save the last model? If true: save the most recent model at each epoch, in addition to the best
         * model (whenever the best model improves). If false: only save the best model. Default: false
         * Useful for example if you might want to continue training after a max-time terminatino condition
         * occurs.
         */
        public Builder<T> saveLastModel(boolean saveLastModel) {
            this.saveLastModel = saveLastModel;
            return this;
        }

        /** How frequently should evaluations be conducted (in terms of epochs)? Defaults to every (1) epochs. */
        public Builder<T> evaluateEveryNEpochs(int everyNEpochs) {
            this.evaluateEveryNEpochs = everyNEpochs;
            return this;
        }

        /** Score calculator. Used to calculate a score (such as loss function on a test set), every N epochs,
         * where N is set by {@link #evaluateEveryNEpochs}
         */
        public Builder<T> scoreCalculator(ScoreCalculator scoreCalculator) {
            this.scoreCalculator = scoreCalculator;
            return this;
        }

        /** Score calculator. Used to calculate a score (such as loss function on a test set), every N epochs,
         * where N is set by {@link #evaluateEveryNEpochs}
         */
        public Builder<T> scoreCalculator(Supplier<ScoreCalculator> scoreCalculatorSupplier){
            this.scoreCalculatorSupplier = scoreCalculatorSupplier;
            return this;
        }

        /** Create the early stopping configuration */
        public EarlyStoppingConfiguration<T> build() {
            return new EarlyStoppingConfiguration<>(this);
        }

    }
}
