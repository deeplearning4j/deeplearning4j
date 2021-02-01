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

package org.nd4j.autodiff.listeners;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;

/**
 * A class to allow Listeners to define what evaluations they need to run during training<br>
 * <p>
 * Usage example - does classification ({@link org.nd4j.evaluation.classification.Evaluation}) evaluation on
 * the training set (as training proceeds) and also Evaluation/ROCMultiClass evaluation on the test/validation set.
 * Assumes that the output predictions are called "softmax" and the first DataSet/MultiDataSet labels are those corresponding
 * to the "softmax" node
 * <pre>{@code
 * ListenerEvaluations.builder()
 *     //trainEvaluations: on the training data (in-line, as training proceeds through the epoch)
 *     .trainEvaluation("softmax", 0, new Evaluation(), new ROCMultiClass())
 *     //validationEvaluation: on the test/validation data, at the end of each epoch
 *     .validationEvaluation("softmax", 0, new Evaluation(), new ROCMultiClass())
 *     .build();
 * }</pre>
 */
@Getter
public class ListenerEvaluations {
    private Map<String, List<IEvaluation>> trainEvaluations;
    private Map<String, Integer> trainEvaluationLabels;

    private Map<String, List<IEvaluation>> validationEvaluations;
    private Map<String, Integer> validationEvaluationLabels;

    public ListenerEvaluations(Map<String, List<IEvaluation>> trainEvaluations,
                               Map<String, Integer> trainEvaluationLabels, Map<String, List<IEvaluation>> validationEvaluations,
                               Map<String, Integer> validationEvaluationLabels) {
        this.trainEvaluations = trainEvaluations;
        this.trainEvaluationLabels = trainEvaluationLabels;
        this.validationEvaluations = validationEvaluations;
        this.validationEvaluationLabels = validationEvaluationLabels;

        Preconditions.checkArgument(trainEvaluations.keySet().equals(trainEvaluationLabels.keySet()),
                "Must specify a label index for each train evaluation.  Expected: %s, got: %s",
                trainEvaluations.keySet(), trainEvaluationLabels.keySet());

        Preconditions.checkArgument(validationEvaluations.keySet().equals(validationEvaluationLabels.keySet()),
                "Must specify a label index for each validation evaluation.  Expected: %s, got: %s",
                validationEvaluations.keySet(), validationEvaluationLabels.keySet());
    }

    private ListenerEvaluations() {

    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Get the requested training evaluations
     */
    public Map<String, List<IEvaluation>> trainEvaluations() {
        return trainEvaluations;
    }

    /**
     * Get the label indices for the requested training evaluations
     */
    public Map<String, Integer> trainEvaluationLabels() {
        return trainEvaluationLabels;
    }

    /**
     * Get the requested validation evaluations
     */
    public Map<String, List<IEvaluation>> validationEvaluations() {
        return validationEvaluations;
    }

    /**
     * Get the label indices for the requested validation evaluations
     */
    public Map<String, Integer> validationEvaluationLabels() {
        return validationEvaluationLabels;
    }

    /**
     * Get the required variables for these evaluations
     */
    public ListenerVariables requiredVariables() {
        return new ListenerVariables(trainEvaluations.keySet(), validationEvaluations.keySet(),
                new HashSet<String>(), new HashSet<String>());
    }

    /**
     * @return true if there are no requested evaluations
     */
    public boolean isEmpty() {
        return trainEvaluations.isEmpty() && validationEvaluations.isEmpty();
    }

    @NoArgsConstructor
    @Getter
    @Setter
    public static class Builder {
        private Map<String, List<IEvaluation>> trainEvaluations = new HashMap<>();
        private Map<String, Integer> trainEvaluationLabels = new HashMap<>();

        private Map<String, List<IEvaluation>> validationEvaluations = new HashMap<>();
        private Map<String, Integer> validationEvaluationLabels = new HashMap<>();

        private void addEvaluations(boolean validation, @NonNull Map<String, List<IEvaluation>> evaluationMap, @NonNull Map<String, Integer> labelMap,
                                    @NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations) {
            if (evaluationMap.containsKey(variableName) && labelMap.get(variableName) != labelIndex) {
                String s;

                if (validation) {
                    s = "This ListenerEvaluations.Builder already has validation evaluations for ";
                } else {
                    s = "This ListenerEvaluations.Builder already has train evaluations for ";
                }

                throw new IllegalArgumentException(s + "variable " +
                        variableName + " with label index " + labelIndex + ".  You can't add " +
                        " evaluations with a different label index.  Got label index " + labelIndex);
            }

            if (evaluationMap.containsKey(variableName)) {
                evaluationMap.get(variableName).addAll(Arrays.asList(evaluations));
            } else {
                evaluationMap.put(variableName, Arrays.asList(evaluations));
                labelMap.put(variableName, labelIndex);
            }
        }

        /**
         * Add requested training evaluations for a parm/variable
         *
         * @param variableName The variable to evaluate
         * @param labelIndex   The index of the label to evaluate against
         * @param evaluations  The evaluations to run
         */
        public Builder trainEvaluation(@NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations) {
            addEvaluations(false, this.trainEvaluations, this.trainEvaluationLabels, variableName,
                    labelIndex, evaluations);
            return this;
        }

        /**
         * Add requested training evaluations for a parm/variable
         *
         * @param variable    The variable to evaluate
         * @param labelIndex  The index of the label to evaluate against
         * @param evaluations The evaluations to run
         */
        public Builder trainEvaluation(@NonNull SDVariable variable, int labelIndex, @NonNull IEvaluation... evaluations) {
            return trainEvaluation(variable.name(), labelIndex, evaluations);
        }

        /**
         * Add requested validation evaluations for a parm/variable
         *
         * @param variableName The variable to evaluate
         * @param labelIndex   The index of the label to evaluate against
         * @param evaluations  The evaluations to run
         */
        public Builder validationEvaluation(@NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations) {
            addEvaluations(true, this.validationEvaluations, this.validationEvaluationLabels, variableName,
                    labelIndex, evaluations);
            return this;
        }

        /**
         * Add requested validation evaluations for a parm/variable
         *
         * @param variable    The variable to evaluate
         * @param labelIndex  The index of the label to evaluate against
         * @param evaluations The evaluations to run
         */
        public Builder validationEvaluation(@NonNull SDVariable variable, int labelIndex, @NonNull IEvaluation... evaluations) {
            return validationEvaluation(variable.name(), labelIndex, evaluations);
        }

        /**
         * Add requested evaluations for a parm/variable, for either training or validation
         *
         * @param validation   Whether to add these evaluations as validation or training
         * @param variableName The variable to evaluate
         * @param labelIndex   The index of the label to evaluate against
         * @param evaluations  The evaluations to run
         */
        public Builder addEvaluations(boolean validation, @NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations) {
            if (validation) {
                return validationEvaluation(variableName, labelIndex, evaluations);
            } else {
                return trainEvaluation(variableName, labelIndex, evaluations);
            }
        }

        public ListenerEvaluations build() {
            return new ListenerEvaluations(trainEvaluations, trainEvaluationLabels, validationEvaluations, validationEvaluationLabels);
        }
    }
}
