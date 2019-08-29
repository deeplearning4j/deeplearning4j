/*
 * Copyright (c) 2015-2019 Skymind, Inc.
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
 */

package org.nd4j.autodiff.listeners;

import org.nd4j.shade.guava.collect.Sets;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * Specifies a Listener's required variables for each operation.
 * Used to ensure those variables end up in the minimum required subgraph calculated by {@link org.nd4j.autodiff.samediff.internal.InferenceSession}.
 * Otherwise, if the variables weren't required by a loss variable, they would not be calculated.
 * <p>
 * Any variables in here are guaranteed to have {@link Listener#activationAvailable(SameDiff, At, MultiDataSet, SameDiffOp, String, INDArray)} called for them.
 */
@RequiredArgsConstructor
@Getter
public class ListenerVariables {

    public static ListenerVariables empty() {
        return ListenerVariables.builder().build();
    }

    @NonNull
    private Set<String> trainingVariables;
    @NonNull
    private Set<String> validationVariables;
    @NonNull
    private Set<String> evaluationVariables;
    @NonNull
    private Set<String> inferenceVariables;

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Get required training variables
     */
    public Set<String> trainingVariables() {
        return trainingVariables;
    }

    /**
     * Get required validation variables
     */
    public Set<String> validationVariables() {
        return validationVariables;
    }

    /**
     * Get required evaluation variables
     */
    public Set<String> evaluationVariables() {
        return evaluationVariables;
    }

    /**
     * Get required inference variables
     */
    public Set<String> inferenceVariables() {
        return inferenceVariables;
    }

    /**
     * Get required variables for specified op
     */
    public Set<String> requiredVariables(Operation op) {
        switch (op) {
            case TRAINING:
                return trainingVariables;
            case TRAINING_VALIDATION:
                return validationVariables;
            case INFERENCE:
                return inferenceVariables;
            case EVALUATION:
                return evaluationVariables;
        }
        throw new IllegalArgumentException("Unknown operation " + op);
    }

    private ListenerVariables() {

    }

    /**
     * Return a new ListenerVariables that contains the variables of this ListenerVariables and of other
     */
    public ListenerVariables merge(ListenerVariables other) {
        return new ListenerVariables(
                Sets.newHashSet(Sets.union(trainingVariables, other.trainingVariables)),
                Sets.newHashSet(Sets.union(validationVariables, other.validationVariables)),
                Sets.newHashSet(Sets.union(evaluationVariables, other.evaluationVariables)),
                Sets.newHashSet(Sets.union(inferenceVariables, other.inferenceVariables)));
    }

    @NoArgsConstructor
    @Getter
    @Setter
    public static class Builder {
        @NonNull
        private Set<String> trainingVariables = new HashSet<>();
        @NonNull
        private Set<String> validationVariables = new HashSet<>();
        @NonNull
        private Set<String> evaluationVariables = new HashSet<>();
        @NonNull
        private Set<String> inferenceVariables = new HashSet<>();

        /**
         * Add required variables for the specified op
         *
         * @param op The op to require the variable for
         */
        public Builder requireVariables(@NonNull Operation op, @NonNull String... variables) {
            switch (op) {
                case TRAINING:
                    trainingVariables.addAll(Arrays.asList(variables));
                    break;
                case TRAINING_VALIDATION:
                    validationVariables.addAll(Arrays.asList(variables));
                    break;
                case INFERENCE:
                    inferenceVariables.addAll(Arrays.asList(variables));
                    break;
                case EVALUATION:
                    evaluationVariables.addAll(Arrays.asList(variables));
                    break;
            }

            return this;
        }

        /**
         * Add required variables for the specified op
         *
         * @param op The op to require the variable for
         */
        public Builder requireVariables(@NonNull Operation op, @NonNull SDVariable... variables) {
            String[] names = new String[variables.length];

            for (int i = 0; i < variables.length; i++)
                names[i] = variables[i].getVarName();

            return requireVariables(op, names);
        }

        /**
         * Add required variables for training
         */
        public Builder trainingVariables(@NonNull String... variables) {
            return requireVariables(Operation.TRAINING, variables);
        }

        /**
         * Add required variables for training
         */
        public Builder trainingVariables(@NonNull SDVariable... variables) {
            return requireVariables(Operation.TRAINING, variables);
        }

        /**
         * Add required variables for validation
         */
        public Builder validationVariables(@NonNull String... variables) {
            return requireVariables(Operation.TRAINING_VALIDATION, variables);
        }

        /**
         * Add required variables for validation
         */
        public Builder validationVariables(@NonNull SDVariable... variables) {
            return requireVariables(Operation.TRAINING_VALIDATION, variables);
        }

        /**
         * Add required variables for inference
         */
        public Builder inferenceVariables(@NonNull String... variables) {
            return requireVariables(Operation.INFERENCE, variables);
        }

        /**
         * Add required variables for inference
         */
        public Builder inferenceVariables(@NonNull SDVariable... variables) {
            return requireVariables(Operation.INFERENCE, variables);
        }

        /**
         * Add required variables for evaluation
         */
        public Builder evaluationVariables(@NonNull String... variables) {
            return requireVariables(Operation.EVALUATION, variables);
        }

        /**
         * Add required variables for evaluation
         */
        public Builder evaluationVariables(@NonNull SDVariable... variables) {
            return requireVariables(Operation.EVALUATION, variables);
        }

        public ListenerVariables build() {
            return new ListenerVariables(trainingVariables, validationVariables, evaluationVariables, inferenceVariables);
        }
    }

}
