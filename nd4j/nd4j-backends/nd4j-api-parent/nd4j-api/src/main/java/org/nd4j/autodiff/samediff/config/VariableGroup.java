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

package org.nd4j.autodiff.samediff.config;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Defines a group of variables that share training settings.
 * Used for discriminative fine-tuning where different parts of the network
 * have different learning rates.
 * <p>
 * Variable groups are matched by regex patterns against variable names.
 * A variable belongs to the first group whose pattern matches its name.
 * <p>
 * Example usage:
 * <pre>
 * VariableGroup earlyLayers = new VariableGroup("early", 0.1, Arrays.asList("encoder.layer.[0-5].*"));
 * VariableGroup lateLayers = new VariableGroup("late", 1.0, Arrays.asList("encoder.layer.[6-11].*"));
 * </pre>
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class VariableGroup implements Serializable {
    private static final long serialVersionUID = 1L;

    private String name;
    private double learningRateMultiplier = 1.0;
    private IUpdater updater;
    private List<Regularization> regularization;
    private List<String> variablePatterns;

    @JsonIgnore
    private transient List<Pattern> compiledPatterns;

    /**
     * Create a variable group with a learning rate multiplier.
     *
     * @param name         Group name for identification
     * @param lrMultiplier Learning rate multiplier relative to base LR
     * @param patterns     Regex patterns to match variable names
     */
    public VariableGroup(String name, double lrMultiplier, List<String> patterns) {
        this.name = name;
        this.learningRateMultiplier = lrMultiplier;
        this.variablePatterns = patterns;
    }

    /**
     * Create a variable group with a learning rate multiplier.
     *
     * @param name         Group name for identification
     * @param lrMultiplier Learning rate multiplier relative to base LR
     * @param patterns     Regex patterns to match variable names
     */
    public VariableGroup(String name, double lrMultiplier, String... patterns) {
        this(name, lrMultiplier, Arrays.asList(patterns));
    }

    /**
     * Create a variable group with a specific updater override.
     *
     * @param name     Group name for identification
     * @param updater  Updater to use for this group (overrides default)
     * @param patterns Regex patterns to match variable names
     */
    public VariableGroup(String name, IUpdater updater, List<String> patterns) {
        this.name = name;
        this.updater = updater;
        this.learningRateMultiplier = 1.0;
        this.variablePatterns = patterns;
    }

    /**
     * Check if a variable name matches this group.
     * Patterns are compiled and cached on first use.
     *
     * @param variableName The variable name to check
     * @return true if the variable matches any pattern in this group
     */
    public boolean matches(String variableName) {
        if (variablePatterns == null || variablePatterns.isEmpty()) {
            return false;
        }

        if (compiledPatterns == null) {
            compiledPatterns = variablePatterns.stream()
                    .map(Pattern::compile)
                    .collect(Collectors.toList());
        }

        for (Pattern p : compiledPatterns) {
            if (p.matcher(variableName).matches()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Check if this group has a custom updater override.
     *
     * @return true if a custom updater is set
     */
    public boolean hasUpdater() {
        return updater != null;
    }

    /**
     * Check if this group has custom regularization.
     *
     * @return true if custom regularization is set
     */
    public boolean hasRegularization() {
        return regularization != null && !regularization.isEmpty();
    }

    /**
     * Create a builder for VariableGroup.
     *
     * @return A new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for VariableGroup.
     */
    public static class Builder {
        private String name;
        private double learningRateMultiplier = 1.0;
        private IUpdater updater;
        private List<Regularization> regularization = new ArrayList<>();
        private List<String> variablePatterns = new ArrayList<>();

        /**
         * Set the group name.
         */
        public Builder name(String name) {
            this.name = name;
            return this;
        }

        /**
         * Set the learning rate multiplier relative to base learning rate.
         *
         * @param multiplier Multiplier value (e.g., 0.1 for 10% of base LR)
         */
        public Builder learningRateMultiplier(double multiplier) {
            this.learningRateMultiplier = multiplier;
            return this;
        }

        /**
         * Set a custom updater for this group (overrides the default updater).
         */
        public Builder updater(IUpdater updater) {
            this.updater = updater;
            return this;
        }

        /**
         * Set custom regularization for this group.
         */
        public Builder regularization(List<Regularization> regularization) {
            this.regularization = regularization;
            return this;
        }

        /**
         * Add regularization to this group.
         */
        public Builder addRegularization(Regularization... regularizations) {
            this.regularization.addAll(Arrays.asList(regularizations));
            return this;
        }

        /**
         * Set variable patterns to match.
         */
        public Builder variablePatterns(List<String> patterns) {
            this.variablePatterns = patterns;
            return this;
        }

        /**
         * Add variable patterns to match.
         */
        public Builder addVariablePatterns(String... patterns) {
            this.variablePatterns.addAll(Arrays.asList(patterns));
            return this;
        }

        /**
         * Build the VariableGroup.
         */
        public VariableGroup build() {
            return new VariableGroup(name, learningRateMultiplier, updater,
                    regularization.isEmpty() ? null : regularization, variablePatterns, null);
        }
    }
}
