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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Configuration for fine-tuning a SameDiff model.
 * Allows setting per-variable or per-group updaters, regularization, and freezing.
 * <p>
 * This class mirrors DL4J's FineTuneConfiguration for familiarity and provides
 * a flexible way to configure transfer learning scenarios.
 * <p>
 * Example usage:
 * <pre>
 * FineTuneConfiguration config = FineTuneConfiguration.builder()
 *     .updater(new Adam(1e-5))           // Default updater for all variables
 *     .freezePrefix("encoder.layer.0")   // Freeze first layer
 *     .variableGroup("head", 2.0, "classifier.*")  // 2x LR for classifier
 *     .l2(1e-5)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 * @see org.nd4j.autodiff.samediff.TransferLearning
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Slf4j
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
public class FineTuneConfiguration implements Serializable {

    /**
     * The default updater applied to all trainable variables that do not have a per-variable
     * or per-group override. If {@code null}, no update step is performed on those variables.
     */
    private IUpdater defaultUpdater;

    /**
     * The default regularization applied to all trainable variables. If {@code null} or empty,
     * no regularization is applied by default. Specific regularizers such as
     * {@link org.nd4j.linalg.learning.regularization.L1Regularization},
     * {@link org.nd4j.linalg.learning.regularization.L2Regularization}, or
     * {@link org.nd4j.linalg.learning.regularization.WeightDecay} can be added via the Builder.
     */
    private List<Regularization> defaultRegularization;

    /**
     * Whether to minimize ({@code true}) or maximize ({@code false}) the loss function.
     * Default: {@code true} (minimize).
     */
    private Boolean minimize;

    /**
     * Per-variable updater overrides, keyed by exact variable name.
     * When a variable name is present in this map its updater takes precedence over
     * any group or default updater. May be {@code null} if no per-variable overrides are set.
     */
    private Map<String, IUpdater> variableUpdaters;

    /**
     * Per-variable regularization overrides, keyed by exact variable name.
     * When a variable name is present in this map its regularization list takes precedence over
     * any group or default regularization. May be {@code null} if no per-variable overrides are set.
     */
    private Map<String, List<Regularization>> variableRegularization;

    /**
     * Named variable groups that can carry their own updater, learning-rate multiplier, or
     * regularization, applied to all variables whose names match the group's regex patterns.
     * Groups are evaluated in insertion order; the first matching group wins.
     * May be {@code null} if no groups are defined.
     */
    private Map<String, VariableGroup> variableGroups;

    /**
     * The set of exact variable names that are frozen (not updated during training).
     * Takes precedence over pattern-based freezing for exact matches.
     * May be {@code null} if no variables are frozen by exact name.
     */
    private Set<String> frozenVariables;

    /**
     * Regex pattern strings used to match variable names that should be frozen.
     * Patterns are compiled lazily into {@link #frozenPatterns} on first use.
     * May be {@code null} if no pattern-based freezing is configured.
     */
    private List<String> frozenPatternStrings;

    /**
     * Compiled versions of {@link #frozenPatternStrings}. This field is transient and
     * is not serialized; patterns are re-compiled from {@code frozenPatternStrings} on demand.
     */
    @JsonIgnore
    private transient List<Pattern> frozenPatterns;

    /**
     * Get the compiled frozen patterns.
     * Patterns are compiled lazily on first access.
     */
    @JsonIgnore
    public List<Pattern> getFrozenPatterns() {
        if (frozenPatterns == null && frozenPatternStrings != null) {
            frozenPatterns = new ArrayList<>();
            for (String pattern : frozenPatternStrings) {
                frozenPatterns.add(Pattern.compile(pattern));
            }
        }
        return frozenPatterns;
    }

    /**
     * Check if a variable is frozen.
     *
     * @param variableName The variable name to check
     * @return true if the variable should be frozen
     */
    public boolean isFrozen(String variableName) {
        if (frozenVariables != null && frozenVariables.contains(variableName)) {
            return true;
        }

        List<Pattern> patterns = getFrozenPatterns();
        if (patterns != null) {
            for (Pattern p : patterns) {
                if (p.matcher(variableName).matches()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Get the updater for a specific variable.
     * Resolution order:
     * 1. Explicit per-variable updater
     * 2. Variable group updater (with LR multiplier applied)
     * 3. Default updater
     *
     * @param variableName The variable name
     * @return The updater to use for this variable
     */
    public IUpdater getUpdaterForVariable(String variableName) {
        // 1. Check explicit per-variable updater
        if (variableUpdaters != null && variableUpdaters.containsKey(variableName)) {
            return variableUpdaters.get(variableName);
        }

        // 2. Check variable groups
        if (variableGroups != null) {
            for (VariableGroup group : variableGroups.values()) {
                if (group.matches(variableName)) {
                    if (group.hasUpdater()) {
                        return group.getUpdater();
                    } else if (group.getLearningRateMultiplier() != 1.0 && defaultUpdater != null) {
                        return createScaledUpdater(defaultUpdater, group.getLearningRateMultiplier());
                    }
                    break;
                }
            }
        }

        // 3. Return default
        return defaultUpdater;
    }

    /**
     * Get the regularization for a specific variable.
     *
     * @param variableName The variable name
     * @return The regularization list for this variable
     */
    public List<Regularization> getRegularizationForVariable(String variableName) {
        // Check explicit per-variable regularization
        if (variableRegularization != null && variableRegularization.containsKey(variableName)) {
            return variableRegularization.get(variableName);
        }

        // Check variable groups
        if (variableGroups != null) {
            for (VariableGroup group : variableGroups.values()) {
                if (group.matches(variableName) && group.hasRegularization()) {
                    return group.getRegularization();
                }
            }
        }

        return defaultRegularization;
    }

    /**
     * Create a copy of the updater with scaled learning rate.
     */
    private IUpdater createScaledUpdater(IUpdater base, double multiplier) {
        try {
            IUpdater scaled = base.clone();
            if (scaled.hasLearningRate()) {
                double baseLr = scaled.getLearningRate(0, 0);
                scaled.setLrAndSchedule(baseLr * multiplier, null);
            }
            return scaled;
        } catch (Exception e) {
            log.warn("Failed to clone updater for scaling, using original: {}", e.getMessage());
            return base;
        }
    }

    /**
     * Create a new builder for FineTuneConfiguration.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for FineTuneConfiguration.
     */
    public static class Builder {
        private IUpdater defaultUpdater;
        private List<Regularization> defaultRegularization = new ArrayList<>();
        private Boolean minimize = true;
        private Map<String, IUpdater> variableUpdaters = new HashMap<>();
        private Map<String, List<Regularization>> variableRegularization = new HashMap<>();
        private Map<String, VariableGroup> variableGroups = new LinkedHashMap<>();
        private Set<String> frozenVariables = new HashSet<>();
        private List<String> frozenPatternStrings = new ArrayList<>();

        /**
         * Set the default updater for all trainable variables.
         */
        public Builder updater(IUpdater updater) {
            this.defaultUpdater = updater;
            return this;
        }

        /**
         * Set a specific updater for a single variable.
         */
        public Builder updater(String variableName, IUpdater updater) {
            this.variableUpdaters.put(variableName, updater);
            return this;
        }

        /**
         * Set updater for multiple variables at once.
         */
        public Builder updater(IUpdater updater, String... variableNames) {
            for (String name : variableNames) {
                this.variableUpdaters.put(name, updater);
            }
            return this;
        }

        /**
         * Define a variable group with specific learning rate multiplier.
         *
         * @param groupName        Name for this group
         * @param lrMultiplier     Learning rate multiplier (e.g., 0.1 for 10% of base LR)
         * @param variablePatterns Regex patterns to match variable names
         */
        public Builder variableGroup(String groupName, double lrMultiplier, String... variablePatterns) {
            VariableGroup group = new VariableGroup(groupName, lrMultiplier, Arrays.asList(variablePatterns));
            this.variableGroups.put(groupName, group);
            return this;
        }

        /**
         * Define a variable group with a custom updater.
         *
         * @param groupName        Name for this group
         * @param updater          Updater to use for this group
         * @param variablePatterns Regex patterns to match variable names
         */
        public Builder variableGroup(String groupName, IUpdater updater, String... variablePatterns) {
            VariableGroup group = new VariableGroup(groupName, updater, Arrays.asList(variablePatterns));
            this.variableGroups.put(groupName, group);
            return this;
        }

        /**
         * Add a pre-built variable group.
         */
        public Builder variableGroup(VariableGroup group) {
            this.variableGroups.put(group.getName(), group);
            return this;
        }

        /**
         * Freeze specific variables by exact name.
         */
        public Builder freeze(String... variableNames) {
            Collections.addAll(this.frozenVariables, variableNames);
            return this;
        }

        /**
         * Freeze variables matching a regex pattern.
         */
        public Builder freezeMatching(String pattern) {
            this.frozenPatternStrings.add(pattern);
            return this;
        }

        /**
         * Freeze all variables whose names start with the given prefix.
         */
        public Builder freezePrefix(String prefix) {
            return freezeMatching("^" + Pattern.quote(prefix) + ".*");
        }

        /**
         * Freeze all variables whose names contain the given substring.
         */
        public Builder freezeContaining(String substring) {
            return freezeMatching(".*" + Pattern.quote(substring) + ".*");
        }

        /**
         * Set L1 regularization for all variables.
         */
        public Builder l1(double l1) {
            Preconditions.checkState(l1 >= 0, "L1 regularization coefficient must be >= 0. Got %s", l1);
            removeInstances(this.defaultRegularization, L1Regularization.class);
            if (l1 > 0) {
                this.defaultRegularization.add(new L1Regularization(l1));
            }
            return this;
        }

        /**
         * Set L1 regularization for a specific variable.
         */
        public Builder l1(String variableName, double l1) {
            Preconditions.checkState(l1 >= 0, "L1 regularization coefficient must be >= 0. Got %s", l1);
            List<Regularization> regs = variableRegularization.computeIfAbsent(variableName, k -> new ArrayList<>());
            removeInstances(regs, L1Regularization.class);
            if (l1 > 0) {
                regs.add(new L1Regularization(l1));
            }
            return this;
        }

        /**
         * Set L2 regularization for all variables.
         */
        public Builder l2(double l2) {
            Preconditions.checkState(l2 >= 0, "L2 regularization coefficient must be >= 0. Got %s", l2);
            removeInstances(this.defaultRegularization, L2Regularization.class);
            removeInstances(this.defaultRegularization, WeightDecay.class);
            if (l2 > 0) {
                this.defaultRegularization.add(new L2Regularization(l2));
            }
            return this;
        }

        /**
         * Set L2 regularization for a specific variable.
         */
        public Builder l2(String variableName, double l2) {
            Preconditions.checkState(l2 >= 0, "L2 regularization coefficient must be >= 0. Got %s", l2);
            List<Regularization> regs = variableRegularization.computeIfAbsent(variableName, k -> new ArrayList<>());
            removeInstances(regs, L2Regularization.class);
            removeInstances(regs, WeightDecay.class);
            if (l2 > 0) {
                regs.add(new L2Regularization(l2));
            }
            return this;
        }

        /**
         * Set weight decay for all variables.
         */
        public Builder weightDecay(double coefficient, boolean applyLR) {
            removeInstances(this.defaultRegularization, WeightDecay.class);
            removeInstances(this.defaultRegularization, L2Regularization.class);
            if (coefficient > 0) {
                this.defaultRegularization.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Set weight decay for a specific variable.
         */
        public Builder weightDecay(String variableName, double coefficient, boolean applyLR) {
            List<Regularization> regs = variableRegularization.computeIfAbsent(variableName, k -> new ArrayList<>());
            removeInstances(regs, WeightDecay.class);
            removeInstances(regs, L2Regularization.class);
            if (coefficient > 0) {
                regs.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Add regularization for all variables.
         */
        public Builder addRegularization(Regularization... regularizations) {
            Collections.addAll(this.defaultRegularization, regularizations);
            return this;
        }

        /**
         * Set whether to minimize (true) or maximize (false) the loss.
         */
        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        /**
         * Build the FineTuneConfiguration.
         */
        public FineTuneConfiguration build() {
            return new FineTuneConfiguration(
                    defaultUpdater,
                    defaultRegularization.isEmpty() ? null : defaultRegularization,
                    minimize,
                    variableUpdaters.isEmpty() ? null : variableUpdaters,
                    variableRegularization.isEmpty() ? null : variableRegularization,
                    variableGroups.isEmpty() ? null : variableGroups,
                    frozenVariables.isEmpty() ? null : frozenVariables,
                    frozenPatternStrings.isEmpty() ? null : frozenPatternStrings,
                    null  // frozenPatterns will be compiled lazily
            );
        }

        private static void removeInstances(List<?> list, Class<?> remove) {
            TrainingConfig.removeInstances(list, remove);
        }
    }
}
