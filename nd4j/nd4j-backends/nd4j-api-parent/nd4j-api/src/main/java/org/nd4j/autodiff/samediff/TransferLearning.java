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

package org.nd4j.autodiff.samediff;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.*;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.autodiff.samediff.peft.PeftModelFactory;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.*;

/**
 * Transfer learning / fine-tuning API for SameDiff.
 * Provides a builder pattern to configure transfer learning scenarios including:
 * <ul>
 *     <li>Freezing variables (converting to constants)</li>
 *     <li>Per-variable learning rates (discriminative fine-tuning)</li>
 *     <li>Weight reinitialization</li>
 *     <li>Feature extraction (efficient training with cached features)</li>
 *     <li><b>PEFT methods</b>: LoRA, Prompt Tuning, Prefix Tuning, Adapters</li>
 * </ul>
 *
 * <p>Example usage - Simple freezing:</p>
 * <pre>
 * SameDiff fineTuned = new TransferLearning.Builder(pretrained)
 *     .freezePrefix("encoder.layer.0")
 *     .freezePrefix("encoder.layer.1")
 *     .build();
 * </pre>
 *
 * <p>Example usage - Discriminative fine-tuning:</p>
 * <pre>
 * FineTuneConfiguration ftConfig = FineTuneConfiguration.builder()
 *     .updater(new Adam(1e-4))
 *     .variableGroup("early", 0.1, "encoder\\.layer\\.[0-5]\\..*")
 *     .variableGroup("late", 1.0, "encoder\\.layer\\.[6-11]\\..*")
 *     .variableGroup("head", 2.0, "classifier\\..*")
 *     .build();
 *
 * SameDiff fineTuned = new TransferLearning.Builder(pretrained)
 *     .fineTuneConfiguration(ftConfig)
 *     .build();
 * </pre>
 *
 * <p>Example usage - LoRA (Parameter-Efficient Fine-Tuning):</p>
 * <pre>
 * // Using PEFT builder methods
 * PeftModel loraModel = new TransferLearning.Builder(pretrained)
 *     .lora(16, 32, Arrays.asList("query", "key", "value"))
 *     .buildPeft();
 *
 * // Or use the PeftModelFactory directly
 * PeftModel loraModel = PeftModelFactory.createLoraModel(pretrained, 16);
 * loraModel.printTrainableParameters();
 * // Output: trainable params: 294,912 || all params: 110,000,000 || trainable%: 0.2681
 * </pre>
 *
 * @author Adam Gibson
 * @see FineTuneConfiguration
 * @see TransferLearningHelper
 * @see PeftModel
 * @see LoraConfig
 */
@Slf4j
public class TransferLearning {

    private TransferLearning() {
        // Utility class
    }

    // ========== Static PEFT Factory Methods ==========

    /**
     * Create a LoRA model from a pretrained model.
     * Uses default LoRA configuration (rank=16, alpha=32).
     *
     * @param baseModel     The pretrained base model
     * @param targetModules List of module names/patterns to apply LoRA to
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel loraModel(SameDiff baseModel, List<String> targetModules) {
        return PeftModelFactory.createLoraModel(baseModel, 16, 32, targetModules);
    }

    /**
     * Create a LoRA model with specified rank.
     *
     * @param baseModel     The pretrained base model
     * @param rank          The LoRA rank
     * @param targetModules List of module names/patterns to apply LoRA to
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel loraModel(SameDiff baseModel, int rank, List<String> targetModules) {
        return PeftModelFactory.createLoraModel(baseModel, rank, rank * 2, targetModules);
    }

    /**
     * Create a LoRA model with full configuration.
     *
     * @param baseModel The pretrained base model
     * @param config    The LoRA configuration
     * @return A PeftModel with LoRA applied
     */
    public static PeftModel loraModel(SameDiff baseModel, LoraConfig config) {
        return PeftModel.fromPretrained(baseModel, config);
    }

    /**
     * Create a Prompt Tuning model.
     *
     * @param baseModel    The pretrained base model
     * @param numTokens    Number of soft prompt tokens
     * @param embeddingDim Embedding dimension
     * @return A PeftModel with Prompt Tuning applied
     */
    public static PeftModel promptTuningModel(SameDiff baseModel, int numTokens, int embeddingDim) {
        return PeftModelFactory.createPromptTuningModel(baseModel, numTokens, embeddingDim);
    }

    /**
     * Create a Prefix Tuning model.
     *
     * @param baseModel  The pretrained base model
     * @param numTokens  Number of prefix tokens
     * @param numLayers  Number of transformer layers
     * @param hiddenSize Hidden size of the model
     * @return A PeftModel with Prefix Tuning applied
     */
    public static PeftModel prefixTuningModel(SameDiff baseModel, int numTokens,
                                               int numLayers, int hiddenSize) {
        return PeftModelFactory.createPrefixTuningModel(baseModel, numTokens, numLayers, hiddenSize);
    }

    /**
     * Create an Adapter model.
     *
     * @param baseModel   The pretrained base model
     * @param adapterSize Size of the adapter bottleneck
     * @param hiddenSize  Hidden size of the model
     * @param numLayers   Number of layers to add adapters to
     * @return A PeftModel with Adapters applied
     */
    public static PeftModel adapterModel(SameDiff baseModel, int adapterSize,
                                          int hiddenSize, int numLayers) {
        return PeftModelFactory.createAdapterModel(baseModel, adapterSize, hiddenSize, numLayers);
    }

    /**
     * Get a PEFT configuration builder for LoRA.
     * Convenience method that returns a pre-configured LoraConfig builder.
     *
     * @return A LoraConfig builder with sensible defaults
     */
    public static LoraConfig.LoraConfigBuilder<?, ?> loraConfigBuilder() {
        return PeftModelFactory.loraConfigBuilder();
    }

    /**
     * Builder class for transfer learning configuration.
     * Supports both traditional fine-tuning and PEFT (Parameter-Efficient Fine-Tuning) methods.
     */
    public static class Builder {
        private final SameDiff origModel;
        private FineTuneConfiguration fineTuneConfig;
        private PeftConfig peftConfig;
        private boolean copyModel = true;
        private Set<String> frozenVariables = new HashSet<>();
        private Set<String> unfrozenVariables = new HashSet<>();
        private Map<String, INDArray> reinitializedVariables = new HashMap<>();
        private List<String> frozenPatterns = new ArrayList<>();
        private List<String> featureExtractorOutputs = new ArrayList<>();

        /**
         * Create a new transfer learning builder for the given model.
         *
         * @param origModel The original pretrained SameDiff model
         */
        public Builder(SameDiff origModel) {
            this.origModel = origModel;
        }

        // ========== PEFT Configuration Methods ==========

        /**
         * Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
         *
         * @param rank          The LoRA rank (r). Lower = fewer parameters, higher = more capacity
         * @param alpha         The LoRA alpha (scaling = alpha/rank)
         * @param targetModules List of module names/patterns to apply LoRA to
         * @return This builder for chaining
         */
        public Builder lora(int rank, int alpha, List<String> targetModules) {
            this.peftConfig = LoraConfig.builder()
                .r(rank)
                .loraAlpha(alpha)
                .loraDropout(0.05)
                .targetModules(targetModules)
                .taskType(TaskType.GENERAL)
                .build();
            return this;
        }

        /**
         * Configure LoRA with default settings (rank=16, alpha=32).
         *
         * @param targetModules List of module names/patterns to apply LoRA to
         * @return This builder for chaining
         */
        public Builder lora(List<String> targetModules) {
            return lora(16, 32, targetModules);
        }

        /**
         * Configure LoRA with a LoraConfig.
         *
         * @param config The LoRA configuration
         * @return This builder for chaining
         */
        public Builder lora(LoraConfig config) {
            this.peftConfig = config;
            return this;
        }

        /**
         * Configure Prompt Tuning for parameter-efficient fine-tuning.
         *
         * @param numTokens    Number of soft prompt tokens to add
         * @param embeddingDim Embedding dimension of the model
         * @return This builder for chaining
         */
        public Builder promptTuning(int numTokens, int embeddingDim) {
            this.peftConfig = PromptTuningConfig.builder()
                .numVirtualTokens(numTokens)
                .tokenEmbeddingDim(embeddingDim)
                .promptTuningInit(PromptTuningConfig.PromptTuningInit.RANDOM)
                .taskType(TaskType.GENERAL)
                .build();
            return this;
        }

        /**
         * Configure Prompt Tuning with a PromptTuningConfig.
         *
         * @param config The Prompt Tuning configuration
         * @return This builder for chaining
         */
        public Builder promptTuning(PromptTuningConfig config) {
            this.peftConfig = config;
            return this;
        }

        /**
         * Configure Prefix Tuning for parameter-efficient fine-tuning.
         *
         * @param numTokens  Number of prefix tokens
         * @param numLayers  Number of transformer layers
         * @param hiddenSize Hidden size of the model
         * @return This builder for chaining
         */
        public Builder prefixTuning(int numTokens, int numLayers, int hiddenSize) {
            this.peftConfig = PrefixTuningConfig.builder()
                .numVirtualTokens(numTokens)
                .numLayers(numLayers)
                .hiddenSize(hiddenSize)
                .prefixProjection(true)
                .taskType(TaskType.GENERAL)
                .build();
            return this;
        }

        /**
         * Configure Prefix Tuning with a PrefixTuningConfig.
         *
         * @param config The Prefix Tuning configuration
         * @return This builder for chaining
         */
        public Builder prefixTuning(PrefixTuningConfig config) {
            this.peftConfig = config;
            return this;
        }

        /**
         * Configure Adapter layers for parameter-efficient fine-tuning.
         *
         * @param adapterSize Size of the adapter bottleneck
         * @param hiddenSize  Hidden size of the model
         * @param numLayers   Number of layers to add adapters to
         * @return This builder for chaining
         */
        public Builder adapters(int adapterSize, int hiddenSize, int numLayers) {
            this.peftConfig = AdapterConfig.builder()
                .adapterSize(adapterSize)
                .hiddenSize(hiddenSize)
                .numLayers(numLayers)
                .adapterActivation("relu")
                .taskType(TaskType.GENERAL)
                .build();
            return this;
        }

        /**
         * Configure Adapters with an AdapterConfig.
         *
         * @param config The Adapter configuration
         * @return This builder for chaining
         */
        public Builder adapters(AdapterConfig config) {
            this.peftConfig = config;
            return this;
        }

        /**
         * Set a custom PEFT configuration.
         *
         * @param config The PEFT configuration
         * @return This builder for chaining
         */
        public Builder peftConfig(PeftConfig config) {
            this.peftConfig = config;
            return this;
        }

        /**
         * Build a PeftModel instead of a regular SameDiff model.
         * Use this when you've configured LoRA, Prompt Tuning, or other PEFT methods.
         *
         * @return A PeftModel wrapping the base model with PEFT applied
         */
        public PeftModel buildPeft() {
            Preconditions.checkNotNull(peftConfig,
                "PEFT configuration is required. Use lora(), promptTuning(), or similar methods first.");
            return PeftModel.fromPretrained(origModel, peftConfig);
        }

        /**
         * Set the fine-tune configuration for per-variable settings.
         * This provides discriminative learning rates and advanced freezing options.
         *
         * @param config The fine-tune configuration
         * @return This builder for chaining
         */
        public Builder fineTuneConfiguration(FineTuneConfiguration config) {
            this.fineTuneConfig = config;
            return this;
        }

        /**
         * Set the feature extractor output(s). All variables upstream of these outputs
         * will be frozen, and only the downstream variables will be trained.
         * This is useful for caching feature representations.
         *
         * @param outputVariableNames Names of the output variables for feature extraction
         * @return This builder for chaining
         */
        public Builder setFeatureExtractor(String... outputVariableNames) {
            this.featureExtractorOutputs.addAll(Arrays.asList(outputVariableNames));
            return this;
        }

        /**
         * Freeze specific variables by exact name.
         * Frozen variables will not be updated during training.
         *
         * @param variableNames Names of variables to freeze
         * @return This builder for chaining
         */
        public Builder freezeVariables(String... variableNames) {
            Collections.addAll(this.frozenVariables, variableNames);
            return this;
        }

        /**
         * Freeze variables matching a regex pattern.
         *
         * @param regex Regex pattern to match variable names
         * @return This builder for chaining
         */
        public Builder freezeMatching(String regex) {
            this.frozenPatterns.add(regex);
            return this;
        }

        /**
         * Freeze all variables whose names start with the given prefix.
         *
         * @param prefix Prefix to match
         * @return This builder for chaining
         */
        public Builder freezePrefix(String prefix) {
            return freezeMatching("^" + java.util.regex.Pattern.quote(prefix) + ".*");
        }

        /**
         * Explicitly unfreeze variables (override FineTuneConfiguration freezing).
         *
         * @param variableNames Names of variables to unfreeze
         * @return This builder for chaining
         */
        public Builder unfreezeVariables(String... variableNames) {
            Collections.addAll(this.unfrozenVariables, variableNames);
            return this;
        }

        /**
         * Reinitialize a variable with new values.
         *
         * @param variableName Name of the variable to reinitialize
         * @param newValue     New value for the variable
         * @return This builder for chaining
         */
        public Builder reinitialize(String variableName, INDArray newValue) {
            this.reinitializedVariables.put(variableName, newValue);
            return this;
        }

        /**
         * Set whether to copy the model before making modifications.
         * If true (default), the original model is not modified.
         * If false, modifications are made in-place on the original model.
         *
         * @param copy Whether to copy the model
         * @return This builder for chaining
         */
        public Builder copyModel(boolean copy) {
            this.copyModel = copy;
            return this;
        }

        /**
         * Build the fine-tuned SameDiff model.
         *
         * @return The configured SameDiff model ready for fine-tuning
         */
        public SameDiff build() {
            SameDiff sd = copyModel ? origModel.dup() : origModel;

            // Collect all variables to freeze
            Set<String> toFreeze = new HashSet<>(frozenVariables);

            // Add variables from FineTuneConfiguration
            if (fineTuneConfig != null) {
                if (fineTuneConfig.getFrozenVariables() != null) {
                    toFreeze.addAll(fineTuneConfig.getFrozenVariables());
                }
                // Check pattern-frozen variables
                for (String varName : sd.variableMap().keySet()) {
                    if (fineTuneConfig.isFrozen(varName)) {
                        toFreeze.add(varName);
                    }
                }
            }

            // Add variables from frozen patterns
            for (String pattern : frozenPatterns) {
                java.util.regex.Pattern p = java.util.regex.Pattern.compile(pattern);
                for (String varName : sd.variableMap().keySet()) {
                    if (p.matcher(varName).matches()) {
                        toFreeze.add(varName);
                    }
                }
            }

            // Add variables upstream of feature extractor outputs
            if (!featureExtractorOutputs.isEmpty()) {
                Set<String> upstream = findUpstreamVariables(sd, featureExtractorOutputs.toArray(new String[0]));
                toFreeze.addAll(upstream);
            }

            // Remove explicitly unfrozen variables
            toFreeze.removeAll(unfrozenVariables);

            // Apply freezing
            if (!toFreeze.isEmpty()) {
                sd.freezeVariables(toFreeze);
            }

            // Apply weight reinitializations
            for (Map.Entry<String, INDArray> entry : reinitializedVariables.entrySet()) {
                SDVariable var = sd.getVariable(entry.getKey());
                Preconditions.checkNotNull(var, "Variable not found for reinitialization: %s", entry.getKey());
                var.setArray(entry.getValue());
            }

            // Create and set training config
            if (fineTuneConfig != null) {
                FineTuneTrainingConfig trainingConfig = new FineTuneTrainingConfig(fineTuneConfig, toFreeze);
                sd.setTrainingConfig(trainingConfig);
            }

            return sd;
        }

        /**
         * Find all trainable variables upstream of the given output variable(s).
         * This is used for feature extraction to determine which variables to freeze.
         *
         * @param sd          The SameDiff instance
         * @param outputNames Names of output variables
         * @return Set of upstream trainable variable names
         */
        public static Set<String> findUpstreamVariables(SameDiff sd, String... outputNames) {
            Set<String> upstream = new HashSet<>();
            Set<String> visited = new HashSet<>();
            Queue<String> toVisit = new LinkedList<>(Arrays.asList(outputNames));

            while (!toVisit.isEmpty()) {
                String current = toVisit.poll();
                if (visited.contains(current)) continue;
                visited.add(current);

                Variable var = sd.getVariables().get(current);
                if (var == null) continue;

                // If this is a trainable variable, add it
                SDVariable sdVar = var.getVariable();
                if (sdVar != null && sdVar.getVariableType() == VariableType.VARIABLE) {
                    upstream.add(current);
                }

                // Find the op that outputs this variable
                String outputOfOp = var.getOutputOfOp();
                if (outputOfOp != null) {
                    SameDiffOp op = sd.getOps().get(outputOfOp);
                    if (op != null && op.getInputsToOp() != null) {
                        toVisit.addAll(op.getInputsToOp());
                    }
                }
            }

            return upstream;
        }
    }

    /**
     * TrainingConfig implementation that supports per-variable settings.
     * This wraps a FineTuneConfiguration and provides the TrainingConfig interface
     * expected by SameDiff training.
     */
    @Getter
    public static class FineTuneTrainingConfig extends TrainingConfig {
        private final FineTuneConfiguration fineTuneConfiguration;
        private final Set<String> frozenVariables;

        public FineTuneTrainingConfig(FineTuneConfiguration config, Set<String> frozenVariables) {
            this.fineTuneConfiguration = config;
            this.frozenVariables = frozenVariables != null ? new HashSet<>(frozenVariables) : new HashSet<>();

            // Set base config from FineTuneConfiguration
            if (config.getDefaultUpdater() != null) {
                setUpdater(config.getDefaultUpdater());
            }
            if (config.getDefaultRegularization() != null) {
                setRegularization(config.getDefaultRegularization());
            }
            if (config.getMinimize() != null) {
                setMinimize(config.getMinimize());
            }
        }

        @Override
        public boolean hasPerVariableConfig() {
            return true;
        }

        @Override
        public IUpdater getUpdaterForVariable(String variableName) {
            if (fineTuneConfiguration != null) {
                IUpdater updater = fineTuneConfiguration.getUpdaterForVariable(variableName);
                if (updater != null) {
                    return updater;
                }
            }
            return getUpdater();
        }

        @Override
        public List<Regularization> getRegularizationForVariable(String variableName) {
            if (fineTuneConfiguration != null) {
                List<Regularization> reg = fineTuneConfiguration.getRegularizationForVariable(variableName);
                if (reg != null) {
                    return reg;
                }
            }
            return getRegularization();
        }

        @Override
        public boolean isTrainable(String variableName) {
            if (frozenVariables != null && frozenVariables.contains(variableName)) {
                return false;
            }
            if (fineTuneConfiguration != null && fineTuneConfiguration.isFrozen(variableName)) {
                return false;
            }
            return true;
        }
    }
}
