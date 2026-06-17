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
import lombok.experimental.SuperBuilder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.List;

/**
 * Base configuration class for Parameter-Efficient Fine-Tuning (PEFT) methods.
 * <p>
 * PEFT methods allow fine-tuning large pretrained models with minimal parameter
 * updates while achieving performance comparable to full fine-tuning. This significantly
 * reduces computational and storage costs.
 * <p>
 * Based on the Hugging Face PEFT library architecture.
 *
 * <p>Example usage:</p>
 * <pre>
 * // Create a LoRA configuration
 * LoraConfig loraConfig = LoraConfig.builder()
 *     .r(16)
 *     .loraAlpha(32)
 *     .targetModules(Arrays.asList("query", "key", "value"))
 *     .loraDropout(0.1)
 *     .taskType(TaskType.CAUSAL_LM)
 *     .build();
 *
 * // Apply PEFT to a SameDiff model
 * PeftModel peftModel = PeftModel.fromPretrained(baseModel, loraConfig);
 *
 * // Train the PEFT model
 * peftModel.fit(trainingData, epochs);
 *
 * // Merge adapters back into base model
 * SameDiff mergedModel = peftModel.mergeAndUnload();
 * </pre>
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see PromptTuningConfig
 * @see PrefixTuningConfig
 * @see <a href="https://github.com/huggingface/peft">Hugging Face PEFT</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
@JsonSubTypes({
    @JsonSubTypes.Type(value = LoraConfig.class, name = "lora"),
    @JsonSubTypes.Type(value = QLoraConfig.class, name = "qlora"),
    @JsonSubTypes.Type(value = DoraConfig.class, name = "dora"),
    @JsonSubTypes.Type(value = AdaLoraConfig.class, name = "adalora"),
    @JsonSubTypes.Type(value = LohaConfig.class, name = "loha"),
    @JsonSubTypes.Type(value = LokrConfig.class, name = "lokr"),
    @JsonSubTypes.Type(value = PromptTuningConfig.class, name = "prompt_tuning"),
    @JsonSubTypes.Type(value = PrefixTuningConfig.class, name = "prefix_tuning"),
    @JsonSubTypes.Type(value = AdapterConfig.class, name = "adapters"),
    @JsonSubTypes.Type(value = IA3Config.class, name = "ia3"),
    @JsonSubTypes.Type(value = LoftQConfig.class, name = "loftq")
})
public abstract class PeftConfig implements Serializable {

    /**
     * The type of PEFT method.
     */
    protected PeftType peftType;

    /**
     * The task type for which the model is being fine-tuned.
     */
    protected TaskType taskType;

    /**
     * Whether to perform inference mode (adapters applied but not trainable).
     */
    protected boolean inferenceMode;

    /**
     * List of module names or patterns to target for adaptation.
     * For LoRA: weight matrices to decompose (e.g., "query", "key", "value")
     * For adapters: layers to insert adapters after
     */
    protected List<String> targetModules;

    /**
     * List of module names to exclude from adaptation.
     */
    protected List<String> modulesToExclude;

    /**
     * Get the PEFT type for this configuration.
     */
    public abstract PeftType getPeftType();

    /**
     * Calculate the number of trainable parameters this PEFT method will add.
     *
     * @param originalParamCount The original number of parameters in the model
     * @return The number of trainable parameters after applying PEFT
     */
    public abstract long calculateTrainableParameters(long originalParamCount);

    /**
     * Check if the configuration is valid.
     *
     * @throws IllegalStateException if the configuration is invalid
     */
    public abstract void validate();

    /**
     * Convenience check used by subclasses: asserts that {@link #targetModules} is
     * non-null and non-empty.
     *
     * @throws IllegalStateException if targetModules is null or empty
     */
    protected void validateTargetModules() {
        Preconditions.checkState(targetModules != null && !targetModules.isEmpty(),
                "Target modules must be specified");
    }

    /**
     * Get a human-readable summary of this configuration.
     */
    @JsonIgnore
    public String getSummary() {
        return String.format("PeftConfig(type=%s, task=%s, targets=%s)",
                getPeftType(), taskType, targetModules);
    }
}
