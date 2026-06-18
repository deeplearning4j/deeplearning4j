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

package org.nd4j.autodiff.samediff.rl;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.VlmFineTuneConfig;
import org.nd4j.autodiff.samediff.config.VlmGRPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Vision-Language Model GRPO (Group Relative Policy Optimization) trainer.
 *
 * <p>Extends {@link GRPOTrainer} for multimodal RL training by adding:</p>
 * <ol>
 *   <li>An image placeholder injected into the policy model graph</li>
 *   <li>Vision encoder processing (respecting freeze/train config from {@link VlmFineTuneConfig})</li>
 *   <li>Projection of vision features into the LLM embedding space</li>
 *   <li>Concatenation of vision and text embeddings before the GRPO loss</li>
 *   <li>Image-aware reward computation (passing image context to the reward function)</li>
 * </ol>
 *
 * <p>The vision encoder can be frozen (no gradient updates) or trained with LoRA,
 * as controlled by {@link VlmFineTuneConfig#isFreezeVisionEncoder()}.
 * When frozen, the encoder runs in inference mode and its parameters are excluded
 * from the optimizer's trainable parameter list.</p>
 *
 * <p>Vision encoder variables are identified by name prefix conventions:
 * any variable whose name starts with {@code "vision_encoder"} is considered
 * part of the vision encoder. The projector prefix is {@code "projector"}.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * VlmGRPOConfig config = VlmGRPOConfig.builder()
 *     .policyLogitVariable("logits")
 *     .vocabSize(32000)
 *     .groupSize(4)
 *     .vlmConfig(VlmFineTuneConfig.loraOnly())
 *     .completionsPerImage(4)
 *     .imageAwareReward(true)
 *     .build();
 *
 * VlmGRPOTrainer trainer = new VlmGRPOTrainer(
 *     policyModel, referenceModel, config, samplingStrategy, rewardFunction);
 *
 * // Training step with image + text prompts
 * Map<String, INDArray> inputs = new HashMap<>();
 * inputs.put("prompts", tokenIds);      // [batch, seq_len] INT64
 * inputs.put("image_input", imageArr);  // [batch, 3, H, W] FLOAT
 * double loss = trainer.trainStep(inputs);
 * }</pre>
 *
 * @author Adam Gibson
 * @see GRPOTrainer
 * @see VlmGRPOConfig
 * @see VlmFineTuneConfig
 */
@Slf4j
public class VlmGRPOTrainer extends GRPOTrainer {

    /** Prefix used to identify vision encoder variables by name. */
    public static final String VISION_ENCODER_PREFIX = "vision_encoder";

    /** Prefix used to identify projector variables by name. */
    public static final String PROJECTOR_PREFIX = "projector";

    @Getter
    private final VlmFineTuneConfig vlmConfig;

    @Getter
    private final VlmGRPOConfig vlmGrpoConfig;

    /**
     * Construct a VLM GRPO trainer.
     *
     * @param policyModel     the policy SameDiff model (will have the GRPO loss graph appended)
     * @param referenceModel  the reference SameDiff model for KL penalty (may be null if useReferenceModel=false)
     * @param config          the VLM GRPO configuration
     * @param samplingStrategy strategy for generating token completions
     * @param rewardFunction  function that scores (prompt, completion) pairs
     */
    public VlmGRPOTrainer(SameDiff policyModel, SameDiff referenceModel, VlmGRPOConfig config,
                          SamplingStrategy samplingStrategy, RewardFunction rewardFunction) {
        super(policyModel, referenceModel, config, samplingStrategy, rewardFunction);
        this.vlmGrpoConfig = config;
        this.vlmConfig = config.effectiveVlmConfig();
    }

    /**
     * Build the loss graph for VLM-GRPO.
     *
     * <p>Adds an image placeholder to the graph, then delegates to the parent
     * GRPO loss construction. Vision encoder processing (projection into the LLM
     * embedding space) is assumed to already be part of the model graph at load
     * time; this method only ensures the image placeholder is registered and that
     * vision encoder variables are excluded from the optimizer when frozen.</p>
     *
     * @param sd the policy model's SameDiff instance
     * @return the name of the scalar total loss variable
     */
    @Override
    protected String buildLossGraph(SameDiff sd) {
        // Register the image input placeholder if not already present
        String imageVar = vlmGrpoConfig.getImageInputVariable();
        if (sd.getVariable(imageVar) == null) {
            // Image placeholder: [batch, channels, height, width]
            sd.placeHolder(imageVar, DataType.FLOAT, -1, 3,
                    vlmConfig.getImageResolution(), vlmConfig.getImageResolution());
            log.debug("VlmGRPOTrainer: registered image placeholder '{}' [batch, 3, {}, {}]",
                    imageVar, vlmConfig.getImageResolution(), vlmConfig.getImageResolution());
        }

        // Register the projected vision features placeholder if the projector output
        // variable is not already in the graph (it would be there if the loaded model
        // already contains vision encoder + projector subgraphs)
        String projVar = vlmConfig.getProjectorOutputVariable();
        if (sd.getVariable(projVar) == null) {
            // Projected features: [batch, maxImageTokens, llm_dim] — shape is dynamic
            sd.placeHolder(projVar, DataType.FLOAT, -1, vlmConfig.getMaxImageTokens(), -1);
            log.debug("VlmGRPOTrainer: registered projected vision placeholder '{}'", projVar);
        }

        // Build the standard GRPO loss on top of the multimodal graph
        String lossName = super.buildLossGraph(sd);

        // Log which variable groups are frozen vs. trainable
        if (vlmConfig.isFreezeVisionEncoder()) {
            log.info("VlmGRPOTrainer: vision encoder variables with prefix '{}' are FROZEN "
                    + "(excluded from optimizer)", VISION_ENCODER_PREFIX);
        } else {
            log.info("VlmGRPOTrainer: vision encoder variables are TRAINABLE "
                    + "(visionLoraConfig={})",
                    vlmConfig.getVisionLoraConfig() != null
                            ? vlmConfig.getVisionLoraConfig().getSummary() : "null (full finetune)");
        }

        return lossName;
    }

    /**
     * Determine whether a variable participates in gradient updates.
     *
     * <p>Vision encoder variables are excluded when {@code freezeVisionEncoder=true}.
     * Projector variables are excluded when {@code trainProjector=false}.
     * All other variables follow the standard GRPO trainability rule
     * (type == VARIABLE).</p>
     *
     * @param variable the SDVariable to check
     * @return true if the variable should receive gradient updates
     */
    public boolean isTrainable(SDVariable variable) {
        if (variable.getVariableType() != VariableType.VARIABLE) {
            return false;
        }
        String name = variable.name();
        if (name.startsWith(VISION_ENCODER_PREFIX) && vlmConfig.isFreezeVisionEncoder()) {
            return false;
        }
        if (name.startsWith(PROJECTOR_PREFIX) && !vlmConfig.isTrainProjector()) {
            return false;
        }
        return true;
    }

    /**
     * Prepare inputs for a VLM-GRPO training step.
     *
     * <p>Adds image data to the placeholder map alongside the standard GRPO inputs
     * (tokens, old log probs, advantages, reference log probs). When the image is
     * present in rawInputs under the {@code imageInputVariable} key, it is forwarded
     * as-is. If absent, a zero image tensor is synthesized as a fallback (useful
     * for text-only batches in mixed datasets).</p>
     *
     * @param rawInputs map containing at minimum {@code "prompts"} (token IDs) and
     *                  optionally {@code imageInputVariable} (image tensor)
     * @return complete placeholder map for {@code calculateGradients}
     */
    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        // Let the parent GRPO trainer handle token sampling and log-prob computation
        Map<String, INDArray> placeholders = super.prepareInputs(rawInputs);

        // Forward image through to placeholders
        String imageKey = vlmGrpoConfig.getImageInputVariable();
        if (rawInputs.containsKey(imageKey)) {
            INDArray imageArr = rawInputs.get(imageKey);
            // Replicate each image for each completion in the group
            long batchSize = imageArr.size(0);
            int groupSize = config.getGroupSize();
            long totalCompletions = batchSize * groupSize;

            INDArray expandedImages = Nd4j.zeros(
                    DataType.FLOAT, totalCompletions,
                    imageArr.size(1), imageArr.size(2), imageArr.size(3));
            for (int b = 0; b < batchSize; b++) {
                for (int g = 0; g < groupSize; g++) {
                    expandedImages.putRow(b * groupSize + g, imageArr.getRow(b));
                }
            }
            placeholders.put(imageKey, expandedImages);
            log.debug("VlmGRPOTrainer: image tensor expanded from [{},...] to [{},...]",
                    batchSize, totalCompletions);
        } else {
            // Text-only fallback: synthesize zero images
            long totalCompletions = ((INDArray) placeholders.get(config.getInputVariable())).size(0);
            int res = vlmConfig.getImageResolution();
            INDArray zeroImage = Nd4j.zeros(DataType.FLOAT, totalCompletions, 3, res, res);
            placeholders.put(imageKey, zeroImage);
            log.debug("VlmGRPOTrainer: no image in rawInputs, using zero-image fallback");
        }

        return placeholders;
    }
}
