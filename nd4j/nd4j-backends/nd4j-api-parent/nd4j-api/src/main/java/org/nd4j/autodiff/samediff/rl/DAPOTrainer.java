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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.RLAlignmentTrainer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.DAPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Decoupled Alignment Policy Optimization (DAPO) trainer.
 *
 * Extends GRPO with:
 * - Asymmetric clipping: clip(ratio, 1 - clipEpsilonLow, 1 + clipEpsilonHigh)
 * - Token-level KL instead of sequence-level KL divergence
 * - Dynamic sampling: skips groups where all rewards are equal
 * - Overlong filtering: filters completions exceeding maxNewTokens
 *
 * The loss graph is built in SameDiff so gradients flow through the policy model.
 * External values (old log probs, advantages, ref log probs, group masks) are
 * computed in {@link #prepareInputs(Map)} and fed as placeholders.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class DAPOTrainer extends RLAlignmentTrainer<DAPOConfig> {

    private final SamplingStrategy samplingStrategy;
    private final RewardFunction rewardFunction;

    public DAPOTrainer(SameDiff policyModel, SameDiff referenceModel, DAPOConfig config,
                       SamplingStrategy samplingStrategy, RewardFunction rewardFunction) {
        super(policyModel, referenceModel, config);
        this.samplingStrategy = samplingStrategy;
        this.rewardFunction = rewardFunction;
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double clipLow = config.getClipEpsilonLow();
        double clipHigh = config.getClipEpsilonHigh();
        double klPenalty = config.getKlCoefficient();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for externally computed values
        SDVariable tokens = sd.placeHolder("_dapo_tokens", DataType.INT64, -1, -1);
        SDVariable oldLogProbs = sd.placeHolder("_dapo_old_log_probs", DataType.FLOAT, -1);
        SDVariable advantages = sd.placeHolder("_dapo_advantages", DataType.FLOAT, -1);
        SDVariable refLogProbs = sd.placeHolder("_dapo_ref_log_probs", DataType.FLOAT, -1);
        SDVariable groupMask = sd.placeHolder("_dapo_group_mask", DataType.FLOAT, -1);

        // Compute current policy log probabilities in-graph
        SDVariable currentLP = computeLogProbOps(sd, "_dapo", logits, tokens, vocabSize);

        // Probability ratio: exp(current - old)
        SDVariable logRatio = currentLP.sub("_dapo_log_ratio", oldLogProbs);
        SDVariable ratio = sd.math().exp("_dapo_ratio", logRatio);

        // Asymmetric clipping: clip to [1 - clipLow, 1 + clipHigh]
        SDVariable lowerBound = sd.constant("_dapo_lower", 1.0 - clipLow);
        SDVariable upperBound = sd.constant("_dapo_upper", 1.0 + clipHigh);
        // First clip from below: max(ratio, 1 - clipLow)
        SDVariable clippedLow = sd.math().max("_dapo_clip_low", ratio, lowerBound);
        // Then clip from above: min(clippedLow, 1 + clipHigh)
        SDVariable clippedRatio = sd.math().min("_dapo_clip_high", clippedLow, upperBound);

        // Surrogate objectives
        SDVariable surrogate1 = ratio.mul("_dapo_surr1", advantages);
        SDVariable surrogate2 = clippedRatio.mul("_dapo_surr2", advantages);
        SDVariable surrogateMin = sd.math().min("_dapo_surr_min", surrogate1, surrogate2);

        // Apply group mask: zero out filtered groups
        SDVariable maskedSurrogate = surrogateMin.mul("_dapo_masked_surr", groupMask);

        // Policy loss: -sum(masked_surrogate) / (sum(mask) + eps)
        SDVariable maskSum = groupMask.sum("_dapo_mask_sum");
        SDVariable maskSumSafe = maskSum.add("_dapo_mask_sum_safe", 1e-10);
        SDVariable surrogateSum = maskedSurrogate.sum("_dapo_surr_sum");
        SDVariable policyLoss = surrogateSum.div("_dapo_surr_mean", maskSumSafe).neg("_dapo_policy_loss");

        // KL divergence: (current_lp - ref_lp) * group_mask
        SDVariable klDiff = currentLP.sub("_dapo_kl_diff", refLogProbs);
        SDVariable maskedKL = klDiff.mul("_dapo_masked_kl", groupMask);
        SDVariable klMean = maskedKL.sum("_dapo_kl_sum").div("_dapo_kl_mean", maskSumSafe);

        // Total loss = policy_loss + kl_penalty * kl_mean
        SDVariable klTerm = klMean.mul("_dapo_kl_term", klPenalty);
        SDVariable totalLoss = policyLoss.add("_dapo_loss", klTerm);

        return totalLoss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        String inputVar = config.getInputVariable();
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        int groupSize = config.getGroupSize();

        // 1. Get prompts
        INDArray prompts = rawInputs.get("prompts");
        long batchSize = prompts.size(0);
        long totalCompletions = batchSize * groupSize;

        // 2. Generate groupSize completions per prompt
        INDArray completions = samplingStrategy.generateMultiple(
                policyModel, prompts, groupSize, config.getMaxNewTokens(), logitVar);

        // 3. Expand prompts and score
        INDArray expandedPrompts = Nd4j.zeros(totalCompletions, prompts.size(1));
        for (int b = 0; b < batchSize; b++) {
            for (int g = 0; g < groupSize; g++) {
                expandedPrompts.putRow(b * groupSize + g, prompts.getRow(b));
            }
        }
        INDArray rewards = rewardFunction.score(expandedPrompts, completions);

        // 4. Compute sequence lengths and overlong filtering
        boolean[] validCompletion = new boolean[(int) totalCompletions];
        int validCount = 0;
        for (int i = 0; i < totalCompletions; i++) {
            if (config.isOverlongFiltering()) {
                int tokenCount = 0;
                for (int t = 0; t < completions.size(1); t++) {
                    if (completions.getInt(i, t) != 0) tokenCount++;
                }
                validCompletion[i] = tokenCount <= config.getMaxNewTokens();
            } else {
                validCompletion[i] = true;
            }
            if (validCompletion[i]) validCount++;
        }

        // 5. Compute z-score advantages with dynamic sampling; build group mask
        INDArray advantagesArr = Nd4j.zeros(DataType.FLOAT, totalCompletions);
        INDArray groupMaskArr = Nd4j.ones(DataType.FLOAT, totalCompletions);

        for (int b = 0; b < batchSize; b++) {
            int start = b * groupSize;
            double[] groupRewards = new double[groupSize];
            int groupValidCount = 0;
            for (int g = 0; g < groupSize; g++) {
                if (validCompletion[start + g]) {
                    groupRewards[g] = rewards.getDouble(start + g);
                    groupValidCount++;
                }
            }

            // Dynamic sampling: skip groups where all rewards are equal
            boolean groupActive = true;
            if (config.isDynamicSampling() && groupValidCount > 0) {
                double first = groupRewards[0];
                boolean allEqual = true;
                for (int g = 1; g < groupSize; g++) {
                    if (validCompletion[start + g] && Math.abs(groupRewards[g] - first) > 1e-8) {
                        allEqual = false;
                        break;
                    }
                }
                if (allEqual) {
                    groupActive = false;
                }
            }

            // Set group mask
            for (int g = 0; g < groupSize; g++) {
                if (!groupActive || !validCompletion[start + g]) {
                    groupMaskArr.putScalar(start + g, 0.0f);
                }
            }

            if (!groupActive) continue;

            // Compute z-score within group
            double sum = 0, sumSq = 0;
            int count = 0;
            for (int g = 0; g < groupSize; g++) {
                if (validCompletion[start + g]) {
                    sum += groupRewards[g];
                    sumSq += groupRewards[g] * groupRewards[g];
                    count++;
                }
            }
            double mean = count > 0 ? sum / count : 0;
            double variance = count > 1 ? (sumSq / count - mean * mean) : 0;
            double std = Math.sqrt(Math.max(variance, 0));

            for (int g = 0; g < groupSize; g++) {
                if (validCompletion[start + g]) {
                    double adv = std < 1e-8 ? 0.0 : (groupRewards[g] - mean) / (std + 1e-8);
                    advantagesArr.putScalar(start + g, adv);
                }
            }
        }

        // 6. Compute old policy log probs (external, no gradients needed)
        Map<String, INDArray> completionInputs = Collections.singletonMap(inputVar, completions);
        INDArray oldLogProbs = computeExternalLogProbs(policyModel, completionInputs, completions, logitVar, vocabSize);

        // 7. Compute reference model log probs (external)
        INDArray refLogProbs = computeExternalLogProbs(referenceModel, completionInputs, completions, logitVar, vocabSize);

        // 8. Build placeholder map
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, completions);
        placeholders.put("_dapo_tokens", completions.castTo(DataType.INT64));
        placeholders.put("_dapo_old_log_probs", oldLogProbs);
        placeholders.put("_dapo_advantages", advantagesArr);
        placeholders.put("_dapo_ref_log_probs", refLogProbs);
        placeholders.put("_dapo_group_mask", groupMaskArr);

        // Pass through any additional placeholders
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals("prompts")) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
