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
import org.nd4j.autodiff.samediff.config.GRPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Group Relative Policy Optimization (GRPO) trainer.
 *
 * Implements the GRPO algorithm with in-graph loss computation and backpropagation:
 * 1. Generate groupSize completions per prompt
 * 2. Score all completions with a reward function
 * 3. Compute z-score advantages within each group (per-prompt normalization)
 * 4. Compute clipped surrogate loss with KL penalty via SameDiff ops
 *
 * The loss graph computes:
 *   L = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage) + klPenalty * KL(pi || ref)
 *
 * where ratio = exp(current_log_prob - old_log_prob)
 *
 * External values (old log probs, advantages, reference log probs) are computed in
 * {@link #prepareInputs(Map)} and fed as placeholders.
 *
 * Adam Gibson
 */
@Slf4j
public class GRPOTrainer extends RLAlignmentTrainer<GRPOConfig> {

    protected final SamplingStrategy samplingStrategy;
    protected final RewardFunction rewardFunction;

    public GRPOTrainer(SameDiff policyModel, SameDiff referenceModel, GRPOConfig config,
                       SamplingStrategy samplingStrategy, RewardFunction rewardFunction) {
        super(policyModel, referenceModel, config);
        this.samplingStrategy = samplingStrategy;
        this.rewardFunction = rewardFunction;
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double clipEps = config.getClipEpsilon();
        double klPenalty = config.getKlPenalty();

        // Placeholders for externally-computed values
        SDVariable tokens = sd.placeHolder("_grpo_tokens", DataType.INT64, -1, -1);
        SDVariable oldLogProbs = sd.placeHolder("_grpo_old_log_probs", DataType.FLOAT, -1);
        SDVariable advantages = sd.placeHolder("_grpo_advantages", DataType.FLOAT, -1);
        SDVariable refLogProbs = sd.placeHolder("_grpo_ref_log_probs", DataType.FLOAT, -1);

        // Get the model's logit output and compute current log probs in-graph
        SDVariable logits = sd.getVariable(logitVar);
        SDVariable currentLP = computeLogProbOps(sd, "_grpo", logits, tokens, vocabSize);

        // ratio = exp(current_lp - old_lp)
        SDVariable logRatio = currentLP.sub("_grpo_log_ratio", oldLogProbs);
        SDVariable ratio = sd.math().exp("_grpo_ratio", logRatio);

        // clipped_ratio = clip(ratio, 1-eps, 1+eps)
        SDVariable clippedRatio = sd.math().clipByValue("_grpo_clipped_ratio", ratio,
                1.0 - clipEps, 1.0 + clipEps);

        // surrogate1 = ratio * advantages
        SDVariable surrogate1 = ratio.mul("_grpo_surr1", advantages);

        // surrogate2 = clipped_ratio * advantages
        SDVariable surrogate2 = clippedRatio.mul("_grpo_surr2", advantages);

        // policy_loss = -min(surrogate1, surrogate2).mean()
        SDVariable minSurrogate = sd.math().min("_grpo_min_surr", surrogate1, surrogate2);
        SDVariable policyLoss = minSurrogate.neg("_grpo_neg_surr").mean("_grpo_policy_loss");

        // Approximate KL divergence: current_lp - ref_lp
        SDVariable kl = currentLP.sub("_grpo_kl", refLogProbs);
        SDVariable klMean = kl.mean("_grpo_kl_mean");

        // total_loss = policy_loss + kl_penalty * kl_mean
        SDVariable klTerm = klMean.mul("_grpo_kl_term", klPenalty);
        SDVariable totalLoss = policyLoss.add("_grpo_loss", klTerm);

        return totalLoss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        String inputVar = config.getInputVariable();
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        int groupSize = config.getGroupSize();
        int maxNewTokens = config.getMaxNewTokens();

        // 1. Get prompts from inputs
        INDArray prompts = rawInputs.get("prompts");
        long batchSize = prompts.size(0);
        long totalCompletions = batchSize * groupSize;

        // 2. Generate groupSize completions per prompt
        INDArray completions = samplingStrategy.generateMultiple(
                policyModel, prompts, groupSize, maxNewTokens, logitVar);

        // 3. Expand prompts to match completions (each prompt repeated groupSize times)
        INDArray expandedPrompts = Nd4j.zeros(DataType.INT64, totalCompletions, prompts.size(1));
        for (int b = 0; b < batchSize; b++) {
            for (int g = 0; g < groupSize; g++) {
                expandedPrompts.putRow(b * groupSize + g, prompts.getRow(b));
            }
        }

        // 4. Score all completions with RewardFunction
        INDArray rewards = rewardFunction.score(expandedPrompts, completions);

        // 5. Compute z-score advantages within each group (per-prompt normalization)
        INDArray advantages = Nd4j.zeros(DataType.FLOAT, totalCompletions);
        for (int b = 0; b < batchSize; b++) {
            int start = b * groupSize;
            double[] groupRewardValues = new double[groupSize];
            for (int g = 0; g < groupSize; g++) {
                groupRewardValues[g] = rewards.getDouble(start + g);
            }
            INDArray groupRewardArr = Nd4j.createFromArray(groupRewardValues);
            double mean = groupRewardArr.meanNumber().doubleValue();
            double std = groupRewardArr.stdNumber().doubleValue();
            for (int g = 0; g < groupSize; g++) {
                double advantage = std < 1e-8 ? 0.0 : (groupRewardValues[g] - mean) / (std + 1e-8);
                advantages.putScalar(start + g, advantage);
            }
        }

        // Normalize advantages globally if configured
        advantages = normalizeAdvantages(advantages);

        // 6. Compute old log probs from policy model (external, before gradient update)
        Map<String, INDArray> completionInputs = Collections.singletonMap(inputVar, completions);
        INDArray oldLogProbs = computeExternalLogProbs(policyModel, completionInputs,
                completions, logitVar, vocabSize);

        // 7. Compute reference log probs from reference model (external)
        INDArray refLogProbs = computeExternalLogProbs(referenceModel, completionInputs,
                completions, logitVar, vocabSize);

        // 8. Build placeholder map
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, completions);
        placeholders.put("_grpo_tokens", completions);
        placeholders.put("_grpo_old_log_probs", oldLogProbs);
        placeholders.put("_grpo_advantages", advantages);
        placeholders.put("_grpo_ref_log_probs", refLogProbs);

        // Pass through any additional placeholders the model needs
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals("prompts")) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
