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
import org.nd4j.autodiff.samediff.config.GSPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Group Stable Policy Optimization (GSPO) trainer.
 *
 * Like GRPO but with:
 * - Importance-weighted advantages for variance reduction
 * - Stability coefficient added to the loss for more robust training
 *
 * The stability term penalizes large deviations of the probability ratio from 1,
 * preventing catastrophic policy updates. Importance weights are fed as external
 * placeholders to avoid gradient flow through them.
 *
 * The loss graph is built in SameDiff so gradients flow through the policy model.
 * External values (old log probs, advantages, ref log probs, importance weights)
 * are computed in {@link #prepareInputs(Map)} and fed as placeholders.
 *
 * Adam Gibson
 */
@Slf4j
public class GSPOTrainer extends RLAlignmentTrainer<GSPOConfig> {

    private final SamplingStrategy samplingStrategy;
    private final RewardFunction rewardFunction;

    public GSPOTrainer(SameDiff policyModel, SameDiff referenceModel, GSPOConfig config,
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
        double stabilityCoeff = config.getStabilityCoeff();
        double klPenalty = config.getKlCoefficient();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for externally computed values
        SDVariable tokens = sd.placeHolder("_gspo_tokens", DataType.INT64, -1, -1);
        SDVariable oldLogProbs = sd.placeHolder("_gspo_old_log_probs", DataType.FLOAT, -1);
        SDVariable advantages = sd.placeHolder("_gspo_advantages", DataType.FLOAT, -1);
        SDVariable refLogProbs = sd.placeHolder("_gspo_ref_log_probs", DataType.FLOAT, -1);
        SDVariable importanceWeights = sd.placeHolder("_gspo_importance_weights", DataType.FLOAT, -1);

        // Compute current policy log probabilities in-graph
        SDVariable currentLP = computeLogProbOps(sd, "_gspo", logits, tokens, vocabSize);

        // Probability ratio: exp(current - old)
        SDVariable logRatio = currentLP.sub("_gspo_log_ratio", oldLogProbs);
        SDVariable ratio = sd.math().exp("_gspo_ratio", logRatio);

        // Importance-weighted advantages (weights are external, no gradient through them)
        SDVariable weightedAdv = advantages.mul("_gspo_weighted_adv", importanceWeights);

        // Clipped ratio: clip(ratio, 1 - eps, 1 + eps)
        SDVariable lowerBound = sd.constant("_gspo_lower", 1.0 - clipEps);
        SDVariable upperBound = sd.constant("_gspo_upper", 1.0 + clipEps);
        SDVariable clippedLow = sd.math().max("_gspo_clip_low", ratio, lowerBound);
        SDVariable clippedRatio = sd.math().min("_gspo_clip_high", clippedLow, upperBound);

        // Surrogate: -min(ratio * weighted_adv, clipped_ratio * weighted_adv).mean()
        SDVariable surrogate1 = ratio.mul("_gspo_surr1", weightedAdv);
        SDVariable surrogate2 = clippedRatio.mul("_gspo_surr2", weightedAdv);
        SDVariable surrogateMin = sd.math().min("_gspo_surr_min", surrogate1, surrogate2);
        SDVariable policyLoss = surrogateMin.neg("_gspo_neg_surr").mean("_gspo_policy_loss");

        // Stability loss: stabilityCoeff * mean((ratio - 1)^2)
        SDVariable ratioDev = ratio.sub("_gspo_ratio_dev", 1.0);
        SDVariable ratioDevSq = ratioDev.mul("_gspo_ratio_dev_sq", ratioDev);
        SDVariable stabilityLoss = ratioDevSq.mean("_gspo_stab_mean").mul("_gspo_stab_loss", stabilityCoeff);

        // KL divergence: mean(current_lp - ref_lp)
        SDVariable klDiff = currentLP.sub("_gspo_kl_diff", refLogProbs);
        SDVariable klMean = klDiff.mean("_gspo_kl_mean");
        SDVariable klTerm = klMean.mul("_gspo_kl_term", klPenalty);

        // Total loss = policy_loss + stability_loss + kl_term
        SDVariable totalLoss = policyLoss.add("_gspo_loss_partial", stabilityLoss)
                .add("_gspo_loss", klTerm);

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

        // 4. Compute old policy log probs (external, no gradient)
        Map<String, INDArray> completionInputs = Collections.singletonMap(inputVar, completions);
        INDArray oldLogProbs = computeExternalLogProbs(policyModel, completionInputs, completions, logitVar, vocabSize);

        // 5. Compute reference model log probs (external)
        INDArray refLogProbs = computeExternalLogProbs(referenceModel, completionInputs, completions, logitVar, vocabSize);

        // 6. Compute z-score advantages with optional importance weighting
        INDArray advantagesArr = Nd4j.zeros(DataType.FLOAT, totalCompletions);
        for (int b = 0; b < batchSize; b++) {
            int start = b * groupSize;
            double[] groupRewards = new double[groupSize];
            for (int g = 0; g < groupSize; g++) {
                groupRewards[g] = rewards.getDouble(start + g);
            }

            double mean = 0;
            for (double r : groupRewards) mean += r;
            mean /= groupSize;

            double variance = 0;
            for (double r : groupRewards) variance += (r - mean) * (r - mean);
            variance /= groupSize;
            double std = Math.sqrt(Math.max(variance, 0));

            for (int g = 0; g < groupSize; g++) {
                double adv = std < 1e-8 ? 0.0 : (groupRewards[g] - mean) / (std + 1e-8);
                advantagesArr.putScalar(start + g, adv);
            }
        }

        // 7. Compute importance weights
        // Standard GSPO: importance weights are ones (first iteration baseline).
        // When importance-weighted advantage is enabled, the weighting effect
        // is achieved via the placeholder so gradients don't flow through it.
        INDArray importanceWeights;
        if (config.isImportanceWeightedAdvantage()) {
            // Compute normalized probability within each group as importance weights
            importanceWeights = Nd4j.ones(DataType.FLOAT, totalCompletions);
            for (int b = 0; b < batchSize; b++) {
                int start = b * groupSize;
                double groupProbSum = 0;
                for (int g = 0; g < groupSize; g++) {
                    groupProbSum += Math.exp(oldLogProbs.getDouble(start + g));
                }
                for (int g = 0; g < groupSize; g++) {
                    double weight = Math.exp(oldLogProbs.getDouble(start + g)) / (groupProbSum + 1e-10);
                    importanceWeights.putScalar(start + g, (float) (weight * groupSize));
                }
            }
        } else {
            importanceWeights = Nd4j.ones(DataType.FLOAT, totalCompletions);
        }

        // 8. Build placeholder map
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, completions);
        placeholders.put("_gspo_tokens", completions.castTo(DataType.INT64));
        placeholders.put("_gspo_old_log_probs", oldLogProbs);
        placeholders.put("_gspo_advantages", advantagesArr);
        placeholders.put("_gspo_ref_log_probs", refLogProbs);
        placeholders.put("_gspo_importance_weights", importanceWeights);

        // Pass through any additional placeholders
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals("prompts")) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
