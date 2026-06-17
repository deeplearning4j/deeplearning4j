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
import org.nd4j.autodiff.samediff.config.DrGRPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * De-biased Reward GRPO (DrGRPO) trainer.
 *
 * Like GRPO but with two key modifications to remove length bias:
 * - Length normalization: divides log probabilities by completion length so that
 *   the probability ratio is not biased toward shorter/longer sequences
 * - Baseline subtraction: subtracts the mean reward before computing advantages
 *   for stronger variance reduction
 *
 * The loss graph is built in SameDiff so gradients flow through the policy model.
 * External values (old log probs, advantages, ref log probs, sequence lengths)
 * are computed in {@link #prepareInputs(Map)} and fed as placeholders.
 *
 * Adam Gibson
 */
@Slf4j
public class DrGRPOTrainer extends RLAlignmentTrainer<DrGRPOConfig> {

    private final SamplingStrategy samplingStrategy;
    private final RewardFunction rewardFunction;

    public DrGRPOTrainer(SameDiff policyModel, SameDiff referenceModel, DrGRPOConfig config,
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
        double klPenalty = config.getKlCoefficient();
        boolean lengthNorm = config.isLengthNormalization();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for externally computed values
        SDVariable tokens = sd.placeHolder("_drgrpo_tokens", DataType.INT64, -1, -1);
        SDVariable oldLogProbs = sd.placeHolder("_drgrpo_old_log_probs", DataType.FLOAT, -1);
        SDVariable advantages = sd.placeHolder("_drgrpo_advantages", DataType.FLOAT, -1);
        SDVariable refLogProbs = sd.placeHolder("_drgrpo_ref_log_probs", DataType.FLOAT, -1);
        SDVariable seqLengths = sd.placeHolder("_drgrpo_seq_lengths", DataType.FLOAT, -1);

        // Compute current policy log probabilities in-graph
        SDVariable currentLP = computeLogProbOps(sd, "_drgrpo", logits, tokens, vocabSize);

        // Length normalization: divide log probs by sequence length
        SDVariable effectiveCurrentLP;
        SDVariable effectiveOldLP;
        if (lengthNorm) {
            // Avoid division by zero
            SDVariable safeLengths = sd.math().max("_drgrpo_safe_len", seqLengths,
                    sd.constant("_drgrpo_one", 1.0f));
            effectiveCurrentLP = currentLP.div("_drgrpo_norm_current", safeLengths);
            effectiveOldLP = oldLogProbs.div("_drgrpo_norm_old", safeLengths);
        } else {
            effectiveCurrentLP = currentLP;
            effectiveOldLP = oldLogProbs;
        }

        // Probability ratio: exp(normalized_current - normalized_old)
        SDVariable logRatio = effectiveCurrentLP.sub("_drgrpo_log_ratio", effectiveOldLP);
        SDVariable ratio = sd.math().exp("_drgrpo_ratio", logRatio);

        // Clipped ratio: clip(ratio, 1 - eps, 1 + eps)
        SDVariable lowerBound = sd.constant("_drgrpo_lower", 1.0 - clipEps);
        SDVariable upperBound = sd.constant("_drgrpo_upper", 1.0 + clipEps);
        SDVariable clippedLow = sd.math().max("_drgrpo_clip_low", ratio, lowerBound);
        SDVariable clippedRatio = sd.math().min("_drgrpo_clip_high", clippedLow, upperBound);

        // Surrogate: -min(ratio * advantages, clipped_ratio * advantages).mean()
        SDVariable surrogate1 = ratio.mul("_drgrpo_surr1", advantages);
        SDVariable surrogate2 = clippedRatio.mul("_drgrpo_surr2", advantages);
        SDVariable surrogateMin = sd.math().min("_drgrpo_surr_min", surrogate1, surrogate2);
        SDVariable policyLoss = surrogateMin.neg("_drgrpo_neg_surr").mean("_drgrpo_policy_loss");

        // KL divergence: mean(current_lp - ref_lp)
        SDVariable klDiff = currentLP.sub("_drgrpo_kl_diff", refLogProbs);
        SDVariable klMean = klDiff.mean("_drgrpo_kl_mean");
        SDVariable klTerm = klMean.mul("_drgrpo_kl_term", klPenalty);

        // Total loss = policy_loss + kl_term
        SDVariable totalLoss = policyLoss.add("_drgrpo_loss", klTerm);

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

        // 4. Compute sequence lengths (count non-padding tokens per completion)
        INDArray seqLengths = Nd4j.zeros(DataType.FLOAT, totalCompletions);
        for (int i = 0; i < totalCompletions; i++) {
            int tokenCount = 0;
            for (int t = 0; t < completions.size(1); t++) {
                if (completions.getInt(i, t) != 0) tokenCount++;
            }
            seqLengths.putScalar(i, Math.max(tokenCount, 1));
        }

        // 5. Length normalization of rewards
        if (config.isLengthNormalization()) {
            for (int i = 0; i < totalCompletions; i++) {
                float len = seqLengths.getFloat(i);
                if (len > 0) {
                    rewards.putScalar(i, rewards.getDouble(i) / len);
                }
            }
        }

        // 6. Baseline subtraction
        if (config.isBaselineSubtraction()) {
            double globalMean = rewards.meanNumber().doubleValue();
            rewards = rewards.sub(globalMean);
        }

        // 7. Compute z-score advantages within each group
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

        // 8. Compute old policy log probs (external, no gradient)
        Map<String, INDArray> completionInputs = Collections.singletonMap(inputVar, completions);
        INDArray oldLogProbs = computeExternalLogProbs(policyModel, completionInputs, completions, logitVar, vocabSize);

        // 9. Compute reference model log probs (external)
        INDArray refLogProbs = computeExternalLogProbs(referenceModel, completionInputs, completions, logitVar, vocabSize);

        // 10. Build placeholder map
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, completions);
        placeholders.put("_drgrpo_tokens", completions.castTo(DataType.INT64));
        placeholders.put("_drgrpo_old_log_probs", oldLogProbs);
        placeholders.put("_drgrpo_advantages", advantagesArr);
        placeholders.put("_drgrpo_ref_log_probs", refLogProbs);
        placeholders.put("_drgrpo_seq_lengths", seqLengths);

        // Pass through any additional placeholders
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals("prompts")) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
