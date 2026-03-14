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
import org.nd4j.autodiff.samediff.config.PPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Proximal Policy Optimization (PPO) trainer with in-graph loss computation.
 *
 * Implements the PPO algorithm with GAE advantage estimation and proper backpropagation:
 * 1. Generate completions using the policy
 * 2. Score with reward function
 * 3. Compute advantages using Generalized Advantage Estimation (GAE)
 * 4. Perform multiple epochs of clipped surrogate updates via SameDiff backprop
 *
 * The loss graph computes:
 *   total_loss = policy_loss + valueLossCoeff * value_loss - entropyCoeff * entropy
 *
 * where:
 *   policy_loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
 *   entropy = -mean(sum(probs * log_probs, axis=-1))
 *   value_loss is computed externally (separate value model) and fed as a placeholder
 *
 * PPO's inner loop of multiple epochs is implemented by overriding {@link #trainStep(Map)}.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class PPOTrainer extends RLAlignmentTrainer<PPOConfig> {

    private final SamplingStrategy samplingStrategy;
    private final RewardFunction rewardFunction;
    private final SameDiff valueModel;

    /**
     * Cached placeholders for PPO's inner epoch loop.
     * When set, prepareInputs returns this map directly instead of recomputing.
     * This allows multiple gradient steps on the same batch of experience.
     */
    private Map<String, INDArray> cachedPlaceholders;

    public PPOTrainer(SameDiff policyModel, SameDiff referenceModel, PPOConfig config,
                      SamplingStrategy samplingStrategy, RewardFunction rewardFunction,
                      SameDiff valueModel) {
        super(policyModel, referenceModel, config);
        this.samplingStrategy = samplingStrategy;
        this.rewardFunction = rewardFunction;
        this.valueModel = valueModel;
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double clipEps = config.getClipEpsilon();
        double valueLossCoeff = config.getValueLossCoeff();
        double entropyCoeff = config.getEntropyCoeff();

        // Placeholders for externally-computed values
        SDVariable tokens = sd.placeHolder("_ppo_tokens", DataType.INT64, -1, -1);
        SDVariable oldLogProbs = sd.placeHolder("_ppo_old_log_probs", DataType.FLOAT, -1);
        SDVariable advantages = sd.placeHolder("_ppo_advantages", DataType.FLOAT, -1);
        SDVariable returns = sd.placeHolder("_ppo_returns", DataType.FLOAT, -1);
        SDVariable valueLossPlaceholder = sd.placeHolder("_ppo_value_loss", DataType.FLOAT);

        // Get the model's logit output and compute current log probs in-graph
        SDVariable logits = sd.getVariable(logitVar);
        SDVariable currentLP = computeLogProbOps(sd, "_ppo", logits, tokens, vocabSize);

        // ratio = exp(current_lp - old_lp)
        SDVariable logRatio = currentLP.sub("_ppo_log_ratio", oldLogProbs);
        SDVariable ratio = sd.math().exp("_ppo_ratio", logRatio);

        // clipped_ratio = clip(ratio, 1-eps, 1+eps)
        SDVariable clippedRatio = sd.math().clipByValue("_ppo_clipped_ratio", ratio,
                1.0 - clipEps, 1.0 + clipEps);

        // surrogate1 = ratio * advantages, surrogate2 = clipped_ratio * advantages
        SDVariable surrogate1 = ratio.mul("_ppo_surr1", advantages);
        SDVariable surrogate2 = clippedRatio.mul("_ppo_surr2", advantages);

        // policy_loss = -min(surrogate1, surrogate2).mean()
        SDVariable minSurrogate = sd.math().min("_ppo_min_surr", surrogate1, surrogate2);
        SDVariable policyLoss = minSurrogate.neg("_ppo_neg_surr").mean("_ppo_policy_loss");

        // Entropy bonus: log_softmax -> probs = exp(log_softmax), entropy = -sum(probs * log_softmax, axis=-1).mean()
        SDVariable logSoftmax = sd.nn().logSoftmax("_ppo_logsm_ent", logits, -1);
        SDVariable probs = sd.math().exp("_ppo_probs_ent", logSoftmax);
        SDVariable pLogP = probs.mul("_ppo_plogp", logSoftmax);
        SDVariable entropyPerToken = pLogP.sum("_ppo_ent_per_token", -1).neg("_ppo_neg_ent_per_token");
        SDVariable entropy = entropyPerToken.mean("_ppo_entropy");

        // total_loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy
        SDVariable valueTerm = valueLossPlaceholder.mul("_ppo_value_term", valueLossCoeff);
        SDVariable entropyTerm = entropy.mul("_ppo_entropy_term", entropyCoeff);
        SDVariable totalLoss = policyLoss.add("_ppo_loss_plus_value", valueTerm)
                .sub("_ppo_loss", entropyTerm);

        return totalLoss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        // If cached placeholders are set (PPO inner epoch loop), return them directly
        if (cachedPlaceholders != null) {
            // Recompute value loss for current value model state each epoch
            updateValueLoss(cachedPlaceholders);
            return cachedPlaceholders;
        }

        String inputVar = config.getInputVariable();
        String logitVar = config.getPolicyLogitVariable();
        String valueVar = config.getValueVariable();
        int vocabSize = config.getVocabSize();
        int maxNewTokens = config.getMaxNewTokens();
        double gaeLambda = config.getGaeLambda();

        // 1. Get prompts and generate completions
        INDArray prompts = rawInputs.get("prompts");
        long batchSize = prompts.size(0);

        INDArray completions = samplingStrategy.generate(
                policyModel, prompts, maxNewTokens, logitVar);
        long seqLen = completions.size(1);

        // 2. Score with reward function
        INDArray rewards = rewardFunction.score(prompts, completions);

        // 3. Get value estimates for each timestep from the value model
        Map<String, INDArray> completionInputs = Collections.singletonMap(inputVar, completions);
        INDArray values = valueModel.output(completionInputs, valueVar).get(valueVar);
        if (values.rank() > 2) {
            values = values.reshape(batchSize, seqLen);
        }

        // 4. Compute GAE advantages and returns
        double gamma = 1.0; // discount factor (typically 1.0 for language tasks)
        INDArray gaeAdvantages = Nd4j.zeros(DataType.FLOAT, batchSize, seqLen);
        INDArray gaeReturns = Nd4j.zeros(DataType.FLOAT, batchSize, seqLen);

        for (int b = 0; b < batchSize; b++) {
            double lastGaeLam = 0.0;
            double reward = rewards.getDouble(b); // terminal reward

            for (int t = (int) seqLen - 1; t >= 0; t--) {
                double nextValue = (t < seqLen - 1) ? values.getDouble(b, t + 1) : 0.0;
                double currentValue = values.getDouble(b, t);

                // Only apply reward at the last timestep
                double stepReward = (t == seqLen - 1) ? reward : 0.0;
                double delta = stepReward + gamma * nextValue - currentValue;

                lastGaeLam = delta + gamma * gaeLambda * lastGaeLam;
                gaeAdvantages.putScalar(new int[]{b, t}, lastGaeLam);
                gaeReturns.putScalar(new int[]{b, t}, lastGaeLam + currentValue);
            }
        }

        // 5. Flatten and normalize advantages
        // Use mean over sequence dimension to get per-sample advantages [batch]
        INDArray flatAdvantages = gaeAdvantages.mean(1);
        flatAdvantages = normalizeAdvantages(flatAdvantages);

        // Flatten returns to per-sample [batch] (mean over seq for the value loss placeholder)
        INDArray flatReturns = gaeReturns.mean(1);

        // 6. Compute old log probs from policy model (external, before gradient update)
        INDArray oldLogProbs = computeExternalLogProbs(policyModel, completionInputs,
                completions, logitVar, vocabSize);

        // 7. Compute value loss externally: (new_values - returns)^2.mean()
        INDArray flatValues = values.mean(1);
        INDArray valueDiff = flatValues.sub(flatReturns);
        INDArray valueLoss = valueDiff.mul(valueDiff).mean();

        // 8. Build placeholder map
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, completions);
        placeholders.put("_ppo_tokens", completions);
        placeholders.put("_ppo_old_log_probs", oldLogProbs);
        placeholders.put("_ppo_advantages", flatAdvantages);
        placeholders.put("_ppo_returns", flatReturns);
        placeholders.put("_ppo_value_loss", valueLoss);

        // Stash returns and completionInputs for value loss recomputation in inner loop
        placeholders.put("__ppo_internal_returns", flatReturns);

        // Pass through any additional placeholders the model needs
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals("prompts")) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }

    /**
     * Override trainStep to implement PPO's inner loop of multiple gradient steps
     * on the same batch of collected experience.
     *
     * @param inputs Raw input data map
     * @return The average loss across all PPO epochs
     */
    @Override
    public double trainStep(Map<String, INDArray> inputs) {
        int ppoEpochs = config.getPpoEpochs();

        // Compute external values ONCE (generation, rewards, GAE, old log probs)
        // by calling prepareInputs with cachedPlaceholders = null
        cachedPlaceholders = null;
        // Trigger lazy init if needed, then prepare inputs once
        Map<String, INDArray> placeholders = prepareInputs(inputs);

        // Cache the placeholders for the inner loop
        // (old_log_probs are fixed, only current policy logits change across epochs)
        cachedPlaceholders = placeholders;

        double totalLoss = 0.0;
        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            // super.trainStep will call prepareInputs, which returns cachedPlaceholders
            // with an updated value loss each epoch
            totalLoss += super.trainStep(inputs);
        }

        // Clear cache after all epochs complete
        cachedPlaceholders = null;

        return totalLoss / ppoEpochs;
    }

    /**
     * Recompute value loss from the current value model state and update the placeholder.
     * Called each PPO epoch to reflect the evolving value function.
     */
    private void updateValueLoss(Map<String, INDArray> placeholders) {
        String inputVar = config.getInputVariable();
        String valueVar = config.getValueVariable();

        INDArray completions = placeholders.get(inputVar);
        INDArray flatReturns = placeholders.get("__ppo_internal_returns");
        if (completions == null || flatReturns == null) return;

        Map<String, INDArray> completionInputs = Collections.singletonMap(inputVar, completions);
        INDArray values = valueModel.output(completionInputs, valueVar).get(valueVar);
        long batchSize = completions.size(0);
        long seqLen = completions.size(1);
        if (values.rank() > 2) {
            values = values.reshape(batchSize, seqLen);
        }
        INDArray flatValues = values.mean(1);
        INDArray valueDiff = flatValues.sub(flatReturns);
        INDArray valueLoss = valueDiff.mul(valueDiff).mean();
        placeholders.put("_ppo_value_loss", valueLoss);
    }
}
