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

import org.nd4j.autodiff.samediff.RLAlignmentTrainer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.DPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Direct Preference Optimization (DPO) trainer.
 *
 * Loss = -log sigmoid(beta * ((log_pi_chosen - log_pi_rejected) - (log_ref_chosen - log_ref_rejected)))
 *
 * The policy model processes chosen and rejected sequences in a single batched forward pass.
 * Chosen sequences occupy the first half of the batch, rejected the second half.
 * Reference model log probabilities are computed externally and fed as placeholders.
 *
 * Supports variants: STANDARD, IPO (identity preference optimization), RDPO (robust DPO).
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class DPOTrainer extends RLAlignmentTrainer<DPOConfig> {

    public DPOTrainer(SameDiff policyModel, SameDiff referenceModel, DPOConfig config) {
        super(policyModel, referenceModel, config);
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double beta = config.getBeta();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for batch indices (to split chosen/rejected from batched logits)
        SDVariable chosenIdx = sd.placeHolder("_dpo_chosen_idx", DataType.INT64, -1);
        SDVariable rejectedIdx = sd.placeHolder("_dpo_rejected_idx", DataType.INT64, -1);

        // Placeholders for token IDs (for log prob computation)
        SDVariable chosenTokens = sd.placeHolder("_dpo_chosen_tokens", DataType.INT64, -1, -1);
        SDVariable rejectedTokens = sd.placeHolder("_dpo_rejected_tokens", DataType.INT64, -1, -1);

        // Placeholders for reference model log probabilities
        SDVariable refChosenLP = sd.placeHolder("_dpo_ref_chosen_lp", DataType.FLOAT, -1);
        SDVariable refRejectedLP = sd.placeHolder("_dpo_ref_rejected_lp", DataType.FLOAT, -1);

        // Split logits into chosen and rejected halves using gather on batch dimension
        SDVariable chosenLogits = sd.gather("_dpo_chosen_logits", logits, chosenIdx, 0);
        SDVariable rejectedLogits = sd.gather("_dpo_rejected_logits", logits, rejectedIdx, 0);

        // Compute per-sequence log probabilities
        SDVariable chosenLP = computeLogProbOps(sd, "_dpo_chosen", chosenLogits, chosenTokens, vocabSize);
        SDVariable rejectedLP = computeLogProbOps(sd, "_dpo_rejected", rejectedLogits, rejectedTokens, vocabSize);

        // Compute DPO loss based on variant
        SDVariable piDiff = chosenLP.sub("_dpo_pi_diff", rejectedLP);
        SDVariable refDiff = refChosenLP.sub("_dpo_ref_diff", refRejectedLP);
        SDVariable rewardDiff = piDiff.sub("_dpo_reward_diff", refDiff);

        SDVariable loss;
        switch (config.getVariant()) {
            case IPO:
                // IPO: (reward_diff - 1/(2*beta))^2
                SDVariable target = sd.constant("_dpo_ipo_target", 1.0 / (2.0 * beta));
                SDVariable diff = rewardDiff.sub("_dpo_ipo_diff", target);
                loss = diff.mul("_dpo_ipo_sq", diff).mean("_dpo_loss");
                break;

            case RDPO:
                // RDPO: same as DPO but with label smoothing
                double eps = config.getLabelSmoothing();
                SDVariable scaled = rewardDiff.mul("_dpo_scaled", beta);
                SDVariable sigm = sd.nn().sigmoid("_dpo_sigmoid", scaled);
                SDVariable logSigm = sd.math().log("_dpo_log_sigmoid", sigm.add("_dpo_eps", 1e-10));
                SDVariable negScaled = scaled.neg("_dpo_neg_scaled");
                SDVariable negSigm = sd.nn().sigmoid("_dpo_neg_sigmoid", negScaled);
                SDVariable negLogSigm = sd.math().log("_dpo_neg_log_sigmoid", negSigm.add("_dpo_neg_eps", 1e-10));
                SDVariable smoothed = logSigm.mul("_dpo_main", 1.0 - eps)
                        .add("_dpo_smooth", negLogSigm.mul("_dpo_smooth_term", eps));
                loss = smoothed.neg("_dpo_neg").mean("_dpo_loss");
                break;

            default: // STANDARD
                SDVariable scaledReward = rewardDiff.mul("_dpo_scaled", beta);
                SDVariable sigmoid = sd.nn().sigmoid("_dpo_sigmoid", scaledReward);
                SDVariable logSigmoid = sd.math().log("_dpo_log_sigmoid", sigmoid.add("_dpo_eps", 1e-10));
                loss = logSigmoid.neg("_dpo_neg").mean("_dpo_loss");
                break;
        }

        return loss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        String inputVar = config.getInputVariable();
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();

        INDArray chosenInput = rawInputs.get(config.getChosenVariable());
        INDArray rejectedInput = rawInputs.get(config.getRejectedVariable());
        int batchSize = (int) chosenInput.size(0);

        // Batch chosen + rejected for single forward pass
        INDArray batchedInput = Nd4j.concat(0, chosenInput, rejectedInput);

        // Batch indices
        INDArray chosenIdx = Nd4j.arange(0, batchSize).castTo(DataType.INT64);
        INDArray rejectedIdx = Nd4j.arange(batchSize, 2 * batchSize).castTo(DataType.INT64);

        // Compute reference model log probabilities
        Map<String, INDArray> chosenRefInputs = Collections.singletonMap(inputVar, chosenInput);
        Map<String, INDArray> rejectedRefInputs = Collections.singletonMap(inputVar, rejectedInput);
        INDArray refChosenLP = computeExternalLogProbs(referenceModel, chosenRefInputs, chosenInput, logitVar, vocabSize);
        INDArray refRejectedLP = computeExternalLogProbs(referenceModel, rejectedRefInputs, rejectedInput, logitVar, vocabSize);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, batchedInput);
        placeholders.put("_dpo_chosen_idx", chosenIdx);
        placeholders.put("_dpo_rejected_idx", rejectedIdx);
        placeholders.put("_dpo_chosen_tokens", chosenInput);
        placeholders.put("_dpo_rejected_tokens", rejectedInput);
        placeholders.put("_dpo_ref_chosen_lp", refChosenLP);
        placeholders.put("_dpo_ref_rejected_lp", refRejectedLP);

        // Pass through any additional placeholders the model needs
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals(config.getChosenVariable())
                    && !entry.getKey().equals(config.getRejectedVariable())) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
