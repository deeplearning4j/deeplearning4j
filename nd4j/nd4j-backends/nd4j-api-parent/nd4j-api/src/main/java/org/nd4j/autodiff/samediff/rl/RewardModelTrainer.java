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
import org.nd4j.autodiff.samediff.config.RewardModelConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Reward model trainer for preference-based learning.
 *
 * Trains a reward model on preference pairs (chosen vs rejected) using one of
 * three objectives, all computed in-graph for proper backpropagation:
 * - BRADLEY_TERRY: -log sigmoid(r_chosen - r_rejected - margin).mean()
 * - REGRESSION: (chosen_reward - target_score)^2.mean()
 * - CONTRASTIVE: max(0, margin - (r_chosen - r_rejected)).mean()
 *
 * The model outputs REWARD SCORES (not logits). Uses config.getRewardOutputVariable()
 * to access the model's reward head output.
 *
 * Uses the same batching trick as DPO: chosen and rejected sequences are
 * concatenated along the batch dimension, then split via gather indices.
 * No reference model is needed.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class RewardModelTrainer extends RLAlignmentTrainer<RewardModelConfig> {

    public RewardModelTrainer(SameDiff policyModel, RewardModelConfig config) {
        super(policyModel, null, config);
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String rewardVar = config.getRewardOutputVariable();
        double margin = config.getMargin();

        SDVariable rewards = sd.getVariable(rewardVar);

        // Placeholders for batch indices (to split chosen/rejected from batched rewards)
        SDVariable chosenIdx = sd.placeHolder("_rm_chosen_idx", DataType.INT64, -1);
        SDVariable rejectedIdx = sd.placeHolder("_rm_rejected_idx", DataType.INT64, -1);

        // Split rewards into chosen and rejected using gather on batch dimension
        SDVariable chosenReward = sd.gather("_rm_chosen_reward", rewards, chosenIdx, 0);
        SDVariable rejectedReward = sd.gather("_rm_rejected_reward", rewards, rejectedIdx, 0);

        SDVariable loss;
        switch (config.getRewardType()) {
            case REGRESSION: {
                // Regression: (chosen_reward - target_score)^2.mean()
                SDVariable targetScores = sd.placeHolder("_rm_target_scores", DataType.FLOAT, -1);
                SDVariable residual = chosenReward.sub("_rm_residual", targetScores);
                loss = residual.mul("_rm_sq", residual).mean("_rm_loss");
                break;
            }

            case CONTRASTIVE: {
                // Contrastive: max(0, margin - (r_chosen - r_rejected)).mean()
                SDVariable rewardDiff = chosenReward.sub("_rm_diff", rejectedReward);
                SDVariable marginConst = sd.constant("_rm_margin", margin);
                SDVariable hingeTerm = marginConst.sub("_rm_hinge_arg", rewardDiff);
                SDVariable zero = sd.constant("_rm_zero", 0.0);
                // max(0, x) via relu-like: (x + abs(x)) / 2 = max(0, x)
                SDVariable absHinge = sd.math().abs("_rm_abs_hinge", hingeTerm);
                SDVariable maxTerm = hingeTerm.add("_rm_sum_abs", absHinge).div("_rm_relu", 2.0);
                loss = maxTerm.mean("_rm_loss");
                break;
            }

            case BRADLEY_TERRY:
            default: {
                // Bradley-Terry: -log sigmoid(r_chosen - r_rejected - margin).mean()
                SDVariable rewardDiff = chosenReward.sub("_rm_diff", rejectedReward);
                SDVariable adjusted;
                if (margin != 0.0) {
                    adjusted = rewardDiff.sub("_rm_margin_adj", margin);
                } else {
                    adjusted = rewardDiff;
                }
                SDVariable sigmoid = sd.nn().sigmoid("_rm_sigmoid", adjusted);
                SDVariable logSigmoid = sd.math().log("_rm_log_sigmoid",
                        sigmoid.add("_rm_eps", 1e-10));
                loss = logSigmoid.neg("_rm_neg").mean("_rm_loss");
                break;
            }
        }

        return loss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        String inputVar = config.getInputVariable();

        INDArray chosenInput = rawInputs.get(config.getChosenVariable());
        INDArray rejectedInput = rawInputs.get(config.getRejectedVariable());
        int batchSize = (int) chosenInput.size(0);

        // Batch chosen + rejected for single forward pass
        INDArray batchedInput = Nd4j.concat(0, chosenInput, rejectedInput);

        // Batch indices
        INDArray chosenIdx = Nd4j.arange(0, batchSize).castTo(DataType.INT64);
        INDArray rejectedIdx = Nd4j.arange(batchSize, 2 * batchSize).castTo(DataType.INT64);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, batchedInput);
        placeholders.put("_rm_chosen_idx", chosenIdx);
        placeholders.put("_rm_rejected_idx", rejectedIdx);

        // For REGRESSION mode, pass through target scores
        if (config.getRewardType() == RewardModelConfig.RewardType.REGRESSION) {
            INDArray targetScores = rawInputs.get("target_scores");
            if (targetScores != null) {
                placeholders.put("_rm_target_scores", targetScores.castTo(DataType.FLOAT));
            }
        }

        // Pass through any additional placeholders the model needs
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals(config.getChosenVariable())
                    && !entry.getKey().equals(config.getRejectedVariable())
                    && !entry.getKey().equals("target_scores")) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
