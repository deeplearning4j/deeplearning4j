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
import org.nd4j.autodiff.samediff.config.ORPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Odds Ratio Preference Optimization (ORPO) trainer.
 *
 * ORPO combines SFT loss with an odds ratio penalty to align preferences
 * without needing a reference model. This reduces memory usage compared
 * to DPO since only one model needs to be loaded.
 *
 * Loss (in-graph): sft_loss + lambda * odds_ratio_loss
 * where:
 *   sft_loss = -mean(chosen_log_probs)
 *   log_odds = chosen_lp - rejected_lp
 *   odds_ratio_loss = -mean(log_sigmoid(log_odds))
 *   total = sft_loss + lambda * odds_ratio_loss
 *
 * Uses the same batching trick as DPO: chosen and rejected sequences are
 * concatenated along the batch dimension, then split via gather indices.
 * All computation is in-graph for proper backpropagation.
 *
 * Adam Gibson
 */
@Slf4j
public class ORPOTrainer extends RLAlignmentTrainer<ORPOConfig> {

    public ORPOTrainer(SameDiff policyModel, ORPOConfig config) {
        super(policyModel, null, config);
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double lambda = config.getOrpoLambda();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for batch indices (to split chosen/rejected from batched logits)
        SDVariable chosenIdx = sd.placeHolder("_orpo_chosen_idx", DataType.INT64, -1);
        SDVariable rejectedIdx = sd.placeHolder("_orpo_rejected_idx", DataType.INT64, -1);

        // Placeholders for token IDs (for log prob computation)
        SDVariable chosenTokens = sd.placeHolder("_orpo_chosen_tokens", DataType.INT64, -1, -1);
        SDVariable rejectedTokens = sd.placeHolder("_orpo_rejected_tokens", DataType.INT64, -1, -1);

        // Split logits into chosen and rejected halves using gather on batch dimension
        SDVariable chosenLogits = sd.gather("_orpo_chosen_logits", logits, chosenIdx, 0);
        SDVariable rejectedLogits = sd.gather("_orpo_rejected_logits", logits, rejectedIdx, 0);

        // Compute per-sequence log probabilities in-graph
        SDVariable chosenLP = computeLogProbOps(sd, "_orpo_chosen", chosenLogits, chosenTokens, vocabSize);
        SDVariable rejectedLP = computeLogProbOps(sd, "_orpo_rejected", rejectedLogits, rejectedTokens, vocabSize);

        // SFT loss: -mean(chosen_log_probs) (standard language modeling loss on chosen)
        SDVariable sftLoss = chosenLP.neg("_orpo_neg_chosen").mean("_orpo_sft_loss");

        // Log odds ratio: chosen_lp - rejected_lp
        SDVariable logOdds = chosenLP.sub("_orpo_log_odds", rejectedLP);

        // Odds ratio loss: -log_sigmoid(log_odds)
        SDVariable sigmoidOdds = sd.nn().sigmoid("_orpo_sigmoid_odds", logOdds);
        SDVariable logSigmoidOdds = sd.math().log("_orpo_log_sigmoid",
                sigmoidOdds.add("_orpo_sigmoid_eps", 1e-10));
        SDVariable oddsRatioLoss = logSigmoidOdds.neg("_orpo_neg_log_sigmoid").mean("_orpo_odds_loss");

        // Total loss: sft_loss + lambda * odds_ratio_loss
        SDVariable loss = sftLoss.add("_orpo_loss", oddsRatioLoss.mul("_orpo_lambda_scaled", lambda));

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
        placeholders.put("_orpo_chosen_idx", chosenIdx);
        placeholders.put("_orpo_rejected_idx", rejectedIdx);
        placeholders.put("_orpo_chosen_tokens", chosenInput.castTo(DataType.INT64));
        placeholders.put("_orpo_rejected_tokens", rejectedInput.castTo(DataType.INT64));

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
