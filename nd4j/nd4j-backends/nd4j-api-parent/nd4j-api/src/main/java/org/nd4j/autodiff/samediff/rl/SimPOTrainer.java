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
import org.nd4j.autodiff.samediff.config.SimPOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Simple Preference Optimization (SimPO) trainer.
 *
 * SimPO (arXiv:2405.14734) eliminates the reference model by using length-normalized
 * log probabilities and a target reward margin:
 *
 * Loss = -log sigmoid(beta * (avg_log_prob_chosen - avg_log_prob_rejected - gamma))
 *
 * Where avg_log_prob = sum(log_probs) / sequence_length.
 *
 * The policy model processes chosen and rejected sequences in a single batched forward pass.
 * Chosen sequences occupy the first half of the batch, rejected the second half.
 * Length normalization is applied using placeholder sequence lengths.
 *
 * No reference model is required, reducing memory usage compared to DPO.
 *
 * Adam Gibson
 */
public class SimPOTrainer extends RLAlignmentTrainer<SimPOConfig> {

    public SimPOTrainer(SameDiff policyModel, SimPOConfig config) {
        super(policyModel, null, config);
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double beta = config.getBeta();
        double gamma = config.getGamma();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for batch indices (to split chosen/rejected from batched logits)
        SDVariable chosenIdx = sd.placeHolder("_simpo_chosen_idx", DataType.INT64, -1);
        SDVariable rejectedIdx = sd.placeHolder("_simpo_rejected_idx", DataType.INT64, -1);

        // Placeholders for token IDs (for log prob computation)
        SDVariable chosenTokens = sd.placeHolder("_simpo_chosen_tokens", DataType.INT64, -1, -1);
        SDVariable rejectedTokens = sd.placeHolder("_simpo_rejected_tokens", DataType.INT64, -1, -1);

        // Placeholders for sequence lengths (for length normalization)
        SDVariable chosenLen = sd.placeHolder("_simpo_chosen_len", DataType.FLOAT, -1);
        SDVariable rejectedLen = sd.placeHolder("_simpo_rejected_len", DataType.FLOAT, -1);

        // Split logits into chosen and rejected halves using gather on batch dimension
        SDVariable chosenLogits = sd.gather("_simpo_chosen_logits", logits, chosenIdx, 0);
        SDVariable rejectedLogits = sd.gather("_simpo_rejected_logits", logits, rejectedIdx, 0);

        // Compute per-sequence sum of log probabilities in-graph
        SDVariable chosenSumLP = computeLogProbOps(sd, "_simpo_chosen", chosenLogits, chosenTokens, vocabSize);
        SDVariable rejectedSumLP = computeLogProbOps(sd, "_simpo_rejected", rejectedLogits, rejectedTokens, vocabSize);

        // Length normalization: avg_log_prob = sum_log_prob / length
        SDVariable chosenAvgLP = chosenSumLP.div("_simpo_chosen_avg_lp", chosenLen);
        SDVariable rejectedAvgLP = rejectedSumLP.div("_simpo_rejected_avg_lp", rejectedLen);

        // SimPO reward difference with target margin:
        // reward_diff = beta * (avg_chosen_lp - avg_rejected_lp - gamma)
        SDVariable lpDiff = chosenAvgLP.sub("_simpo_lp_diff", rejectedAvgLP);
        SDVariable marginDiff = lpDiff.sub("_simpo_margin_diff", gamma);
        SDVariable scaled = marginDiff.mul("_simpo_scaled", beta);

        // Loss = -log(sigmoid(scaled)) = mean over batch
        SDVariable sigmoid = sd.nn().sigmoid("_simpo_sigmoid", scaled);
        SDVariable logSigmoid = sd.math().log("_simpo_log_sigmoid", sigmoid.add("_simpo_eps", 1e-10));
        SDVariable loss = logSigmoid.neg("_simpo_neg").mean("_simpo_loss");

        return loss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        String inputVar = config.getInputVariable();

        INDArray chosenInput = rawInputs.get(config.getChosenVariable());
        INDArray rejectedInput = rawInputs.get(config.getRejectedVariable());
        int batchSize = (int) chosenInput.size(0);
        int chosenSeqLen = (int) chosenInput.size(1);
        int rejectedSeqLen = (int) rejectedInput.size(1);

        // Batch chosen + rejected for single forward pass
        INDArray batchedInput = Nd4j.concat(0, chosenInput, rejectedInput);

        // Batch indices
        INDArray chosenIdx = Nd4j.arange(0, batchSize).castTo(DataType.INT64);
        INDArray rejectedIdx = Nd4j.arange(batchSize, 2 * batchSize).castTo(DataType.INT64);

        // Sequence lengths for length normalization (same for all items in the batch)
        INDArray chosenLen = Nd4j.ones(DataType.FLOAT, batchSize).muli(chosenSeqLen);
        INDArray rejectedLen = Nd4j.ones(DataType.FLOAT, batchSize).muli(rejectedSeqLen);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, batchedInput);
        placeholders.put("_simpo_chosen_idx", chosenIdx);
        placeholders.put("_simpo_rejected_idx", rejectedIdx);
        placeholders.put("_simpo_chosen_tokens", chosenInput.castTo(DataType.INT64));
        placeholders.put("_simpo_rejected_tokens", rejectedInput.castTo(DataType.INT64));
        placeholders.put("_simpo_chosen_len", chosenLen);
        placeholders.put("_simpo_rejected_len", rejectedLen);

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
