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
import org.nd4j.autodiff.samediff.config.KTOConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Kahneman-Tversky Optimization (KTO) trainer.
 *
 * KTO aligns models using individual examples labeled as desirable or undesirable.
 * Unlike DPO which requires paired preferences, KTO works with unpaired binary feedback.
 *
 * Loss computation (in-graph):
 * - For desirable samples:   loss_aversion * (1 - sigmoid(beta_d * (pi_lp - ref_lp - kl_ref)))
 * - For undesirable samples: 1 - sigmoid(beta_u * (kl_ref - pi_lp + ref_lp))
 * - Final loss = mean over batch of desirability-weighted combination
 *
 * The loss graph uses SameDiff ops so gradients flow through the policy model
 * for proper backpropagation. Reference model log probs and KL are computed
 * externally (no gradients needed for the reference model).
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class KTOTrainer extends RLAlignmentTrainer<KTOConfig> {

    public KTOTrainer(SameDiff policyModel, SameDiff referenceModel, KTOConfig config) {
        super(policyModel, referenceModel, config);
    }

    @Override
    protected String buildLossGraph(SameDiff sd) {
        String logitVar = config.getPolicyLogitVariable();
        int vocabSize = config.getVocabSize();
        double betaD = config.getBetaDesirable();
        double betaU = config.getBetaUndesirable();
        double lossAversion = config.getLossAversion();

        SDVariable logits = sd.getVariable(logitVar);

        // Placeholders for externally computed values
        SDVariable desirability = sd.placeHolder("_kto_desirability", DataType.FLOAT, -1);
        SDVariable refLP = sd.placeHolder("_kto_ref_lp", DataType.FLOAT, -1);
        SDVariable tokens = sd.placeHolder("_kto_tokens", DataType.INT64, -1, -1);
        SDVariable klRef = sd.placeHolder("_kto_kl_ref", DataType.FLOAT, -1);

        // Compute policy log probabilities in-graph (gradients flow through here)
        SDVariable piLP = computeLogProbOps(sd, "_kto_pi", logits, tokens, vocabSize);

        // Desirable branch: w_d = sigmoid(beta_d * (pi_lp - ref_lp - kl_ref))
        // loss_d = loss_aversion * (1 - w_d)
        SDVariable dArg = piLP.sub("_kto_d_diff", refLP).sub("_kto_d_kl", klRef).mul("_kto_d_scaled", betaD);
        SDVariable wD = sd.nn().sigmoid("_kto_w_d", dArg);
        SDVariable lossD = sd.constant("_kto_one_d", 1.0).sub("_kto_one_minus_wd", wD).mul("_kto_loss_d", lossAversion);

        // Undesirable branch: w_u = sigmoid(beta_u * (kl_ref - pi_lp + ref_lp))
        // loss_u = 1 - w_u
        SDVariable uArg = klRef.sub("_kto_u_diff", piLP).add("_kto_u_ref", refLP).mul("_kto_u_scaled", betaU);
        SDVariable wU = sd.nn().sigmoid("_kto_w_u", uArg);
        SDVariable lossU = sd.constant("_kto_one_u", 1.0).sub("_kto_one_minus_wu", wU);

        // Combine: desirability * loss_d + (1 - desirability) * loss_u
        SDVariable oneMinusDesirability = sd.constant("_kto_one_label", 1.0).sub("_kto_inv_label", desirability);
        SDVariable combinedLoss = desirability.mul("_kto_des_term", lossD)
                .add("_kto_combined", oneMinusDesirability.mul("_kto_undes_term", lossU));

        // Mean over batch
        SDVariable loss = combinedLoss.mean("_kto_loss");

        return loss.name();
    }

    @Override
    protected Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs) {
        String inputVar = config.getInputVariable();
        String logitVar = config.getPolicyLogitVariable();
        String desirabilityVar = config.getDesirabilityVariable();
        int vocabSize = config.getVocabSize();

        INDArray inputTokens = rawInputs.get(inputVar);
        INDArray desirabilityLabels = rawInputs.get(desirabilityVar);

        // Compute reference model log probabilities externally (no gradients needed)
        Map<String, INDArray> refInputs = Collections.singletonMap(inputVar, inputTokens);
        INDArray refLP = computeExternalLogProbs(referenceModel, refInputs, inputTokens, logitVar, vocabSize);

        // Compute KL divergence externally: need policy log probs too (external, no grad)
        INDArray piLP = computeExternalLogProbs(policyModel, Collections.singletonMap(inputVar, inputTokens),
                inputTokens, logitVar, vocabSize);
        INDArray logRatios = piLP.sub(refLP);
        INDArray expLogRatio = Transforms.exp(logRatios);
        INDArray klPerSample = expLogRatio.sub(1.0).sub(logRatios);
        // KL reference is the batch mean, broadcast to each sample
        double klMean = klPerSample.meanNumber().doubleValue();
        INDArray klRefArray = Nd4j.valueArrayOf(new long[]{inputTokens.size(0)}, klMean, DataType.FLOAT);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(inputVar, inputTokens);
        placeholders.put("_kto_desirability", desirabilityLabels.castTo(DataType.FLOAT));
        placeholders.put("_kto_ref_lp", refLP);
        placeholders.put("_kto_tokens", inputTokens.castTo(DataType.INT64));
        placeholders.put("_kto_kl_ref", klRefArray);

        // Pass through any additional placeholders the model needs
        for (Map.Entry<String, INDArray> entry : rawInputs.entrySet()) {
            if (!entry.getKey().equals(inputVar) && !entry.getKey().equals(desirabilityVar)) {
                placeholders.putIfAbsent(entry.getKey(), entry.getValue());
            }
        }

        return placeholders;
    }
}
