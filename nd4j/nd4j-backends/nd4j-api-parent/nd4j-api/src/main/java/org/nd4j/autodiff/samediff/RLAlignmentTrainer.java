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

package org.nd4j.autodiff.samediff;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.api.OutAndGrad;
import org.nd4j.autodiff.samediff.config.RLAlignmentConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

/**
 * Abstract base trainer for RL alignment methods.
 *
 * Provides training infrastructure that actually performs backpropagation and weight updates:
 * <ol>
 *   <li>Subclasses define their loss as SameDiff ops via {@link #buildLossGraph(SameDiff)}</li>
 *   <li>Subclasses prepare inputs (including external values like ref logprobs) via {@link #prepareInputs(Map)}</li>
 *   <li>{@link #trainStep(Map)} computes gradients via {@code calculateGradientsAndOutputs()} and applies updates</li>
 * </ol>
 *
 * Adam Gibson
 */
@Slf4j
public abstract class RLAlignmentTrainer<C extends RLAlignmentConfig> {

    @Getter
    protected final SameDiff policyModel;

    @Getter
    protected final SameDiff referenceModel;

    @Getter
    protected final C config;

    protected String lossVariableName;
    private Map<String, GradientUpdater> updaters;
    private List<String> trainableParams;
    private boolean trainingInitialized = false;

    protected int totalSteps = 0;
    protected int totalEpochs = 0;

    protected RLAlignmentTrainer(SameDiff policyModel, SameDiff referenceModel, C config) {
        Preconditions.checkNotNull(policyModel, "Policy model must not be null");
        Preconditions.checkNotNull(config, "Config must not be null");
        if (config.isUseReferenceModel()) {
            Preconditions.checkNotNull(referenceModel, "Reference model must not be null when useReferenceModel=true");
        }
        config.validate();

        this.policyModel = policyModel;
        this.referenceModel = referenceModel;
        this.config = config;
    }

    /**
     * Initialize training: build loss graph, create gradient function, initialize updaters.
     * Called lazily on first trainStep.
     */
    private void initTraining() {
        if (trainingInitialized) return;

        // Build loss graph (subclass adds loss ops to the policy model)
        lossVariableName = buildLossGraph(policyModel);
        policyModel.addLossVariable(lossVariableName);

        // Collect trainable parameters
        trainableParams = new ArrayList<>();
        for (SDVariable v : policyModel.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE) {
                trainableParams.add(v.name());
            }
        }

        // Initialize updaters
        IUpdater updaterConfig = new Adam(config.getLearningRate());
        updaters = new LinkedHashMap<>();
        for (String paramName : trainableParams) {
            INDArray param = policyModel.getVariable(paramName).getArr();
            if (param == null) continue;
            long numParams = param.length();
            long stateSize = updaterConfig.stateSize(numParams);
            INDArray stateView = stateSize > 0
                    ? Nd4j.createUninitialized(DataType.FLOAT, 1, stateSize)
                    : Nd4j.empty(DataType.FLOAT);
            GradientUpdater gu = updaterConfig.instantiate(stateView, true);
            updaters.put(paramName, gu);
        }

        trainingInitialized = true;
        log.info("RL training initialized: {} trainable parameters, loss variable: {}",
                trainableParams.size(), lossVariableName);
    }

    /**
     * Perform a single training step with backpropagation and weight updates.
     *
     * @param inputs Raw input data map (method-specific keys)
     * @return The computed loss value
     */
    public double trainStep(Map<String, INDArray> inputs) {
        initTraining();
        totalSteps++;

        // Subclass prepares all placeholders (batches inputs, computes ref logprobs, etc.)
        Map<String, INDArray> placeholders = prepareInputs(inputs);

        // Forward + backward pass
        OutAndGrad oag = policyModel.calculateGradientsAndOutputs(
                placeholders,
                Collections.singletonList(lossVariableName),
                trainableParams
        );

        double lossValue = oag.getOutputs().get(lossVariableName).meanNumber().doubleValue();

        // Apply gradient updates
        for (String paramName : trainableParams) {
            INDArray grad = oag.getGradients().get(paramName);
            if (grad == null) continue;

            // Gradient clipping
            grad = clipGradients(grad);

            // Apply updater (Adam, etc.)
            updaters.get(paramName).applyUpdater(grad, totalSteps, totalEpochs);

            // Update parameter: param -= processed_grad
            policyModel.getVariable(paramName).getArr().subi(grad);
        }

        // EMA update of reference model
        if (config.getReferenceModelEmaDecay() > 0 && referenceModel != null) {
            maybeUpdateReferenceModel();
        }

        return lossValue;
    }

    /**
     * Build the loss computation graph in the policy model.
     * Subclasses add SameDiff ops (placeholders, log_softmax, loss formula, etc.)
     * and return the name of the scalar loss variable.
     *
     * @param sd The policy model's SameDiff instance
     * @return Name of the loss SDVariable
     */
    protected abstract String buildLossGraph(SameDiff sd);

    /**
     * Prepare all placeholder values for a training step.
     * Subclasses compute external values (reference logprobs, rewards, advantages)
     * and return a complete placeholder map including model inputs.
     *
     * @param rawInputs The raw inputs provided by the caller
     * @return Complete placeholder map for calculateGradients
     */
    protected abstract Map<String, INDArray> prepareInputs(Map<String, INDArray> rawInputs);

    /**
     * Build log probability computation ops in the SameDiff graph.
     * Uses log_softmax + one_hot to compute per-sequence log probabilities.
     *
     * @param sd       The SameDiff instance
     * @param prefix   Unique prefix for variable names
     * @param logits   Logit variable [batch, seq, vocab]
     * @param tokenIds Token ID variable [batch, seq] (INT64)
     * @param vocabSize Vocabulary size
     * @return SDVariable of shape [batch] containing per-sequence log probabilities
     */
    protected SDVariable computeLogProbOps(SameDiff sd, String prefix, SDVariable logits,
                                           SDVariable tokenIds, int vocabSize) {
        SDVariable logSM = sd.nn().logSoftmax(prefix + "_logsm", logits, -1);
        SDVariable oh = sd.oneHot(prefix + "_oh", tokenIds, vocabSize);
        oh = oh.castTo(prefix + "_oh_cast", logSM.dataType());
        SDVariable tokenLP = logSM.mul(prefix + "_tokenlp", oh);
        SDVariable sumLast = tokenLP.sum(prefix + "_sumlast", -1);
        return sumLast.sum(prefix + "_seqlp", -1);
    }

    /**
     * Compute log probabilities externally (not in graph) for a model.
     * Used for reference model forward passes where gradients aren't needed.
     *
     * @param model    The model (typically the reference model)
     * @param inputs   Model inputs
     * @param tokens   Token IDs [batch, seq]
     * @param logitVar Name of the logit output variable
     * @param vocabSize Vocabulary size
     * @return Per-sequence log probabilities [batch]
     */
    protected INDArray computeExternalLogProbs(SameDiff model, Map<String, INDArray> inputs,
                                               INDArray tokens, String logitVar, int vocabSize) {
        INDArray logits = model.output(inputs, logitVar).get(logitVar);
        INDArray logSM = Transforms.log(Transforms.softmax(logits, false));

        long batchSize = logits.size(0);
        long seqLen = logits.size(1);
        INDArray result = Nd4j.zeros(DataType.FLOAT, batchSize);

        for (int b = 0; b < batchSize; b++) {
            double sum = 0.0;
            for (int t = 0; t < seqLen; t++) {
                int tok = tokens.getInt(b, t);
                if (tok >= 0 && tok < vocabSize) {
                    sum += logSM.getDouble(b, t, tok);
                }
            }
            result.putScalar(b, sum);
        }
        return result;
    }

    /**
     * Clip gradients by global norm if configured.
     */
    protected INDArray clipGradients(INDArray gradients) {
        if (config.getMaxGradNorm() > 0) {
            double norm = gradients.norm2Number().doubleValue();
            if (norm > config.getMaxGradNorm()) {
                gradients.muli(config.getMaxGradNorm() / norm);
            }
        }
        return gradients;
    }

    /**
     * Normalize advantages to zero mean and unit variance.
     */
    protected INDArray normalizeAdvantages(INDArray advantages) {
        if (!config.isNormalizeAdvantages()) return advantages;
        double mean = advantages.meanNumber().doubleValue();
        double std = advantages.stdNumber().doubleValue();
        if (std < 1e-8) return advantages.sub(mean);
        return advantages.sub(mean).div(std + 1e-8);
    }

    /**
     * Update the reference model using exponential moving average of policy weights.
     */
    protected void maybeUpdateReferenceModel() {
        if (referenceModel == null) return;
        double decay = config.getReferenceModelEmaDecay();
        if (decay <= 0) return;

        for (SDVariable policyVar : policyModel.variables()) {
            if (policyVar.getVariableType() == VariableType.VARIABLE) {
                SDVariable refVar = referenceModel.getVariable(policyVar.name());
                if (refVar != null && refVar.getArr() != null && policyVar.getArr() != null) {
                    INDArray refArr = refVar.getArr();
                    INDArray policyArr = policyVar.getArr();
                    refArr.muli(decay).addi(policyArr.mul(1.0 - decay));
                }
            }
        }
    }

    /**
     * Get forward pass outputs from a model.
     */
    protected Map<String, INDArray> getModelOutputs(SameDiff model, Map<String, INDArray> inputs, String... outputVars) {
        return model.output(inputs, outputVars);
    }
}
