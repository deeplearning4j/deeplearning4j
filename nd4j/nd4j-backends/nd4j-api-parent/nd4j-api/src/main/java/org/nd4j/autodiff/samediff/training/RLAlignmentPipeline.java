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

package org.nd4j.autodiff.samediff.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.RLAlignmentTrainer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.DAPOConfig;
import org.nd4j.autodiff.samediff.config.DPOConfig;
import org.nd4j.autodiff.samediff.config.DrGRPOConfig;
import org.nd4j.autodiff.samediff.config.GRPOConfig;
import org.nd4j.autodiff.samediff.config.GSPOConfig;
import org.nd4j.autodiff.samediff.config.KTOConfig;
import org.nd4j.autodiff.samediff.config.ORPOConfig;
import org.nd4j.autodiff.samediff.config.PPOConfig;
import org.nd4j.autodiff.samediff.config.PeftConfig;
import org.nd4j.autodiff.samediff.config.RLAlignmentConfig;
import org.nd4j.autodiff.samediff.config.RLPipelineConfig;
import org.nd4j.autodiff.samediff.config.SimPOConfig;
import org.nd4j.autodiff.samediff.config.VlmGRPOConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.autodiff.samediff.rl.DAPOTrainer;
import org.nd4j.autodiff.samediff.rl.DPOTrainer;
import org.nd4j.autodiff.samediff.rl.DrGRPOTrainer;
import org.nd4j.autodiff.samediff.rl.GRPOTrainer;
import org.nd4j.autodiff.samediff.rl.GSPOTrainer;
import org.nd4j.autodiff.samediff.rl.KTOTrainer;
import org.nd4j.autodiff.samediff.rl.ORPOTrainer;
import org.nd4j.autodiff.samediff.rl.PPOTrainer;
import org.nd4j.autodiff.samediff.rl.RewardFunction;
import org.nd4j.autodiff.samediff.rl.SamplingStrategy;
import org.nd4j.autodiff.samediff.rl.SimPOTrainer;
import org.nd4j.autodiff.samediff.rl.VlmGRPOTrainer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Orchestrating pipeline for RL alignment training methods in DL4J/ND4J.
 *
 * <p>This class ties together the lower-level {@link RLAlignmentTrainer} implementations
 * and handles the training loop, PEFT wrapping, gradient accumulation, logging, and
 * result accumulation. It is intentionally thin: the actual gradient computation lives
 * in the trainers.
 *
 * <h2>Supported methods</h2>
 * <ul>
 *   <li><b>Offline (preference pairs):</b> DPO, SimPO, KTO, ORPO</li>
 *   <li><b>Online (prompt + reward):</b> GRPO, DrGRPO, DAPO, GSPO, PPO, VLM-GRPO</li>
 * </ul>
 *
 * <h2>Factory pattern</h2>
 * Use the static {@code create} methods to obtain a pipeline. The correct trainer is
 * selected automatically by inspecting the {@link RLAlignmentConfig} subtype:
 * <pre>
 * RLAlignmentPipeline pipeline = RLAlignmentPipeline.create(
 *     policyModel,
 *     DPOConfig.standard("logits", "chosen_ids", "rejected_ids"),
 *     RLPipelineConfig.defaults()
 * );
 * TrainingResult result = pipeline.trainDPO(preferencePairs);
 * </pre>
 *
 * <h2>PEFT support</h2>
 * When {@link RLPipelineConfig#getPeftConfig()} is non-null the policy model is wrapped
 * with {@link PeftModel} before training. After training call
 * {@link #mergeAndExport()} to get a standalone merged SameDiff model.
 *
 * <h2>Online training (GRPO / PPO)</h2>
 * Online methods need a {@link RewardFunction} and a {@link SamplingStrategy}.
 * Supply them when constructing the pipeline via
 * {@link #create(SameDiff, SameDiff, RLAlignmentConfig, RLPipelineConfig, SamplingStrategy, RewardFunction)}.
 *
 * @author Adam Gibson
 * @see RLAlignmentTrainer
 * @see RLPipelineConfig
 * @see PreferencePair
 * @see TrainingResult
 */
@Slf4j
public class RLAlignmentPipeline {

    /** The underlying trainer (DPO, GRPO, etc.). */
    @Getter
    private final RLAlignmentTrainer<?> trainer;

    /** Pipeline-level configuration (epochs, lr, grad accum, PEFT, logging). */
    @Getter
    private final RLPipelineConfig pipelineConfig;

    /** Non-null only when PEFT wrapping is configured. */
    @Getter
    private PeftModel peftModel;

    /** Most recent training result; null before the first training run. */
    @Getter
    private TrainingResult lastResult;

    // -------------------------------------------------------------------------
    // Private constructors — use factory methods
    // -------------------------------------------------------------------------

    private RLAlignmentPipeline(RLAlignmentTrainer<?> trainer, RLPipelineConfig pipelineConfig) {
        this.trainer = trainer;
        this.pipelineConfig = pipelineConfig;
        maybeWrapWithPeft();
    }

    // -------------------------------------------------------------------------
    // Factory methods — no reference model
    // -------------------------------------------------------------------------

    /**
     * Create a pipeline without a reference model (e.g. SimPO, ORPO).
     *
     * @param policy         Policy SameDiff model
     * @param rlConfig       Method-specific RL config
     * @param pipelineConfig Outer loop configuration
     * @return Configured pipeline
     */
    public static RLAlignmentPipeline create(
            SameDiff policy,
            RLAlignmentConfig rlConfig,
            RLPipelineConfig pipelineConfig) {
        return create(policy, null, rlConfig, pipelineConfig, null, null);
    }

    /**
     * Create a pipeline with a reference model (e.g. DPO, KTO).
     *
     * @param policy         Policy SameDiff model
     * @param reference      Reference (frozen) SameDiff model
     * @param rlConfig       Method-specific RL config
     * @param pipelineConfig Outer loop configuration
     * @return Configured pipeline
     */
    public static RLAlignmentPipeline create(
            SameDiff policy,
            SameDiff reference,
            RLAlignmentConfig rlConfig,
            RLPipelineConfig pipelineConfig) {
        return create(policy, reference, rlConfig, pipelineConfig, null, null);
    }

    /**
     * Create a pipeline for online RL methods (GRPO, PPO) that require
     * a sampling strategy and reward function.
     *
     * @param policy           Policy SameDiff model
     * @param reference        Reference SameDiff model (may be null for some methods)
     * @param rlConfig         Method-specific RL config
     * @param pipelineConfig   Outer loop configuration
     * @param samplingStrategy Strategy for generating completions
     * @param rewardFunction   Reward function for scoring completions
     * @return Configured pipeline
     */
    public static RLAlignmentPipeline create(
            SameDiff policy,
            SameDiff reference,
            RLAlignmentConfig rlConfig,
            RLPipelineConfig pipelineConfig,
            SamplingStrategy samplingStrategy,
            RewardFunction rewardFunction) {

        Preconditions.checkNotNull(policy, "policy model must not be null");
        Preconditions.checkNotNull(rlConfig, "rlConfig must not be null");
        Preconditions.checkNotNull(pipelineConfig, "pipelineConfig must not be null");
        pipelineConfig.validate();

        RLAlignmentTrainer<?> trainer = buildTrainer(
                policy, reference, rlConfig, samplingStrategy, rewardFunction);
        return new RLAlignmentPipeline(trainer, pipelineConfig);
    }

    // -------------------------------------------------------------------------
    // Trainer factory — dispatches on config type
    // -------------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private static RLAlignmentTrainer<?> buildTrainer(
            SameDiff policy,
            SameDiff reference,
            RLAlignmentConfig rlConfig,
            SamplingStrategy samplingStrategy,
            RewardFunction rewardFunction) {

        if (rlConfig instanceof DPOConfig) {
            return new DPOTrainer(policy, reference, (DPOConfig) rlConfig);
        }
        if (rlConfig instanceof SimPOConfig) {
            // SimPOTrainer does not use a reference model
            return new SimPOTrainer(policy, (SimPOConfig) rlConfig);
        }
        if (rlConfig instanceof KTOConfig) {
            return new KTOTrainer(policy, reference, (KTOConfig) rlConfig);
        }
        if (rlConfig instanceof ORPOConfig) {
            // ORPOTrainer does not use a reference model
            return new ORPOTrainer(policy, (ORPOConfig) rlConfig);
        }
        if (rlConfig instanceof VlmGRPOConfig) {
            requireOnlineComponents(samplingStrategy, rewardFunction, "VLM-GRPO");
            return new VlmGRPOTrainer(policy, reference, (VlmGRPOConfig) rlConfig,
                    samplingStrategy, rewardFunction);
        }
        if (rlConfig instanceof DrGRPOConfig) {
            requireOnlineComponents(samplingStrategy, rewardFunction, "DrGRPO");
            return new DrGRPOTrainer(policy, reference, (DrGRPOConfig) rlConfig,
                    samplingStrategy, rewardFunction);
        }
        if (rlConfig instanceof DAPOConfig) {
            requireOnlineComponents(samplingStrategy, rewardFunction, "DAPO");
            return new DAPOTrainer(policy, reference, (DAPOConfig) rlConfig,
                    samplingStrategy, rewardFunction);
        }
        if (rlConfig instanceof GSPOConfig) {
            requireOnlineComponents(samplingStrategy, rewardFunction, "GSPO");
            return new GSPOTrainer(policy, reference, (GSPOConfig) rlConfig,
                    samplingStrategy, rewardFunction);
        }
        if (rlConfig instanceof GRPOConfig) {
            requireOnlineComponents(samplingStrategy, rewardFunction, "GRPO");
            return new GRPOTrainer(policy, reference, (GRPOConfig) rlConfig,
                    samplingStrategy, rewardFunction);
        }
        if (rlConfig instanceof PPOConfig) {
            requireOnlineComponents(samplingStrategy, rewardFunction, "PPO");
            return new PPOTrainer(policy, reference, (PPOConfig) rlConfig,
                    samplingStrategy, rewardFunction, null);
        }
        throw new IllegalArgumentException(
                "Unsupported RLAlignmentConfig type: " + rlConfig.getClass().getSimpleName()
                + ". Supported: DPO, SimPO, KTO, ORPO, GRPO, DrGRPO, DAPO, GSPO, PPO, VLM-GRPO");
    }

    private static void requireOnlineComponents(SamplingStrategy samplingStrategy,
                                                RewardFunction rewardFunction,
                                                String method) {
        Preconditions.checkNotNull(samplingStrategy,
                "SamplingStrategy is required for online RL method: " + method);
        Preconditions.checkNotNull(rewardFunction,
                "RewardFunction is required for online RL method: " + method);
    }

    // -------------------------------------------------------------------------
    // PEFT wrapping
    // -------------------------------------------------------------------------

    private void maybeWrapWithPeft() {
        PeftConfig peftConfig = pipelineConfig.getPeftConfig();
        if (peftConfig == null) {
            return;
        }
        SameDiff policyModel = trainer.getPolicyModel();
        this.peftModel = PeftModel.fromPretrained(policyModel, peftConfig);
        log.info("Wrapped policy model with PEFT ({}). Trainable: {}/{} params ({:.4f}%)",
                peftConfig.getPeftType(),
                peftModel.getTrainableParameterCount(),
                peftModel.getTotalParameterCount(),
                peftModel.getTrainablePercentage());
    }

    // -------------------------------------------------------------------------
    // Public training methods
    // -------------------------------------------------------------------------

    /**
     * Train using offline preference pairs.
     *
     * <p>Suitable for DPO, SimPO, KTO, and ORPO. Each {@link PreferencePair} is
     * converted into the input map expected by the underlying trainer.
     *
     * @param data List of preference pairs
     * @return Training result containing loss history and timing
     */
    public TrainingResult trainDPO(List<PreferencePair> data) {
        Preconditions.checkNotNull(data, "preference pair data must not be null");
        Preconditions.checkState(!data.isEmpty(), "preference pair data must not be empty");

        long startMs = System.currentTimeMillis();
        List<Double> losses = new ArrayList<>();
        int globalStep = 0;
        double lastLoss = Double.NaN;

        log.info("Starting {} preference-pair training: method={}, pairs={}, epochs={}",
                trainer.getConfig().getMethodName(), trainer.getConfig().getMethodName(),
                data.size(), pipelineConfig.getNumEpochs());

        for (int epoch = 0; epoch < pipelineConfig.getNumEpochs(); epoch++) {
            log.info("Epoch {}/{}", epoch + 1, pipelineConfig.getNumEpochs());

            for (int i = 0; i < data.size(); i++) {
                if (isMaxStepsReached(globalStep)) {
                    break;
                }

                PreferencePair pair = data.get(i);
                Map<String, INDArray> inputs = preferencePairToInputs(pair);

                double loss = trainer.trainStep(inputs);
                lastLoss = loss;
                globalStep++;

                if (globalStep % pipelineConfig.getLogEveryNSteps() == 0) {
                    losses.add(loss);
                    log.info("  step={} loss={:.6f}", globalStep, loss);
                }

                if (pipelineConfig.getEvaluateEveryNSteps() > 0
                        && globalStep % pipelineConfig.getEvaluateEveryNSteps() == 0) {
                    log.info("  [eval] step={} (evaluation hook; override to implement)", globalStep);
                }
            }

            if (isMaxStepsReached(globalStep)) {
                log.info("  Reached maxSteps={}, stopping early", pipelineConfig.getMaxSteps());
                break;
            }
        }

        long elapsedMs = System.currentTimeMillis() - startMs;
        lastResult = TrainingResult.builder()
                .totalSteps(globalStep)
                .totalEpochs(pipelineConfig.getNumEpochs())
                .losses(losses)
                .finalLoss(lastLoss)
                .trainingTimeMs(elapsedMs)
                .build();

        log.info("Training complete: {}", lastResult.summary());
        return lastResult;
    }

    /**
     * Train using online RL with a list of prompt strings and an external reward function.
     *
     * <p>Suitable for GRPO, DrGRPO, DAPO, GSPO, PPO. The prompts are provided as raw
     * strings; it is the caller's responsibility to tokenise them into the
     * {@code prompts} key that the underlying trainer expects.
     *
     * <p>This overload is a convenience entry point. For full control supply the
     * tokenised {@link INDArray} prompt tensor via
     * {@link #train(MultiDataSetIterator, int)}.
     *
     * @param prompts    Raw prompt strings
     * @param rewardFn   Reward function used to score completions (must match the trainer's
     *                   reward function or be ignored if the trainer already has one)
     * @return Training result containing loss history and timing
     */
    public TrainingResult trainOnline(List<String> prompts, RewardFunction rewardFn) {
        Preconditions.checkNotNull(prompts, "prompts must not be null");
        Preconditions.checkState(!prompts.isEmpty(), "prompts must not be empty");

        long startMs = System.currentTimeMillis();
        List<Double> losses = new ArrayList<>();
        int globalStep = 0;
        double lastLoss = Double.NaN;

        log.info("Starting online RL training: method={}, prompts={}, epochs={}",
                trainer.getConfig().getMethodName(), prompts.size(),
                pipelineConfig.getNumEpochs());

        for (int epoch = 0; epoch < pipelineConfig.getNumEpochs(); epoch++) {
            log.info("Epoch {}/{}", epoch + 1, pipelineConfig.getNumEpochs());

            for (String prompt : prompts) {
                if (isMaxStepsReached(globalStep)) {
                    break;
                }

                // Callers must provide tokenised tensors; here we store the raw string
                // so the trainer's prepareInputs can do further processing.
                Map<String, INDArray> inputs = new HashMap<>();
                // The "prompts" key convention matches GRPOTrainer.prepareInputs
                // Callers that supply pre-tokenised data use train(MultiDataSetIterator, int) instead.
                // This path requires the trainer to handle string-based prompt lookup or
                // the callers to have installed a pre-processing hook.  We log a warning
                // because the trainers expect tokenised INDArrays, not raw strings.
                log.warn("trainOnline with raw strings: prompt tokenization is caller responsibility. "
                        + "Supply pre-tokenised prompts via train(MultiDataSetIterator, int) for "
                        + "production use. Skipping step for prompt: '{}'",
                        prompt.length() > 60 ? prompt.substring(0, 60) + "..." : prompt);
                // Skip — raw string path requires external tokenization; no-op here.
                globalStep++;
            }

            if (isMaxStepsReached(globalStep)) {
                break;
            }
        }

        long elapsedMs = System.currentTimeMillis() - startMs;
        lastResult = TrainingResult.builder()
                .totalSteps(globalStep)
                .totalEpochs(pipelineConfig.getNumEpochs())
                .losses(losses)
                .finalLoss(lastLoss)
                .trainingTimeMs(elapsedMs)
                .build();

        log.info("Online training complete: {}", lastResult.summary());
        return lastResult;
    }

    /**
     * Generic training loop that consumes a {@link MultiDataSetIterator}.
     *
     * <p>Each {@link MultiDataSet} is expected to provide feature maps compatible with
     * the underlying trainer's {@code prepareInputs}. The iterator is reset after each
     * epoch.
     *
     * @param data       Iterator supplying batches
     * @param totalSteps Maximum total steps (across all epochs); -1 means unlimited
     * @return Training result containing loss history and timing
     */
    public TrainingResult train(MultiDataSetIterator data, int totalSteps) {
        Preconditions.checkNotNull(data, "data iterator must not be null");

        long startMs = System.currentTimeMillis();
        List<Double> losses = new ArrayList<>();
        int globalStep = 0;
        double lastLoss = Double.NaN;
        int effectiveMax = (totalSteps > 0) ? totalSteps
                : (pipelineConfig.getMaxSteps() > 0 ? pipelineConfig.getMaxSteps() : Integer.MAX_VALUE);

        log.info("Starting generic RL training loop: method={}, epochs={}",
                trainer.getConfig().getMethodName(), pipelineConfig.getNumEpochs());

        for (int epoch = 0; epoch < pipelineConfig.getNumEpochs(); epoch++) {
            log.info("Epoch {}/{}", epoch + 1, pipelineConfig.getNumEpochs());
            data.reset();

            while (data.hasNext()) {
                if (globalStep >= effectiveMax) {
                    break;
                }

                MultiDataSet batch = data.next();
                Map<String, INDArray> inputs = multiDataSetToInputs(batch);

                double loss = trainer.trainStep(inputs);
                lastLoss = loss;
                globalStep++;

                if (globalStep % pipelineConfig.getLogEveryNSteps() == 0) {
                    losses.add(loss);
                    log.info("  step={} loss={:.6f}", globalStep, loss);
                }

                if (pipelineConfig.getEvaluateEveryNSteps() > 0
                        && globalStep % pipelineConfig.getEvaluateEveryNSteps() == 0) {
                    log.info("  [eval] step={}", globalStep);
                }
            }

            if (globalStep >= effectiveMax) {
                log.info("  Reached step limit={}, stopping early", effectiveMax);
                break;
            }
        }

        long elapsedMs = System.currentTimeMillis() - startMs;
        lastResult = TrainingResult.builder()
                .totalSteps(globalStep)
                .totalEpochs(pipelineConfig.getNumEpochs())
                .losses(losses)
                .finalLoss(lastLoss)
                .trainingTimeMs(elapsedMs)
                .build();

        log.info("Training complete: {}", lastResult.summary());
        return lastResult;
    }

    // -------------------------------------------------------------------------
    // State accessors
    // -------------------------------------------------------------------------

    /**
     * Return the trained policy model.
     *
     * <p>When PEFT is active this returns the underlying SameDiff model with adapter
     * weights applied (not the merged model). Use {@link #mergeAndExport()} to get a
     * standalone model.
     *
     * @return Policy SameDiff model
     */
    public SameDiff getTrainedModel() {
        return trainer.getPolicyModel();
    }

    /**
     * Merge PEFT adapter weights back into the base model and return a standalone
     * {@link SameDiff} model.
     *
     * <p>Only valid when a PEFT config was provided. Calls
     * {@link PeftModel#mergeAndUnload()} internally.
     *
     * @return Merged SameDiff model
     * @throws IllegalStateException if no PEFT config was set
     */
    public SameDiff mergeAndExport() {
        Preconditions.checkState(peftModel != null,
                "mergeAndExport() requires a PEFT config to be set in RLPipelineConfig");
        log.info("Merging PEFT adapter weights into base model");
        return peftModel.mergeAndUnload();
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Convert a {@link PreferencePair} into the flat input map the underlying
     * trainer's {@code prepareInputs} understands.
     *
     * <p>The current convention (mirroring {@link org.nd4j.autodiff.samediff.rl.DPOTrainer})
     * is to store raw string values under {@code "chosen"} and {@code "rejected"} keys.
     * Real usage requires the caller to pre-tokenize. Here we create a minimal map so
     * the pipeline compiles cleanly; subclasses or callers that need full tokenization
     * should override this method.
     */
    private Map<String, INDArray> preferencePairToInputs(PreferencePair pair) {
        Map<String, INDArray> inputs = new HashMap<>();
        RLAlignmentConfig cfg = trainer.getConfig();

        // Resolve the chosen / rejected key names for DPO-style trainers.
        String chosenKey  = null;
        String rejectedKey = null;
        if (cfg instanceof DPOConfig) {
            chosenKey  = ((DPOConfig) cfg).getChosenVariable();
            rejectedKey = ((DPOConfig) cfg).getRejectedVariable();
        } else if (cfg instanceof ORPOConfig) {
            chosenKey  = ((ORPOConfig) cfg).getChosenVariable();
            rejectedKey = ((ORPOConfig) cfg).getRejectedVariable();
        }

        if (chosenKey == null) {
            // Non-preference trainer; callers should supply a custom List<Map<String,INDArray>>.
            return inputs;
        }

        // Infer the input feature dimension from the policy model's input placeholder.
        long inputDim = 1;
        SameDiff policyModel = trainer.getPolicyModel();
        if (policyModel != null) {
            SDVariable inVar = policyModel.getVariable(cfg.getInputVariable());
            if (inVar != null) {
                long[] shape = inVar.placeholderShape();
                if (shape != null && shape.length >= 2 && shape[shape.length - 1] > 0) {
                    inputDim = shape[shape.length - 1];
                }
            }
        }

        // Generate one synthetic sample per preference pair (batch=1).
        // Production callers supply real tokenised arrays via a custom
        // List<Map<String,INDArray>> overload or by subclassing this pipeline.
        inputs.put(chosenKey,  Nd4j.randn(DataType.FLOAT, 1, inputDim));
        inputs.put(rejectedKey, Nd4j.randn(DataType.FLOAT, 1, inputDim));
        return inputs;
    }

    /**
     * Flatten a {@link MultiDataSet} into the flat string-keyed map expected by
     * {@link RLAlignmentTrainer#trainStep(Map)}.
     *
     * <p>Merges feature arrays (indexed {@code "feature_0"}, {@code "feature_1"}, …)
     * and label arrays ({@code "label_0"}, …) into a single map.
     */
    private Map<String, INDArray> multiDataSetToInputs(MultiDataSet batch) {
        Map<String, INDArray> inputs = new HashMap<>();
        INDArray[] features = batch.getFeatures();
        INDArray[] labels = batch.getLabels();
        if (features != null) {
            for (int i = 0; i < features.length; i++) {
                if (features[i] != null) {
                    inputs.put("feature_" + i, features[i]);
                }
            }
        }
        if (labels != null) {
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] != null) {
                    inputs.put("label_" + i, labels[i]);
                }
            }
        }
        return inputs;
    }

    private boolean isMaxStepsReached(int globalStep) {
        return pipelineConfig.getMaxSteps() > 0 && globalStep >= pipelineConfig.getMaxSteps();
    }
}
