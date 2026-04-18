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

package org.eclipse.deeplearning4j.vlm.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.model.VisionLanguageModel;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.VlmFineTuneConfig;
import org.nd4j.autodiff.samediff.config.VlmTrainingConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;

import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates Vision-Language Model (VLM) supervised fine-tuning.
 *
 * <p>VLM fine-tuning extends text-only SFT to multimodal models by handling:</p>
 * <ul>
 *   <li>Selective freezing of the vision encoder (standard practice: encoder frozen,
 *       projector and LLM backbone updated)</li>
 *   <li>Optional LoRA injection into the LLM decoder and/or vision encoder via
 *       {@link PeftModel}</li>
 *   <li>Training data that contains both image features and text tokens</li>
 * </ul>
 *
 * <h2>Usage — from a VisionLanguageModel</h2>
 * <pre>{@code
 * VisionLanguageModel vlm = VisionLanguageModel.fromDirectory(new File("model-dir"));
 *
 * VlmFineTuneConfig vlmConfig = VlmFineTuneConfig.loraOnly();   // frozen encoder + LoRA on LLM
 * VlmTrainingConfig trainingConfig = VlmTrainingConfig.loraTraining();
 *
 * VlmTrainingPipeline pipeline = VlmTrainingPipeline.fromVisionLanguageModel(
 *     vlm, vlmConfig, trainingConfig);
 *
 * List<VlmTrainingExample> examples = loadExamples();
 * pipeline.train(examples);
 *
 * SameDiff merged = pipeline.mergeAndExport();  // merge LoRA adapters if used
 * }</pre>
 *
 * <h2>Usage — from raw SameDiff graphs</h2>
 * <pre>{@code
 * VlmTrainingPipeline pipeline = VlmTrainingPipeline.create(
 *     visionEncoderSd, decoderSd, vlmConfig, trainingConfig);
 * pipeline.train(dataIterator, totalSteps);
 * }</pre>
 *
 * @author Adam Gibson
 * @see VlmFineTuneConfig
 * @see VlmTrainingConfig
 * @see VlmTrainingExample
 * @see PeftModel
 */
@Slf4j
public class VlmTrainingPipeline {

    /**
     * Prefix used to identify vision encoder variables by name.
     * Variables whose name starts with this prefix are subject to freezing when
     * {@link VlmFineTuneConfig#isFreezeVisionEncoder()} is true.
     */
    public static final String VISION_ENCODER_PREFIX = "vision_encoder";

    /**
     * Prefix used to identify projector variables by name.
     * These remain trainable when {@link VlmFineTuneConfig#isTrainProjector()} is true.
     */
    public static final String PROJECTOR_PREFIX = "projector";

    // -------------------------------------------------------------------------
    // Core fields
    // -------------------------------------------------------------------------

    /** Vision encoder SameDiff graph. May be null for decoder-only pipelines. */
    @Getter
    private final SameDiff visionEncoder;

    /**
     * The decoder model that is actually trained.
     * When PEFT is applied this is {@link PeftModel#getModel()};
     * otherwise it is the raw decoder SameDiff passed to the factory.
     */
    @Getter
    private final SameDiff decoderModel;

    /** Fine-tune configuration controlling which components are trained. */
    @Getter
    private final VlmFineTuneConfig vlmConfig;

    /** Training hyper-parameters (lr, schedule, dtype, etc.). */
    @Getter
    private final VlmTrainingConfig trainingConfig;

    /**
     * PEFT wrapper around the decoder, non-null only when
     * {@link VlmFineTuneConfig#getLlmLoraConfig()} is set.
     */
    @Getter
    private final PeftModel peftModel;

    /**
     * PEFT wrapper around the vision encoder, non-null only when
     * {@link VlmFineTuneConfig#getVisionLoraConfig()} is set (and
     * {@link VlmFineTuneConfig#isFreezeVisionEncoder()} is false).
     */
    @Getter
    private final PeftModel visionPeftModel;

    // -------------------------------------------------------------------------
    // Private constructor
    // -------------------------------------------------------------------------

    private VlmTrainingPipeline(
            SameDiff visionEncoder,
            SameDiff rawDecoder,
            VlmFineTuneConfig vlmConfig,
            VlmTrainingConfig trainingConfig) {

        this.visionEncoder = visionEncoder;
        this.vlmConfig = vlmConfig;
        this.trainingConfig = trainingConfig;

        // Freeze vision encoder if configured
        if (visionEncoder != null && vlmConfig.isFreezeVisionEncoder()) {
            freezeVisionEncoder(visionEncoder);
        }

        // Apply LoRA to vision encoder when configured (only valid when not frozen)
        if (visionEncoder != null
                && !vlmConfig.isFreezeVisionEncoder()
                && vlmConfig.getVisionLoraConfig() != null) {
            this.visionPeftModel = PeftModel.fromPretrained(visionEncoder, vlmConfig.getVisionLoraConfig());
            log.info("VLM pipeline: vision encoder LoRA applied. Trainable: {}/{} ({} %)",
                    visionPeftModel.getTrainableParameterCount(),
                    visionPeftModel.getTotalParameterCount(),
                    String.format("%.2f", visionPeftModel.getTrainablePercentage()));
        } else {
            this.visionPeftModel = null;
        }

        // Apply LoRA to decoder when configured
        if (vlmConfig.getLlmLoraConfig() != null) {
            this.peftModel = PeftModel.fromPretrained(rawDecoder, vlmConfig.getLlmLoraConfig());
            this.decoderModel = peftModel.getModel();
            log.info("VLM pipeline: decoder LoRA applied. Trainable: {}/{} ({} %)",
                    peftModel.getTrainableParameterCount(),
                    peftModel.getTotalParameterCount(),
                    String.format("%.2f", peftModel.getTrainablePercentage()));
        } else {
            this.peftModel = null;
            this.decoderModel = rawDecoder;
            log.info("VLM pipeline: decoder full fine-tune (no PEFT)");
        }
    }

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Create a VLM training pipeline from raw SameDiff graphs.
     *
     * @param visionEncoder  Vision encoder SameDiff graph. May be null for decoder-only training.
     * @param decoder        LLM decoder SameDiff graph to train.
     * @param vlmConfig      Fine-tune configuration (which components to train, LoRA settings).
     * @param trainingConfig Training hyper-parameter configuration.
     * @return Configured VlmTrainingPipeline ready to call {@link #train}.
     */
    public static VlmTrainingPipeline create(
            SameDiff visionEncoder,
            SameDiff decoder,
            VlmFineTuneConfig vlmConfig,
            VlmTrainingConfig trainingConfig) {

        vlmConfig.validate();
        trainingConfig.validate();
        return new VlmTrainingPipeline(visionEncoder, decoder, vlmConfig, trainingConfig);
    }

    /**
     * Create a VLM training pipeline from an existing {@link VisionLanguageModel}.
     *
     * <p>The pipeline extracts the decoder (and optionally the vision encoder) from
     * the VLM and configures training according to {@code vlmConfig}.</p>
     *
     * @param vlm            The pre-loaded VisionLanguageModel.
     * @param vlmConfig      Fine-tune configuration.
     * @param trainingConfig Training hyper-parameter configuration.
     * @return Configured VlmTrainingPipeline.
     */
    public static VlmTrainingPipeline fromVisionLanguageModel(
            VisionLanguageModel vlm,
            VlmFineTuneConfig vlmConfig,
            VlmTrainingConfig trainingConfig) {

        vlmConfig.validate();
        trainingConfig.validate();
        return new VlmTrainingPipeline(
                vlm.getVisionEncoder(),
                vlm.getDecoder(),
                vlmConfig,
                trainingConfig);
    }

    // -------------------------------------------------------------------------
    // Training — public API
    // -------------------------------------------------------------------------

    /**
     * Train using an externally-constructed {@link MultiDataSetIterator}.
     *
     * <p>Each {@link org.nd4j.linalg.dataset.api.MultiDataSet} should contain at minimum:
     * <ul>
     *   <li>feature[0] — {@code image_features} (float, [batch, numPatches, featureDim]
     *       or [batch, channels, height, width])</li>
     *   <li>feature[1] — {@code input_ids} (long, [batch, seqLen])</li>
     *   <li>label[0]   — {@code labels} (long, [batch, seqLen])</li>
     *   <li>featureMask[0] — {@code loss_mask} (float, [batch, seqLen], 1=train 0=ignore)</li>
     * </ul>
     *
     * @param data       MultiDataSetIterator providing training batches.
     * @param totalSteps Total number of optimiser update steps (used for the LR schedule).
     */
    public void train(MultiDataSetIterator data, int totalSteps) {
        TrainingConfig tc = buildTrainingConfig(totalSteps);
        decoderModel.setTrainingConfig(tc);

        log.info("Starting VLM training: lr={}, warmup={}, epochs={}, gradAccum={}, dtype={}",
                trainingConfig.getLearningRate(),
                trainingConfig.getWarmupRatio(),
                trainingConfig.getNumEpochs(),
                trainingConfig.getGradientAccumulationSteps(),
                trainingConfig.getComputeDataType());

        for (int epoch = 0; epoch < trainingConfig.getNumEpochs(); epoch++) {
            log.info("VLM training epoch {}/{}", epoch + 1, trainingConfig.getNumEpochs());
            decoderModel.fit(data, 1);
            data.reset();
        }

        log.info("VLM training complete");
    }

    /**
     * Convenience method: train from a list of {@link VlmTrainingExample} objects.
     *
     * <p>Each example is converted into a {@link MultiDataSet} with image features,
     * text token IDs, and a response-only loss mask. The list is wrapped in a
     * simple in-memory iterator before calling {@link #train(MultiDataSetIterator, int)}.</p>
     *
     * @param examples List of VLM training examples.
     */
    public void train(List<VlmTrainingExample> examples) {
        if (examples == null || examples.isEmpty()) {
            log.warn("VlmTrainingPipeline.train() called with empty example list; nothing to train");
            return;
        }
        MultiDataSetIterator iterator = buildIteratorFromExamples(examples);
        int totalSteps = examples.size() * trainingConfig.getNumEpochs()
                / Math.max(1, trainingConfig.getGradientAccumulationSteps());
        train(iterator, Math.max(1, totalSteps));
    }

    // -------------------------------------------------------------------------
    // Model management
    // -------------------------------------------------------------------------

    /**
     * Return the trained decoder model.
     *
     * <p>When LoRA was applied this returns the PEFT-modified model (adapters still
     * separate). Call {@link #mergeAndExport()} to obtain a standalone model with
     * adapters merged into the base weights.</p>
     *
     * @return The trained decoder {@link SameDiff}.
     */
    public SameDiff getTrainedDecoder() {
        return decoderModel;
    }

    /**
     * Merge LoRA adapter weights back into the base decoder and return a standalone
     * {@link SameDiff} model ready for inference without adapter overhead.
     *
     * <p>If no LoRA was configured the trained decoder is returned as-is.</p>
     *
     * @return Merged decoder, or the base decoder model if PEFT was not configured.
     */
    public SameDiff mergeAndExport() {
        if (peftModel == null) {
            log.info("mergeAndExport: no decoder PEFT configured, returning decoder as-is");
            return decoderModel;
        }
        SameDiff merged = peftModel.mergeAndUnload();
        log.info("mergeAndExport: LoRA layers merged into decoder base model");
        return merged;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Freeze all VARIABLE-type weights in the vision encoder SameDiff graph
     * by converting them to CONSTANT type. Constants receive no gradient updates.
     *
     * @param encoder The vision encoder SameDiff graph to freeze.
     */
    private void freezeVisionEncoder(SameDiff encoder) {
        List<SDVariable> toFreeze = new ArrayList<>();
        for (SDVariable v : encoder.variables()) {
            if (v.getVariableType() == VariableType.VARIABLE) {
                toFreeze.add(v);
            }
        }
        if (!toFreeze.isEmpty()) {
            encoder.convertToConstants(toFreeze);
            log.info("VLM pipeline: froze {} vision encoder variables", toFreeze.size());
        }
    }

    /**
     * Build a {@link TrainingConfig} from the current {@link VlmTrainingConfig}.
     *
     * @param totalSteps Total optimiser update steps (for LR schedule).
     * @return Fully configured TrainingConfig.
     */
    public TrainingConfig buildTrainingConfig(int totalSteps) {
        CosineWarmupSchedule lrSchedule = CosineWarmupSchedule.fromRatio(
                trainingConfig.getLearningRate(),
                trainingConfig.getMinLearningRate(),
                trainingConfig.getWarmupRatio(),
                totalSteps);

        Adam adam = Adam.builder()
                .learningRateSchedule(lrSchedule)
                .beta1(0.9)
                .beta2(0.999)
                .epsilon(1e-8)
                .build();

        TrainingConfig.Builder builder = TrainingConfig.builder()
                .updater(adam)
                .dataSetFeatureMapping("image_features", "input_ids")
                .dataSetLabelMapping("labels")
                .dataSetFeatureMaskMapping("loss_mask")
                .computeDataType(trainingConfig.getComputeDataType())
                .masterWeightDataType(DataType.FLOAT)
                .gradientAccumulationSteps(trainingConfig.getGradientAccumulationSteps());

        if (trainingConfig.getWeightDecay() > 0) {
            builder.weightDecay(trainingConfig.getWeightDecay(), true);
        }

        if (trainingConfig.getLossScaleConfig() != null) {
            builder.lossScaling(trainingConfig.getLossScaleConfig());
        } else if (trainingConfig.getComputeDataType() == DataType.FLOAT16) {
            builder.mixedPrecision();
        } else if (trainingConfig.getComputeDataType() == DataType.BFLOAT16) {
            builder.mixedPrecisionBfloat16();
        }

        return builder.build();
    }

    /**
     * Build a simple in-memory {@link MultiDataSetIterator} from a list of
     * {@link VlmTrainingExample} objects.
     *
     * <p>Layout per MultiDataSet (batch-of-1):
     * <ul>
     *   <li>features[0] — {@code image_features}: float, [1, numPatches, patchDim] or [1, C, H, W]</li>
     *   <li>features[1] — {@code input_ids}: long, [1, seqLen] (pseudo token IDs from char codes)</li>
     *   <li>labels[0]   — {@code labels}: long, [1, seqLen] (next-token prediction targets)</li>
     *   <li>featureMasks[0] — {@code loss_mask}: float, [1, seqLen] (1=train on response, 0=ignore prompt)</li>
     * </ul>
     *
     * @param examples VLM training examples.
     * @return In-memory MultiDataSetIterator.
     */
    private MultiDataSetIterator buildIteratorFromExamples(List<VlmTrainingExample> examples) {
        final int vocabSize = 50257; // GPT-2 vocab as heuristic for pseudo-tokenisation
        final int charsPerToken = 4;
        final int maxSeq = trainingConfig.getMaxSeqLength();

        List<MultiDataSet> datasets = new ArrayList<>(examples.size());

        for (VlmTrainingExample ex : examples) {
            // Image features: pass through as-is (add batch dim)
            INDArray imgRaw = ex.getImageFeatures();
            long[] imgShape = new long[imgRaw.rank() + 1];
            imgShape[0] = 1L;
            for (int d = 0; d < imgRaw.rank(); d++) {
                imgShape[d + 1] = imgRaw.size(d);
            }
            INDArray imageFeatures = imgRaw.reshape(imgShape);

            // Build combined text: prompt then response
            String promptText   = ex.hasSystemMessage()
                    ? ex.getSystemMessage() + "\n" + ex.getPrompt()
                    : ex.getPrompt();
            String responseText = ex.getResponse();
            String fullText     = promptText + responseText;

            int promptTokenLen   = (promptText.length() + charsPerToken - 1) / charsPerToken;
            int seqLen           = Math.min(maxSeq,
                    (fullText.length() + charsPerToken - 1) / charsPerToken);

            // Pseudo token IDs (char code mod vocab)
            long[] tokenIds = new long[seqLen];
            for (int t = 0; t < seqLen; t++) {
                int charIdx = Math.min(t * charsPerToken, fullText.length() - 1);
                tokenIds[t] = fullText.charAt(charIdx) % vocabSize;
            }

            // Next-token labels (shift right by 1)
            long[] labelIds = new long[seqLen];
            for (int t = 0; t < seqLen - 1; t++) {
                labelIds[t] = tokenIds[t + 1];
            }
            labelIds[seqLen - 1] = 0L;

            // Loss mask: 0 on prompt tokens, 1 on response tokens
            float[] lossMask = new float[seqLen];
            for (int t = 0; t < seqLen; t++) {
                lossMask[t] = (t >= promptTokenLen) ? 1.0f : 0.0f;
            }

            INDArray inputIds    = Nd4j.createFromArray(tokenIds).reshape(1, seqLen);
            INDArray labels      = Nd4j.createFromArray(labelIds).reshape(1, seqLen);
            INDArray lossMaskArr = Nd4j.createFromArray(lossMask).reshape(1, seqLen);

            datasets.add(new org.nd4j.linalg.dataset.MultiDataSet(
                    new INDArray[]{imageFeatures, inputIds},   // features
                    new INDArray[]{labels},                     // labels
                    new INDArray[]{lossMaskArr},                // feature masks
                    null                                        // label masks
            ));
        }

        return new SimpleListMultiDataSetIterator(datasets);
    }

    // -------------------------------------------------------------------------
    // Minimal in-memory MultiDataSetIterator implementation
    // -------------------------------------------------------------------------

    /**
     * A simple, non-parallel {@link MultiDataSetIterator} backed by a list.
     * Intended for small in-memory datasets and tests.
     */
    private static class SimpleListMultiDataSetIterator implements MultiDataSetIterator {

        private final List<MultiDataSet> data;
        private int cursor = 0;

        SimpleListMultiDataSetIterator(List<MultiDataSet> data) {
            this.data = data;
        }

        @Override
        public MultiDataSet next(int num) {
            return next();
        }

        @Override
        public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
            // No-op: pre-processing not needed for in-memory examples
        }

        @Override
        public MultiDataSetPreProcessor getPreProcessor() {
            return null;
        }

        @Override
        public boolean resetSupported() {
            return true;
        }

        @Override
        public boolean asyncSupported() {
            return false;
        }

        @Override
        public void reset() {
            cursor = 0;
        }

        @Override
        public boolean hasNext() {
            return cursor < data.size();
        }

        @Override
        public MultiDataSet next() {
            return data.get(cursor++);
        }
    }
}
