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

package org.eclipse.deeplearning4j.audio.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.TtsFineTuneConfig;
import org.nd4j.autodiff.samediff.config.TtsTrainingConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.SimpleListMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.curation.audio.AudioDataProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Orchestrates TTS (Text-to-Speech) fine-tuning with SameDiff.
 *
 * <p>The pipeline:
 * <ol>
 *   <li>Optionally freezes the text/acoustic encoder based on
 *       {@link TtsFineTuneConfig#isFreeze TextEncoder()}.</li>
 *   <li>Optionally wraps the model with {@link PeftModel} when
 *       {@link TtsTrainingConfig#getPeftConfig()} is non-null.</li>
 *   <li>Builds an AdamW optimizer with a cosine warmup LR schedule.</li>
 *   <li>Runs the training loop via {@link SameDiff#fit}.</li>
 * </ol>
 *
 * <p>Two training entry points are provided:
 * <ul>
 *   <li>{@link #train(MultiDataSetIterator, int)} — for externally-constructed iterators.</li>
 *   <li>{@link #train(List)} — convenience method that processes raw
 *       {@link TtsTrainingExample} objects (waveform + text) into mel spectrograms
 *       and builds an in-memory iterator.</li>
 * </ul>
 *
 * <p>Each {@link MultiDataSet} fed to the model is expected to contain:
 * <ul>
 *   <li>{@code features[0]} — mel spectrogram {@code [batch, melBins, frames]} as FLOAT</li>
 *   <li>{@code labels[0]}   — target mel spectrogram for reconstruction loss
 *       {@code [batch, melBins, frames]} as FLOAT</li>
 * </ul>
 *
 * <p>Example:
 * <pre>
 * SameDiff ttsModel = SameDiff.load(new File("tts.sd"), true);
 *
 * TtsFineTuneConfig ttsConfig = TtsFineTuneConfig.voiceCloning();
 * TtsTrainingConfig trainConfig = TtsTrainingConfig.builder()
 *     .learningRate(1e-4)
 *     .numEpochs(10)
 *     .build();
 *
 * TtsTrainingPipeline pipeline = TtsTrainingPipeline.create(ttsModel, ttsConfig, trainConfig);
 *
 * // Train from raw examples
 * pipeline.train(examples);
 *
 * // Export merged model (if LoRA was used)
 * SameDiff merged = pipeline.mergeAndExport();
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TtsFineTuneConfig
 * @see TtsTrainingConfig
 * @see TtsTrainingExample
 */
@Slf4j
public class TtsTrainingPipeline {

    /** The TTS audio feature configuration (mel spectrogram parameters, etc.). */
    @Getter
    private final TtsFineTuneConfig ttsConfig;

    /** The training loop configuration (LR, epochs, PEFT, etc.). */
    @Getter
    private final TtsTrainingConfig trainingConfig;

    /**
     * The model that will actually be trained.  When PEFT is configured, this is
     * the underlying {@link PeftModel#getModel()}; otherwise it is the base model.
     */
    @Getter
    private final SameDiff model;

    /**
     * PEFT wrapper.  Non-null only when {@link TtsTrainingConfig#getPeftConfig()} is set.
     */
    @Getter
    private final PeftModel peftModel;

    /** Audio preprocessing pipeline driven by {@link #ttsConfig}. */
    private final AudioDataProcessor audioProcessor;

    /**
     * Private constructor — use {@link #create(SameDiff, TtsFineTuneConfig, TtsTrainingConfig)}.
     */
    private TtsTrainingPipeline(SameDiff baseModel,
                                 TtsFineTuneConfig ttsConfig,
                                 TtsTrainingConfig trainingConfig) {
        this.ttsConfig = ttsConfig;
        this.trainingConfig = trainingConfig;
        this.audioProcessor = new AudioDataProcessor(ttsConfig);

        // Optionally freeze encoder variables
        if (ttsConfig.isFreezeTextEncoder()) {
            freezeEncoder(baseModel);
        }

        // Optionally wrap with PEFT
        if (trainingConfig.getPeftConfig() != null) {
            this.peftModel = PeftModel.fromPretrained(baseModel, trainingConfig.getPeftConfig());
            this.model = this.peftModel.getModel();
            log.info("TTS pipeline: {} applied. Trainable params: {}/{} ({} %)",
                    trainingConfig.getPeftConfig().getPeftType(),
                    peftModel.getTrainableParameterCount(),
                    peftModel.getTotalParameterCount(),
                    String.format("%.2f", peftModel.getTrainablePercentage()));
        } else {
            this.peftModel = null;
            this.model = baseModel;
            log.info("TTS pipeline: full fine-tune (no PEFT), freezeEncoder={}",
                    ttsConfig.isFreezeTextEncoder());
        }
    }

    /**
     * Create a new TTS training pipeline.
     *
     * @param model          Base TTS SameDiff model.
     * @param ttsConfig      Audio feature configuration (mel spectrogram parameters).
     * @param trainingConfig Training loop configuration (LR, epochs, PEFT, etc.).
     * @return A new pipeline ready for training.
     */
    public static TtsTrainingPipeline create(SameDiff model,
                                              TtsFineTuneConfig ttsConfig,
                                              TtsTrainingConfig trainingConfig) {
        Preconditions.checkNotNull(model, "model must not be null");
        Preconditions.checkNotNull(ttsConfig, "ttsConfig must not be null");
        Preconditions.checkNotNull(trainingConfig, "trainingConfig must not be null");
        ttsConfig.validate();

        // Auto-detect target modules for LoRA when none are specified.
        // Collect all 2-D VARIABLE-type variable names from the model and use them
        // so callers don't need to know the variable naming scheme in advance.
        if (trainingConfig.getPeftConfig() instanceof LoraConfig) {
            LoraConfig loraConfig = (LoraConfig) trainingConfig.getPeftConfig();
            if (loraConfig.getTargetModules() == null || loraConfig.getTargetModules().isEmpty()) {
                List<String> autoTargets = new ArrayList<>();
                for (SDVariable var : model.variables()) {
                    if (var.getVariableType() == VariableType.VARIABLE) {
                        long[] shape = var.getShape();
                        if (shape != null && shape.length == 2) {
                            autoTargets.add(var.name());
                        }
                    }
                }
                Preconditions.checkState(!autoTargets.isEmpty(),
                        "Target modules must contain at least one valid module. " +
                        "No 2-D weight variables found in the model to apply LoRA to.");
                log.info("LoRA: auto-detected target modules from model variables: {}", autoTargets);
                loraConfig.setTargetModules(autoTargets);
            }
        }

        trainingConfig.validate();
        return new TtsTrainingPipeline(model, ttsConfig, trainingConfig);
    }

    // -------------------------------------------------------------------------
    // Training — public API
    // -------------------------------------------------------------------------

    /**
     * Train using an externally-constructed {@link MultiDataSetIterator}.
     *
     * <p>Each {@link MultiDataSet} must contain at minimum:
     * <ul>
     *   <li>{@code features[0]} — mel spectrogram {@code [batch, melBins, frames]} as FLOAT</li>
     *   <li>{@code labels[0]}   — target mel spectrogram {@code [batch, melBins, frames]} as FLOAT</li>
     * </ul>
     *
     * @param data       Iterator providing batches of mel spectrogram data.
     * @param totalSteps Total number of optimizer update steps (used for LR schedule).
     */
    public void train(MultiDataSetIterator data, int totalSteps) {
        TrainingConfig tc = buildTrainingConfig(totalSteps);
        model.setTrainingConfig(tc);

        log.info("TTS fine-tuning start: lr={}, minLr={}, warmupSteps={}, epochs={}, " +
                        "batchSize={}, gradAccum={}, dtype={}",
                trainingConfig.getLearningRate(),
                trainingConfig.getMinLearningRate(),
                trainingConfig.getWarmupSteps(),
                trainingConfig.getNumEpochs(),
                trainingConfig.getBatchSize(),
                trainingConfig.getGradientAccumulationSteps(),
                trainingConfig.getComputeDataType());

        for (int epoch = 0; epoch < trainingConfig.getNumEpochs(); epoch++) {
            log.info("TTS epoch {}/{}", epoch + 1, trainingConfig.getNumEpochs());
            model.fit(data, 1);
            data.reset();
        }

        log.info("TTS fine-tuning complete");
    }

    /**
     * Convenience method: process raw {@link TtsTrainingExample} objects and train.
     *
     * <p>This method:
     * <ol>
     *   <li>Truncates each waveform to {@link TtsTrainingConfig#getMaxAudioLengthSec()}.</li>
     *   <li>Runs {@link AudioDataProcessor#process(INDArray)} to produce mel spectrograms.</li>
     *   <li>Batches spectrograms into {@link MultiDataSet} objects grouped by
     *       {@link TtsTrainingConfig#getBatchSize()}.</li>
     *   <li>Calls {@link #train(MultiDataSetIterator, int)} with the resulting iterator.</li>
     * </ol>
     *
     * @param examples List of training examples (waveform + text pairs).
     */
    public void train(List<TtsTrainingExample> examples) {
        Preconditions.checkState(!examples.isEmpty(), "examples list must not be empty");

        log.info("Processing {} TTS training examples into mel spectrograms...", examples.size());
        List<MultiDataSet> datasets = buildDatasetsFromExamples(examples);

        int totalSteps = estimateTotalSteps(examples.size());
        MultiDataSetIterator iterator = new SimpleListMultiDataSetIterator(datasets);
        train(iterator, totalSteps);
    }

    // -------------------------------------------------------------------------
    // Export
    // -------------------------------------------------------------------------

    /**
     * Merge PEFT adapter weights back into the base model and return a standalone
     * SameDiff model suitable for inference.
     *
     * @return The merged model, or the base model unchanged if PEFT was not configured.
     */
    public SameDiff mergeAndExport() {
        if (peftModel == null) {
            log.info("mergeAndExport: no PEFT configured, returning model as-is");
            return model;
        }
        SameDiff merged = peftModel.mergeAndUnload();
        log.info("mergeAndExport: LoRA layers merged into base model");
        return merged;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Build a {@link TrainingConfig} from this pipeline's {@link TtsTrainingConfig}.
     *
     * @param totalSteps Total optimizer update steps (for LR schedule computation).
     * @return A fully configured TrainingConfig.
     */
    private TrainingConfig buildTrainingConfig(int totalSteps) {
        ISchedule lrSchedule = new CosineWarmupSchedule(
                trainingConfig.getLearningRate(),
                trainingConfig.getMinLearningRate(),
                trainingConfig.getWarmupSteps(),
                totalSteps);

        Adam adam = Adam.builder()
                .learningRateSchedule(lrSchedule)
                .beta1(0.9)
                .beta2(0.999)
                .epsilon(1e-8)
                .build();

        TrainingConfig.Builder builder = TrainingConfig.builder()
                .updater(adam)
                .dataSetFeatureMapping("mel_spectrogram")
                .dataSetLabelMapping("target_mel")
                .computeDataType(trainingConfig.getComputeDataType())
                .masterWeightDataType(DataType.FLOAT)
                .gradientAccumulationSteps(trainingConfig.getGradientAccumulationSteps());

        if (trainingConfig.getWeightDecay() > 0) {
            builder.weightDecay(trainingConfig.getWeightDecay(), true);
        }

        if (trainingConfig.getComputeDataType() == DataType.BFLOAT16) {
            builder.mixedPrecisionBfloat16();
        } else if (trainingConfig.getComputeDataType() == DataType.FLOAT16) {
            builder.mixedPrecision();
        }

        return builder.build();
    }

    /**
     * Freeze variables whose names suggest they belong to the text encoder.
     *
     * <p>Variables matching patterns like "encoder*", "text_encoder*", or
     * "embed_tokens*" are marked non-trainable. This heuristic mirrors the
     * convention used in most publicly available TTS model exports.
     */
    private void freezeEncoder(SameDiff sd) {
        List<String> toFreeze = new ArrayList<>();
        for (String varName : sd.variableNames()) {
            if (isEncoderVariable(varName)) {
                toFreeze.add(varName);
            }
        }
        if (!toFreeze.isEmpty()) {
            sd.freezeVariables(toFreeze);
        }
        log.info("freezeEncoder: froze {} encoder variables", toFreeze.size());
    }

    /**
     * Return true if the variable name matches common TTS encoder naming patterns.
     */
    private boolean isEncoderVariable(String name) {
        String lower = name.toLowerCase();
        return lower.startsWith("encoder")
                || lower.startsWith("text_encoder")
                || lower.startsWith("embed_tokens")
                || lower.contains("/encoder/")
                || lower.contains(".encoder.");
    }

    /**
     * Process a list of {@link TtsTrainingExample} objects into batched
     * {@link MultiDataSet} objects ready for {@link SameDiff#fit}.
     */
    private List<MultiDataSet> buildDatasetsFromExamples(List<TtsTrainingExample> examples) {
        List<MultiDataSet> result = new ArrayList<>();
        int batchSize = trainingConfig.getBatchSize();
        int maxSamples = (int) (trainingConfig.getMaxAudioLengthSec() * ttsConfig.getSampleRate());

        List<INDArray> batchMels = new ArrayList<>(batchSize);

        for (TtsTrainingExample ex : examples) {
            INDArray waveform = ex.getAudioWaveform();

            // Truncate to max audio length
            waveform = truncateWaveform(waveform, maxSamples);

            // Compute mel spectrogram: [batch, melBins, frames] or [melBins, frames]
            INDArray mel = audioProcessor.process(waveform);

            // Ensure 3-D: [1, melBins, frames]
            if (mel.rank() == 2) {
                mel = mel.reshape(1, mel.size(0), mel.size(1));
            }

            batchMels.add(mel);

            if (batchMels.size() == batchSize) {
                result.add(buildBatch(batchMels));
                batchMels.clear();
            }
        }

        // Remainder batch
        if (!batchMels.isEmpty()) {
            result.add(buildBatch(batchMels));
        }

        log.info("Built {} batches from {} examples (batchSize={})",
                result.size(), examples.size(), batchSize);
        return result;
    }

    /**
     * Assemble a list of per-example mel spectrograms into a single padded
     * {@link MultiDataSet}.  Spectrograms shorter than the longest one in the batch
     * are zero-padded along the frames dimension.
     */
    private MultiDataSet buildBatch(List<INDArray> mels) {
        // Find max frames in this batch
        long maxFrames = 0;
        for (INDArray m : mels) {
            maxFrames = Math.max(maxFrames, m.size(2));
        }
        long melBins = mels.get(0).size(1);
        int b = mels.size();

        INDArray batch = Nd4j.zeros(DataType.FLOAT, b, melBins, maxFrames);
        for (int i = 0; i < b; i++) {
            INDArray m = mels.get(i);
            batch.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(i),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, m.size(2))
            ).assign(m.reshape(melBins, m.size(2)));
        }

        // For reconstruction loss the label equals the input (autoencoder objective)
        INDArray target = batch.dup();

        return new MultiDataSet(
                new INDArray[]{batch},
                new INDArray[]{target});
    }

    /**
     * Truncate a waveform to at most {@code maxSamples} samples.
     */
    private INDArray truncateWaveform(INDArray waveform, int maxSamples) {
        if (waveform.rank() == 1) {
            if (waveform.size(0) > maxSamples) {
                return waveform.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, maxSamples));
            }
        } else {
            if (waveform.size(1) > maxSamples) {
                return waveform.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, maxSamples));
            }
        }
        return waveform;
    }

    /**
     * Estimate total optimizer update steps from example count, batch size,
     * gradient accumulation, and epoch count.
     */
    private int estimateTotalSteps(int numExamples) {
        int batchesPerEpoch = Math.max(1,
                (numExamples + trainingConfig.getBatchSize() - 1) / trainingConfig.getBatchSize());
        int stepsPerEpoch = Math.max(1,
                batchesPerEpoch / trainingConfig.getGradientAccumulationSteps());
        return stepsPerEpoch * trainingConfig.getNumEpochs();
    }

}
