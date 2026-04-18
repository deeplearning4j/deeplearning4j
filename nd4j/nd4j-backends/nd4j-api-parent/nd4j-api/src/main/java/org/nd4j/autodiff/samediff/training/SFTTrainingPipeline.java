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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.config.SFTConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.dataset.curation.format.ConversationExtension;
import org.nd4j.linalg.dataset.curation.format.ConversationTurn;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.dataset.curation.format.InstructionDataFormatter;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.linalg.schedule.CosineWarmupSchedule;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Orchestrates Supervised Fine-Tuning (SFT) of language models in DL4J/ND4J.
 *
 * <p>SFT is the standard recipe for adapting a pre-trained language model to follow
 * instructions or behave according to a particular chat style.  The critical
 * difference from continued pretraining is <b>response-token masking</b>: loss is
 * computed only on assistant-generated tokens, never on system or user prompt tokens.
 *
 * <h2>Response masking pipeline</h2>
 * <ol>
 *   <li>{@link InstructionDataFormatter} formats a conversation and returns a
 *       {@link FormattedExample} with a character-level {@code lossMask} array
 *       ({@code 1} = train on this character, {@code 0} = ignore).</li>
 *   <li>{@link #convertCharMaskToTokenMask} maps the character-level mask to a
 *       token-level mask using the {@code charsPerToken} heuristic (default 4).
 *       A token is trainable if the majority of its characters are masked.</li>
 *   <li>The token-level mask is supplied as the {@code loss_mask} feature in the
 *       {@link MultiDataSet} fed to {@link SameDiff#fit}.</li>
 * </ol>
 *
 * <h2>PEFT integration</h2>
 * When {@link SFTConfig#getPeftConfig()} is non-null the pipeline wraps the base
 * {@link SameDiff} model with {@link PeftModel} before training, so only adapter
 * weights are updated.  After training the adapters can be merged with
 * {@link #mergeAndExport()}.
 *
 * <h2>Usage</h2>
 * <pre>
 * SFTConfig config = SFTConfig.builder()
 *     .chatTemplate(ChatTemplate.CHATML)
 *     .learningRate(2e-5)
 *     .numEpochs(3)
 *     .gradientAccumulationSteps(4)
 *     .peftConfig(LoraConfig.defaultTransformer())
 *     .build();
 *
 * SFTTrainingPipeline pipeline = new SFTTrainingPipeline(baseModel, config);
 *
 * // Option A – bring your own MultiDataSetIterator
 * pipeline.train(myDataIterator, totalSteps);
 *
 * // Option B – provide raw instruction/response pairs
 * pipeline.trainFromPairs(pairs);
 *
 * // Option C – pre-built conversation lists
 * pipeline.train(conversations);
 *
 * // Export merged model (if PEFT was used)
 * SameDiff merged = pipeline.mergeAndExport();
 * </pre>
 *
 * @author Adam Gibson
 * @see SFTConfig
 * @see InstructionDataFormatter
 * @see PeftModel
 */
@Slf4j
public class SFTTrainingPipeline {

    /** Characters-per-token heuristic used when converting char masks to token masks. */
    private static final int CHARS_PER_TOKEN = 4;

    /** The SFT configuration (immutable after construction). */
    @Getter
    private final SFTConfig config;

    /**
     * The model that will actually be trained.  When PEFT is configured this is
     * the underlying {@link PeftModel#getModel()}, i.e. the modified SameDiff graph.
     * When PEFT is absent this equals the base model passed to the constructor.
     */
    @Getter
    private final SameDiff model;

    /**
     * PEFT wrapper.  Non-null only when {@link SFTConfig#getPeftConfig()} is set.
     */
    @Getter
    private final PeftModel peftModel;

    /** Formatter used to convert conversations into text + char-level masks. */
    private final InstructionDataFormatter formatter;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Create a new SFT pipeline.
     *
     * @param baseModel The pre-trained SameDiff model to fine-tune.
     * @param config    SFT configuration (validated inside this constructor).
     */
    public SFTTrainingPipeline(SameDiff baseModel, SFTConfig config) {
        config.validate();
        this.config = config;

        // Build the formatter using the chat template from config
        this.formatter = InstructionDataFormatter.builder()
                .template(config.getChatTemplate())
                .build();

        // Optionally wrap with PEFT
        if (config.getPeftConfig() != null) {
            this.peftModel = PeftModel.fromPretrained(baseModel, config.getPeftConfig());
            this.model = this.peftModel.getModel();
            log.info("SFT pipeline: {} applied. Trainable params: {}/{} ({} %)",
                    config.getPeftConfig().getPeftType(),
                    peftModel.getTrainableParameterCount(),
                    peftModel.getTotalParameterCount(),
                    String.format("%.2f", peftModel.getTrainablePercentage()));
        } else {
            this.peftModel = null;
            this.model = baseModel;
            log.info("SFT pipeline: full fine-tune (no PEFT)");
        }
    }

    // -------------------------------------------------------------------------
    // Training – public API
    // -------------------------------------------------------------------------

    /**
     * Train using an externally-constructed {@link MultiDataSetIterator}.
     *
     * <p>Each {@link MultiDataSet} should contain at minimum:
     * <ul>
     *   <li>feature[0] — {@code input_ids} (long, [batch, seqLen])</li>
     *   <li>label[0]   — {@code labels} (long, [batch, seqLen])</li>
     *   <li>featureMask[0] — {@code loss_mask} (float, [batch, seqLen], 1=train 0=ignore)</li>
     * </ul>
     *
     * @param dataIterator Data iterator providing batches.
     * @param totalSteps   Total number of optimiser update steps (used to build the LR schedule).
     */
    public void train(MultiDataSetIterator dataIterator, int totalSteps) {
        TrainingConfig trainingConfig = buildTrainingConfig(totalSteps);
        model.setTrainingConfig(trainingConfig);

        log.info("Starting SFT: lr={}, warmupRatio={}, epochs={}, gradAccum={}, dtype={}",
                config.getLearningRate(), config.getWarmupRatio(),
                config.getNumEpochs(), config.getGradientAccumulationSteps(),
                config.getComputeDataType());

        for (int epoch = 0; epoch < config.getNumEpochs(); epoch++) {
            log.info("SFT epoch {}/{}", epoch + 1, config.getNumEpochs());
            model.fit(dataIterator, 1);
            dataIterator.reset();
        }

        log.info("SFT training complete");
    }

    /**
     * Convenience method: format a list of already-built conversations and train.
     *
     * <p>Each element of {@code conversations} is a multi-turn dialogue represented
     * as an ordered list of {@link ConversationTurn} objects.  The pipeline formats
     * them, converts char-level masks to token-level masks, assembles a single-pass
     * in-memory iterator, and calls {@link #train(MultiDataSetIterator, int)}.
     *
     * @param conversations List of multi-turn conversations to train on.
     */
    public void train(List<List<ConversationTurn>> conversations) {
        List<FormattedExample> examples = formatConversations(conversations);
        MultiDataSetIterator iterator = buildIteratorFromExamples(examples);
        int totalSteps = estimateTotalSteps(examples.size());
        train(iterator, totalSteps);
    }

    /**
     * Convenience method: take a list of raw (instruction, response) string pairs,
     * optionally synthesise multi-turn conversations via {@link ConversationExtension},
     * and train.
     *
     * @param pairs List of {@link Map.Entry} where key=instruction and value=response.
     */
    public void trainFromPairs(List<Map.Entry<String, String>> pairs) {
        // Convert to ConversationTurn pairs, optionally adding system message
        List<ConversationTurn> turns = buildTurnsFromPairs(pairs);

        // Wrap each pair into a single-turn conversation list without extension
        List<List<ConversationTurn>> conversations = new ArrayList<>();
        for (int i = 0; i + 1 < turns.size(); i += 2) {
            List<ConversationTurn> conv = new ArrayList<>();
            if (config.getSystemMessage() != null) {
                conv.add(ConversationTurn.system(config.getSystemMessage()));
            }
            conv.add(turns.get(i));
            conv.add(turns.get(i + 1));
            conversations.add(conv);
        }

        train(conversations);
    }

    // -------------------------------------------------------------------------
    // Export
    // -------------------------------------------------------------------------

    /**
     * Merge PEFT adapter weights back into the base model and return a standalone
     * {@link SameDiff} model ready for inference without adapter overhead.
     *
     * @return Merged model, or the base model if PEFT was not configured.
     * @throws UnsupportedOperationException if the active PEFT type does not support merging.
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
    // TrainingConfig construction
    // -------------------------------------------------------------------------

    /**
     * Build a {@link TrainingConfig} from this pipeline's {@link SFTConfig}.
     *
     * @param totalSteps Total optimiser update steps (for LR schedule).
     * @return A fully configured {@link TrainingConfig}.
     */
    public TrainingConfig buildTrainingConfig(int totalSteps) {
        // Learning-rate schedule
        ISchedule lrSchedule = config.getLrSchedule();
        if (lrSchedule == null) {
            lrSchedule = CosineWarmupSchedule.fromRatio(
                    config.getLearningRate(),
                    config.getMinLearningRate(),
                    config.getWarmupRatio(),
                    totalSteps
            );
        }

        // AdamW optimizer
        Adam adam = Adam.builder()
                .learningRateSchedule(lrSchedule)
                .beta1(0.9)
                .beta2(0.999)
                .epsilon(1e-8)
                .build();

        TrainingConfig.Builder builder = TrainingConfig.builder()
                .updater(adam)
                // input_ids fed as first feature; loss_mask as first feature mask
                .dataSetFeatureMapping("input_ids")
                .dataSetLabelMapping("labels")
                .dataSetFeatureMaskMapping("loss_mask")
                .computeDataType(config.getComputeDataType())
                .masterWeightDataType(DataType.FLOAT)
                .gradientAccumulationSteps(config.getGradientAccumulationSteps());

        if (config.getWeightDecay() > 0) {
            builder.weightDecay(config.getWeightDecay(), true);
        }

        if (config.getLossScaleConfig() != null) {
            builder.lossScaling(config.getLossScaleConfig());
        } else if (config.getComputeDataType() == DataType.FLOAT16) {
            // FP16 without explicit loss scale config: enable dynamic scaling by default
            builder.mixedPrecision();
        } else if (config.getComputeDataType() == DataType.BFLOAT16) {
            builder.mixedPrecisionBfloat16();
        }

        return builder.build();
    }

    // -------------------------------------------------------------------------
    // Response masking
    // -------------------------------------------------------------------------

    /**
     * Convert a character-level loss mask from {@link FormattedExample#getLossMask()}
     * into a token-level loss mask array of length {@code seqLen}.
     *
     * <p>The heuristic: a token at position {@code t} spans characters
     * {@code [t*charsPerToken, (t+1)*charsPerToken)}.  The token is trainable
     * ({@code 1}) if the majority of those characters are marked {@code 1} in the
     * char mask.  This is a best-effort approximation; production systems should use
     * an actual tokenizer offset mapping instead.
     *
     * @param charMask     Character-level loss mask from {@link FormattedExample}.
     * @param seqLen       Number of tokens in the tokenised sequence.
     * @param charsPerToken Estimated characters per token (default 4).
     * @return Token-level mask of length {@code seqLen} (values: 0 or 1).
     */
    public static int[] convertCharMaskToTokenMask(int[] charMask, int seqLen, int charsPerToken) {
        int[] tokenMask = new int[seqLen];
        for (int t = 0; t < seqLen; t++) {
            int charStart = t * charsPerToken;
            int charEnd = Math.min(charStart + charsPerToken, charMask.length);
            if (charStart >= charMask.length) {
                // Beyond end of formatted text — treat as non-trainable padding
                tokenMask[t] = 0;
                continue;
            }
            int ones = 0;
            int total = charEnd - charStart;
            for (int c = charStart; c < charEnd; c++) {
                ones += charMask[c];
            }
            // Majority vote: train if > half the chars are masked
            tokenMask[t] = (ones * 2 > total) ? 1 : 0;
        }
        return tokenMask;
    }

    /**
     * Overload using the default {@link #CHARS_PER_TOKEN} constant.
     */
    public static int[] convertCharMaskToTokenMask(int[] charMask, int seqLen) {
        return convertCharMaskToTokenMask(charMask, seqLen, CHARS_PER_TOKEN);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Format a list of conversations into {@link FormattedExample} objects.
     */
    private List<FormattedExample> formatConversations(List<List<ConversationTurn>> conversations) {
        List<FormattedExample> examples = new ArrayList<>(conversations.size());
        for (List<ConversationTurn> conv : conversations) {
            examples.add(formatter.format(conv));
        }
        return examples;
    }

    /**
     * Convert (instruction, response) map entries into a flat list of
     * {@link ConversationTurn} pairs, prepending system message if configured.
     */
    private List<ConversationTurn> buildTurnsFromPairs(List<Map.Entry<String, String>> pairs) {
        List<ConversationTurn> turns = new ArrayList<>(pairs.size() * 2);
        for (Map.Entry<String, String> entry : pairs) {
            turns.add(ConversationTurn.user(entry.getKey()));
            turns.add(ConversationTurn.assistant(entry.getValue()));
        }
        return turns;
    }

    /**
     * Build a simple in-memory {@link MultiDataSetIterator} from a list of formatted
     * examples.  Each example becomes one batch-of-1 {@link MultiDataSet}.
     *
     * <p>The layout per dataset:
     * <ul>
     *   <li>features[0] — {@code input_ids}: long tokens, shape [1, seqLen]</li>
     *   <li>labels[0]   — {@code labels}: long token labels (same as input_ids shifted by 1), shape [1, seqLen]</li>
     *   <li>featureMasks[0] — {@code loss_mask}: float, shape [1, seqLen]</li>
     * </ul>
     *
     * <p>In a real production pipeline the token IDs would come from a proper
     * tokenizer.  Here we synthesise pseudo-token IDs from character code points
     * modulo a vocabulary size of 50257 (GPT-2 vocab) for demonstration/testing
     * purposes.  The important invariant — the loss mask — is derived correctly from
     * the {@link FormattedExample#getLossMask()} char-level mask.
     */
    private MultiDataSetIterator buildIteratorFromExamples(List<FormattedExample> examples) {
        List<MultiDataSet> datasets = new ArrayList<>(examples.size());
        final int vocabSize = 50257;

        for (FormattedExample ex : examples) {
            String text = ex.getText();
            int[] charMask = ex.getLossMask();

            // Truncate / pad to maxSeqLength using char heuristic for token count
            int seqLen = Math.min(config.getMaxSeqLength(),
                    (text.length() + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN);

            // Build pseudo token IDs from char code points
            long[] tokenIds = new long[seqLen];
            for (int t = 0; t < seqLen; t++) {
                int charIdx = Math.min(t * CHARS_PER_TOKEN, text.length() - 1);
                tokenIds[t] = text.charAt(charIdx) % vocabSize;
            }

            // Shift right by one for causal LM labels (next-token prediction)
            long[] labelIds = new long[seqLen];
            for (int t = 0; t < seqLen - 1; t++) {
                labelIds[t] = tokenIds[t + 1];
            }
            labelIds[seqLen - 1] = 0L; // padding label at last position

            // Convert char mask to token mask
            int[] tokenMaskInt = convertCharMaskToTokenMask(charMask, seqLen);
            float[] tokenMaskFloat = new float[seqLen];
            for (int t = 0; t < seqLen; t++) {
                tokenMaskFloat[t] = tokenMaskInt[t];
            }

            INDArray inputIds = Nd4j.createFromArray(tokenIds).reshape(1, seqLen);
            INDArray labels   = Nd4j.createFromArray(labelIds).reshape(1, seqLen);
            INDArray lossMask = Nd4j.createFromArray(tokenMaskFloat).reshape(1, seqLen);

            datasets.add(new org.nd4j.linalg.dataset.MultiDataSet(
                    new INDArray[]{inputIds},
                    new INDArray[]{labels},
                    new INDArray[]{lossMask},
                    null));
        }

        return new SimpleListMultiDataSetIterator(datasets);
    }

    /**
     * Estimate total optimiser update steps from the number of examples.
     * Accounts for gradient accumulation and the number of epochs.
     */
    private int estimateTotalSteps(int numExamples) {
        int stepsPerEpoch = Math.max(1, numExamples / config.getGradientAccumulationSteps());
        return stepsPerEpoch * config.getNumEpochs();
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
        public void setPreProcessor(
                org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor preProcessor) {
            // No-op: pre-processing not needed for in-memory examples
        }

        @Override
        public org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor getPreProcessor() {
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
