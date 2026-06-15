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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.tts;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.audio.training.SpeakerEncoder;
import org.eclipse.deeplearning4j.audio.training.TtsTrainingExample;
import org.eclipse.deeplearning4j.audio.training.TtsTrainingPipeline;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.autodiff.samediff.config.TaskType;
import org.nd4j.autodiff.samediff.config.TtsFineTuneConfig;
import org.nd4j.autodiff.samediff.config.TtsTrainingConfig;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.curation.audio.AudioDataProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the TTS training pipeline: {@link TtsTrainingPipeline},
 * {@link TtsTrainingConfig}, {@link TtsTrainingExample}, and {@link SpeakerEncoder}.
 *
 * <p>All tests use small synthetic models and arrays — no real TTS model weights
 * or audio files are required.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
@DisplayName("TTS Training Pipeline Tests")
public class TestTtsTrainingPipeline extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Build a tiny encoder-decoder SameDiff model for TTS tests.
     *
     * <p>The encoder variables are prefixed with "encoder." to match the freeze heuristic.
     * The decoder accepts a mel spectrogram and produces a reconstruction target.
     */
    private static SameDiff buildTinyTtsModel(int melBins, int frames) {
        SameDiff sd = SameDiff.create();

        // Encoder weights (should be frozen in voiceCloning mode)
        SDVariable encoderW = sd.var("encoder.weight",
                Nd4j.randn(DataType.FLOAT, melBins, melBins).muli(0.02));

        // Decoder weights (trainable)
        SDVariable decoderW = sd.var("decoder.weight",
                Nd4j.randn(DataType.FLOAT, melBins, melBins).muli(0.02));

        // Input placeholder: mel spectrogram [1, melBins, frames]
        SDVariable melInput = sd.placeHolder("mel_spectrogram", DataType.FLOAT, 1, melBins, frames);
        SDVariable target   = sd.placeHolder("target_mel",      DataType.FLOAT, 1, melBins, frames);

        // Trivial forward: reshape to [melBins, frames], linear, reshape back
        SDVariable flat     = melInput.reshape(melBins, frames);
        SDVariable encOut   = sd.mmul(encoderW, flat);
        SDVariable decOut   = sd.mmul(decoderW, encOut);
        SDVariable recon    = decOut.reshape(1, melBins, frames);

        // MSE reconstruction loss
        SDVariable diff = recon.sub(target);
        sd.loss.meanSquaredError("loss", target, recon, null);

        return sd;
    }

    /**
     * Build a tiny SameDiff speaker encoder that maps [1, samples] → [1, embDim].
     */
    private static SameDiff buildTinySpeakerEncoderModel(int samples, int embDim) {
        SameDiff sd = SameDiff.create();
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, samples, embDim).muli(0.02));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, samples);
        SDVariable out = sd.mmul(input, w);
        sd.setOutputs(out.name());
        return sd;
    }

    /**
     * Produce a list of synthetic TtsTrainingExamples with random audio data.
     */
    private static List<TtsTrainingExample> syntheticExamples(int n, int samples) {
        List<TtsTrainingExample> examples = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            examples.add(TtsTrainingExample.builder()
                    .audioWaveform(Nd4j.randn(DataType.FLOAT, 1, samples))
                    .text("Sample text number " + i)
                    .sampleRate(22050)
                    .build());
        }
        return examples;
    }

    // =========================================================================
    // 1. TtsTrainingConfig defaults
    // =========================================================================

    @Test
    @DisplayName("testTtsTrainingConfigDefaults")
    public void testTtsTrainingConfigDefaults() {
        TtsTrainingConfig config = TtsTrainingConfig.builder().build();

        assertEquals(1e-4, config.getLearningRate(), 1e-10);
        assertEquals(1e-6, config.getMinLearningRate(), 1e-12);
        assertEquals(500, config.getWarmupSteps());
        assertEquals(10, config.getNumEpochs());
        assertEquals(8, config.getBatchSize());
        assertEquals(30.0, config.getMaxAudioLengthSec(), 1e-9);
        assertEquals(2, config.getGradientAccumulationSteps());
        assertEquals(DataType.BFLOAT16, config.getComputeDataType());
        assertEquals(0.01, config.getWeightDecay(), 1e-10);
        assertNull(config.getPeftConfig());
    }

    // =========================================================================
    // 2. TtsTrainingConfig validation
    // =========================================================================

    @Test
    @DisplayName("testTtsTrainingConfigValidation")
    public void testTtsTrainingConfigValidation() {
        // Valid config should not throw
        TtsTrainingConfig valid = TtsTrainingConfig.builder().build();
        assertDoesNotThrow(valid::validate);

        // Negative learning rate
        TtsTrainingConfig badLr = TtsTrainingConfig.builder().learningRate(-1e-4).build();
        assertThrows(Exception.class, badLr::validate, "Negative learningRate should throw");

        // minLr > lr
        TtsTrainingConfig badMinLr = TtsTrainingConfig.builder()
                .learningRate(1e-5)
                .minLearningRate(1e-4)
                .build();
        assertThrows(Exception.class, badMinLr::validate, "minLr > lr should throw");

        // Zero epochs
        TtsTrainingConfig badEpochs = TtsTrainingConfig.builder().numEpochs(0).build();
        assertThrows(Exception.class, badEpochs::validate, "numEpochs=0 should throw");

        // Zero batch size
        TtsTrainingConfig badBatch = TtsTrainingConfig.builder().batchSize(0).build();
        assertThrows(Exception.class, badBatch::validate, "batchSize=0 should throw");
    }

    // =========================================================================
    // 3. TtsTrainingExample creation
    // =========================================================================

    @Test
    @DisplayName("testTtsTrainingExampleCreation")
    public void testTtsTrainingExampleCreation() {
        INDArray waveform = Nd4j.randn(DataType.FLOAT, 1, 22050);
        TtsTrainingExample ex = TtsTrainingExample.builder()
                .audioWaveform(waveform)
                .text("Hello world")
                .sampleRate(22050)
                .build();

        assertNotNull(ex.getAudioWaveform());
        assertEquals("Hello world", ex.getText());
        assertEquals(22050, ex.getSampleRate());
        assertNull(ex.getSpeakerEmbedding(), "no speaker embedding by default");
    }

    @Test
    @DisplayName("testTtsTrainingExampleWithSpeakerEmbedding")
    public void testTtsTrainingExampleWithSpeakerEmbedding() {
        INDArray waveform = Nd4j.randn(DataType.FLOAT, 1, 16000);
        INDArray spkEmb = Nd4j.randn(DataType.FLOAT, 256);

        TtsTrainingExample ex = TtsTrainingExample.builder()
                .audioWaveform(waveform)
                .text("Testing speaker embedding")
                .speakerEmbedding(spkEmb)
                .sampleRate(16000)
                .build();

        assertNotNull(ex.getSpeakerEmbedding());
        assertArrayEquals(new long[]{256}, ex.getSpeakerEmbedding().shape());
    }

    @Test
    @DisplayName("testTtsTrainingExampleDefaultSampleRate")
    public void testTtsTrainingExampleDefaultSampleRate() {
        TtsTrainingExample ex = TtsTrainingExample.builder()
                .audioWaveform(Nd4j.ones(DataType.FLOAT, 1, 100))
                .text("default sample rate")
                .build();
        assertEquals(22050, ex.getSampleRate(), "default sample rate should be 22050");
    }

    // =========================================================================
    // 4. SpeakerEncoder creation
    // =========================================================================

    @Test
    @DisplayName("testSpeakerEncoderCreation")
    public void testSpeakerEncoderCreation() {
        SameDiff encoderModel = buildTinySpeakerEncoderModel(16000, 256);
        SpeakerEncoder encoder = SpeakerEncoder.create(encoderModel, 256);

        assertNotNull(encoder);
        assertEquals(256, encoder.getEmbeddingDim());
        assertSame(encoderModel, encoder.getModel());
    }

    @Test
    @DisplayName("testSpeakerEncoderCreationValidation")
    public void testSpeakerEncoderCreationValidation() {
        assertThrows(Exception.class,
                () -> SpeakerEncoder.create(null, 256),
                "null model should throw");
        SameDiff model = buildTinySpeakerEncoderModel(100, 64);
        assertThrows(Exception.class,
                () -> SpeakerEncoder.create(model, 0),
                "zero embeddingDim should throw");
    }

    // =========================================================================
    // 5. SpeakerEncoder encode — correct output shape
    // =========================================================================

    @Test
    @DisplayName("testSpeakerEncoderEncode")
    public void testSpeakerEncoderEncode() {
        int samples = 100;
        int embDim = 64;
        SameDiff encoderModel = buildTinySpeakerEncoderModel(samples, embDim);
        SpeakerEncoder encoder = SpeakerEncoder.create(encoderModel, embDim);

        INDArray waveform = Nd4j.randn(DataType.FLOAT, 1, samples);
        INDArray embedding = encoder.encode(waveform);

        assertNotNull(embedding, "embedding must not be null");
        assertEquals(2, embedding.rank(), "embedding should be rank 2 [batch, embDim]");
        assertEquals(1, embedding.size(0), "batch dimension should be 1");
        assertEquals(embDim, embedding.size(1), "second dimension should equal embeddingDim");
    }

    @Test
    @DisplayName("testSpeakerEncoderEncodeRank1Input")
    public void testSpeakerEncoderEncodeRank1Input() {
        int samples = 100;
        int embDim = 64;
        SameDiff encoderModel = buildTinySpeakerEncoderModel(samples, embDim);
        SpeakerEncoder encoder = SpeakerEncoder.create(encoderModel, embDim);

        // Rank-1 input: should be auto-reshaped to [1, samples]
        INDArray waveform = Nd4j.randn(DataType.FLOAT, samples);
        INDArray embedding = encoder.encode(waveform);

        assertNotNull(embedding);
        assertEquals(embDim, embedding.size(1));
    }

    // =========================================================================
    // 6. TtsPipeline creation — frozen encoder
    // =========================================================================

    @Test
    @DisplayName("testTtsPipelineCreation")
    public void testTtsPipelineCreation() {
        int melBins = 8;
        int frames = 16;
        SameDiff model = buildTinyTtsModel(melBins, frames);

        TtsFineTuneConfig ttsConfig = TtsFineTuneConfig.voiceCloning();
        TtsTrainingConfig trainConfig = TtsTrainingConfig.builder()
                .numEpochs(1)
                .batchSize(1)
                .build();

        TtsTrainingPipeline pipeline = TtsTrainingPipeline.create(model, ttsConfig, trainConfig);

        assertNotNull(pipeline);
        assertNotNull(pipeline.getModel());
        assertNull(pipeline.getPeftModel(), "no PEFT configured → peftModel should be null");

        // Encoder variables should be frozen (converted to CONSTANT by freezeVariables)
        SameDiff sd = pipeline.getModel();
        for (String varName : sd.variableNames()) {
            if (varName.startsWith("encoder.")) {
                VariableType type = sd.getVariable(varName).getVariableType();
                assertEquals(VariableType.CONSTANT, type,
                        "encoder variable '" + varName + "' should be CONSTANT after freeze");
            }
        }
    }

    // =========================================================================
    // 7. TtsPipeline creation with LoRA
    // =========================================================================

    @Test
    @DisplayName("testTtsPipelineWithLoRA")
    public void testTtsPipelineWithLoRA() {
        int melBins = 8;
        int frames = 16;
        SameDiff model = buildTinyTtsModel(melBins, frames);

        LoraConfig lora = LoraConfig.builder()
                .r(4)
                .loraAlpha(8)
                .loraDropout(0.0)
                .taskType(TaskType.SEQ_2_SEQ_LM)
                .build();

        TtsFineTuneConfig ttsConfig = TtsFineTuneConfig.builder().build();
        TtsTrainingConfig trainConfig = TtsTrainingConfig.builder()
                .numEpochs(1)
                .batchSize(1)
                .peftConfig(lora)
                .build();

        TtsTrainingPipeline pipeline = TtsTrainingPipeline.create(model, ttsConfig, trainConfig);

        assertNotNull(pipeline.getPeftModel(), "PEFT configured → peftModel should not be null");
        assertNotNull(pipeline.getModel());
    }

    // =========================================================================
    // 8. AudioDataProcessor integration — mel spectrogram shape
    // =========================================================================

    @Test
    @DisplayName("testAudioProcessorIntegration")
    public void testAudioProcessorIntegration() {
        TtsFineTuneConfig config = TtsFineTuneConfig.builder()
                .sampleRate(22050)
                .fftSize(512)
                .hopLength(128)
                .winLength(512)
                .numMelBins(40)
                .fMin(0.0)
                .fMax(8000.0)
                .normalization(TtsFineTuneConfig.AudioNormalization.NONE)
                .build();

        AudioDataProcessor processor = new AudioDataProcessor(config);

        // 1 second of audio at 22050 Hz
        int samples = 22050;
        INDArray waveform = Nd4j.randn(DataType.FLOAT, 1, samples);

        INDArray mel = processor.process(waveform);

        assertNotNull(mel, "mel spectrogram must not be null");
        // Expected frames = floor((samples - fftSize) / hopLength) + 1
        int expectedFrames = (samples - 512) / 128 + 1;
        // Shape should be [1, numMelBins, frames] or [numMelBins, frames]
        boolean shapeOk = (mel.rank() == 2 && mel.size(0) == 40)
                || (mel.rank() == 3 && mel.size(0) == 1 && mel.size(1) == 40);
        assertTrue(shapeOk,
                "mel shape should be [40, frames] or [1, 40, frames], got: "
                        + java.util.Arrays.toString(mel.shape()));
    }

    // =========================================================================
    // 9. TtsTrainingConfig mergeAndExport without PEFT returns same model
    // =========================================================================

    @Test
    @DisplayName("testMergeAndExportNoPeft")
    public void testMergeAndExportNoPeft() {
        SameDiff model = buildTinyTtsModel(8, 16);
        TtsFineTuneConfig ttsConfig = TtsFineTuneConfig.builder().build();
        TtsTrainingConfig trainConfig = TtsTrainingConfig.builder().numEpochs(1).build();

        TtsTrainingPipeline pipeline = TtsTrainingPipeline.create(model, ttsConfig, trainConfig);
        SameDiff exported = pipeline.mergeAndExport();

        assertSame(pipeline.getModel(), exported,
                "Without PEFT, mergeAndExport should return the same model instance");
    }
}
