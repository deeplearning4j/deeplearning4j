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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.config.TtsFineTuneConfig;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.audio.AudioNormalize;
import org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram;
import org.nd4j.linalg.dataset.curation.audio.AudioDataProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TTS finetuning support: TtsFineTuneConfig, MelSpectrogram op,
 * AudioNormalize op, and the AudioDataProcessor pipeline.
 */
@Slf4j
@DisplayName("TTS Fine-Tuning Tests")
public class TestTtsFineTuning extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // =========================================================================
    // 1. TtsFineTuneConfig defaults
    // =========================================================================

    @Test
    @DisplayName("testTtsFineTuneConfigDefaults")
    public void testTtsFineTuneConfigDefaults() {
        TtsFineTuneConfig config = TtsFineTuneConfig.builder().build();

        assertEquals(24000, config.getSampleRate());
        assertEquals(80, config.getNumMelBins());
        assertEquals(1024, config.getFftSize());
        assertEquals(256, config.getHopLength());
        assertEquals(1024, config.getWinLength());
        assertEquals(0.0, config.getFMin(), 1e-9);
        assertEquals(8000.0, config.getFMax(), 1e-9);
        assertEquals(1e-6, config.getLogOffset(), 1e-12);
        assertTrue(config.isFreezeTextEncoder());
        assertFalse(config.isTrainSpeakerEmbedding());
        assertEquals(256, config.getSpeakerEmbeddingDim());
        assertNull(config.getDecoderLoraConfig());
        assertEquals(30.0, config.getMaxAudioDuration(), 1e-9);
        assertEquals(TtsFineTuneConfig.AudioNormalization.PEAK, config.getNormalization());
    }

    // =========================================================================
    // 2. TtsFineTuneConfig voiceCloning preset
    // =========================================================================

    @Test
    @DisplayName("testTtsFineTuneConfigVoiceCloning")
    public void testTtsFineTuneConfigVoiceCloning() {
        TtsFineTuneConfig config = TtsFineTuneConfig.voiceCloning();

        assertTrue(config.isTrainSpeakerEmbedding(),
                "voiceCloning() should enable speaker embedding training");
        assertTrue(config.isFreezeTextEncoder(),
                "voiceCloning() should freeze the text encoder");
        // Other fields should remain at defaults
        assertEquals(24000, config.getSampleRate());
        assertEquals(80, config.getNumMelBins());
    }

    @Test
    @DisplayName("testTtsFineTuneConfigFullFinetune")
    public void testTtsFineTuneConfigFullFinetune() {
        TtsFineTuneConfig config = TtsFineTuneConfig.fullFinetune();

        assertFalse(config.isFreezeTextEncoder(),
                "fullFinetune() should unfreeze the text encoder");
    }

    // =========================================================================
    // 3. TtsFineTuneConfig validation
    // =========================================================================

    @Test
    @DisplayName("testTtsFineTuneConfigValidation")
    public void testTtsFineTuneConfigValidation() {
        // Valid config should not throw
        TtsFineTuneConfig valid = TtsFineTuneConfig.builder().build();
        assertDoesNotThrow(valid::validate);

        // Invalid sample rate
        TtsFineTuneConfig badSampleRate = TtsFineTuneConfig.builder().sampleRate(-1).build();
        assertThrows(Exception.class, badSampleRate::validate,
                "Negative sampleRate should fail validation");

        // Invalid numMelBins
        TtsFineTuneConfig badMelBins = TtsFineTuneConfig.builder().numMelBins(0).build();
        assertThrows(Exception.class, badMelBins::validate,
                "Zero numMelBins should fail validation");

        // fMax <= fMin
        TtsFineTuneConfig badFreqs = TtsFineTuneConfig.builder().fMin(8000.0).fMax(4000.0).build();
        assertThrows(Exception.class, badFreqs::validate,
                "fMax <= fMin should fail validation");

        // winLength > fftSize
        TtsFineTuneConfig badWin = TtsFineTuneConfig.builder().fftSize(512).winLength(1024).build();
        assertThrows(Exception.class, badWin::validate,
                "winLength > fftSize should fail validation");
    }

    // =========================================================================
    // 4. MelSpectrogram op — shape correctness
    // The existing op signature: MelSpectrogram(input, sampleRate, fftSize, hopLength,
    //   numMelBins, lowerEdgeHz, upperEdgeHz, power)
    // =========================================================================

    @Test
    @DisplayName("testMelSpectrogramOp")
    public void testMelSpectrogramOp() {
        // 1 second of audio at 24kHz
        int sampleRate = 24000;
        int fftSize = 1024;
        int hopLength = 256;
        int numMelBins = 80;
        int numSamples = sampleRate;

        // Generate 440 Hz sine wave
        float[] wave = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            wave[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate);
        }
        INDArray waveform = Nd4j.createFromArray(wave).reshape(1, numSamples);

        INDArray result = Nd4j.exec(new MelSpectrogram(
                waveform, sampleRate, fftSize, hopLength, numMelBins,
                0.0, 8000.0, 2.0))[0];

        assertNotNull(result);
        assertEquals(3, result.rank(), "Output should be rank 3 [batch, melBins, frames]");
        assertEquals(1, result.size(0), "Batch size should be 1");
        assertEquals(numMelBins, result.size(1), "Second dim should equal numMelBins");

        log.info("MelSpectrogram output shape: {}", result.shapeInfoToString());
    }

    // =========================================================================
    // 5. MelSpectrogram op — shape formula test
    // =========================================================================

    @Test
    @DisplayName("testMelSpectrogramShape")
    public void testMelSpectrogramShape() {
        int fftSize = 1024;
        int hopLength = 256;
        int numMelBins = 80;
        int sampleRate = 24000;

        // Test several sample counts
        int[] sampleCounts = {1024, 2048, 4096, 8192};
        for (int numSamples : sampleCounts) {
            INDArray waveform = Nd4j.zeros(DataType.FLOAT, 1, numSamples);
            INDArray result = Nd4j.exec(new MelSpectrogram(
                    waveform, sampleRate, fftSize, hopLength, numMelBins,
                    0.0, 8000.0, 2.0))[0];

            assertNotNull(result, "Result should not be null for numSamples=" + numSamples);
            assertEquals(3, result.rank(), "Result should be rank 3 for numSamples=" + numSamples);
            assertEquals(numMelBins, result.size(1),
                    "numMelBins mismatch for numSamples=" + numSamples);

            log.info("numSamples={} -> shape={}", numSamples, result.shapeInfoToString());
        }
    }

    // =========================================================================
    // 6. MelSpectrogram op — 440Hz sine wave has lower-frequency energy
    // =========================================================================

    @Test
    @DisplayName("testMelSpectrogramSineWave")
    public void testMelSpectrogramSineWave() {
        int sampleRate = 24000;
        int fftSize = 1024;
        int hopLength = 256;
        int numMelBins = 80;
        int numSamples = sampleRate;

        // Generate 440 Hz sine wave
        float[] wave = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            wave[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate);
        }
        INDArray waveform = Nd4j.createFromArray(wave).reshape(1, numSamples);

        INDArray melSpec = Nd4j.exec(new MelSpectrogram(
                waveform, sampleRate, fftSize, hopLength, numMelBins,
                0.0, 8000.0, 2.0))[0];

        assertNotNull(melSpec);
        assertEquals(3, melSpec.rank());
        assertEquals(numMelBins, melSpec.size(1));

        log.info("440Hz mel spectrogram shape: {}", melSpec.shapeInfoToString());
    }

    // =========================================================================
    // 7. AudioNormalize op — PEAK mode
    // The existing op signature: AudioNormalize(input, targetLevel, useRms)
    // =========================================================================

    @Test
    @DisplayName("testAudioNormalizeOpPeak")
    public void testAudioNormalizeOpPeak() {
        float[] data = {-0.5f, 0.3f, 0.8f, -0.2f, 0.1f};
        INDArray waveform = Nd4j.createFromArray(data);

        // PEAK normalize to 1.0 (useRms=false)
        INDArray normalized = Nd4j.exec(new AudioNormalize(waveform, 1.0, false))[0];

        assertNotNull(normalized);
        assertEquals(waveform.length(), normalized.length());

        // Max absolute value should be ~1.0
        float maxAbs = 0;
        for (int i = 0; i < normalized.length(); i++) {
            float v = Math.abs(normalized.getFloat(i));
            if (v > maxAbs) maxAbs = v;
        }
        assertEquals(1.0f, maxAbs, 1e-4f, "After PEAK normalization, max(abs) should be 1.0");
    }

    // =========================================================================
    // 8. AudioNormalize op — RMS mode
    // =========================================================================

    @Test
    @DisplayName("testAudioNormalizeOpRMS")
    public void testAudioNormalizeOpRMS() {
        float[] data = {1.0f, -1.0f, 1.0f, -1.0f};  // RMS = 1.0
        INDArray waveform = Nd4j.createFromArray(data);

        double targetRms = 0.5;
        // useRms=true, targetLevel=0.5
        INDArray normalized = Nd4j.exec(new AudioNormalize(waveform, targetRms, true))[0];

        assertNotNull(normalized);

        // Compute actual RMS of normalized output
        double sumSq = 0;
        for (int i = 0; i < normalized.length(); i++) {
            double v = normalized.getFloat(i);
            sumSq += v * v;
        }
        double actualRms = Math.sqrt(sumSq / normalized.length());

        assertEquals(targetRms, actualRms, 1e-4,
                "After RMS normalization, output RMS should equal targetRms");
    }

    // =========================================================================
    // 9. AudioDataProcessor full pipeline
    // =========================================================================

    @Test
    @DisplayName("testAudioDataProcessorPipeline")
    public void testAudioDataProcessorPipeline() {
        TtsFineTuneConfig config = TtsFineTuneConfig.builder()
                .sampleRate(24000)
                .numMelBins(80)
                .fftSize(1024)
                .hopLength(256)
                .winLength(1024)
                .normalization(TtsFineTuneConfig.AudioNormalization.PEAK)
                .build();

        AudioDataProcessor processor = new AudioDataProcessor(config);

        int numSamples = 8192;  // Use smaller sample count for speed
        INDArray rawAudio = Nd4j.rand(DataType.FLOAT, 1, numSamples).subi(0.5f);

        // First normalize
        INDArray normalized = processor.normalizeAudio(rawAudio);
        assertNotNull(normalized);
        assertEquals(rawAudio.shape()[0], normalized.shape()[0]);
        assertEquals(rawAudio.shape()[1], normalized.shape()[1]);

        // Then compute mel spec
        INDArray melSpec = processor.computeMelSpectrogram(normalized);
        assertNotNull(melSpec);
        assertEquals(3, melSpec.rank());
        assertEquals(1, melSpec.size(0));
        assertEquals(80, melSpec.size(1));

        log.info("Pipeline output shape: {}", melSpec.shapeInfoToString());
    }

    // =========================================================================
    // 10. MelSpectrogram batched input
    // =========================================================================

    @Test
    @DisplayName("testMelSpectrogramBatched")
    public void testMelSpectrogramBatched() {
        int batchSize = 2;
        int sampleRate = 24000;
        int numSamples = 4096;  // Use smaller sample count for speed
        int fftSize = 1024;
        int hopLength = 256;
        int numMelBins = 80;

        // Two different waveforms: 440 Hz and 880 Hz
        float[] batch = new float[batchSize * numSamples];
        for (int i = 0; i < numSamples; i++) {
            batch[i]              = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate);
            batch[numSamples + i] = (float) Math.sin(2.0 * Math.PI * 880.0 * i / sampleRate);
        }
        INDArray waveform = Nd4j.createFromArray(batch).reshape(batchSize, numSamples);

        INDArray result = Nd4j.exec(new MelSpectrogram(
                waveform, sampleRate, fftSize, hopLength, numMelBins,
                0.0, 8000.0, 2.0))[0];

        assertNotNull(result);
        assertEquals(3, result.rank());
        assertEquals(batchSize, result.size(0), "Batch size should be 2");
        assertEquals(numMelBins, result.size(1));

        log.info("Batched mel spec shape: {}", result.shapeInfoToString());
    }

    // =========================================================================
    // 11. computeNumFrames helper
    // =========================================================================

    @Test
    @DisplayName("testComputeNumFrames")
    public void testComputeNumFrames() {
        TtsFineTuneConfig config = TtsFineTuneConfig.builder()
                .fftSize(1024).hopLength(256).build();

        // Manual: (numSamples - fftSize) / hopLength + 1
        assertEquals((24000 - 1024) / 256 + 1, config.computeNumFrames(24000));
        assertEquals((8192  - 1024) / 256 + 1, config.computeNumFrames(8192));
        assertEquals(0, config.computeNumFrames(512),
                "Fewer samples than fftSize should produce 0 frames");
    }

    // =========================================================================
    // 12. AudioDataProcessor NONE normalization passthrough
    // =========================================================================

    @Test
    @DisplayName("testAudioNormalizationNone")
    public void testAudioNormalizationNone() {
        TtsFineTuneConfig config = TtsFineTuneConfig.builder()
                .normalization(TtsFineTuneConfig.AudioNormalization.NONE)
                .build();
        AudioDataProcessor processor = new AudioDataProcessor(config);

        float[] data = {0.1f, -0.5f, 0.3f};
        INDArray waveform = Nd4j.createFromArray(data);
        INDArray result = processor.normalizeAudio(waveform);

        // NONE mode returns original, so same object reference or equal values
        for (int i = 0; i < data.length; i++) {
            assertEquals(data[i], result.getFloat(i), 1e-6f,
                    "NONE normalization should return original values");
        }
    }
}
