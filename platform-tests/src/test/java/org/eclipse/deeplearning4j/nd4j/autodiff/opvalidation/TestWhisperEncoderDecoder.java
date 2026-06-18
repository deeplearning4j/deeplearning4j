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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.eclipse.deeplearning4j.llm.generation.DecoderInputBuilder;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.audio.WhisperMelSpectrogramOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for:
 * 1. ModelIOConfig encoder-decoder fields and discovery
 * 2. DecoderInputBuilder encoder-decoder input map building
 * 3. Native whisper_mel_spectrogram op
 */
public class TestWhisperEncoderDecoder {

    // ========== ModelIOConfig encoder-decoder tests ==========

    @Test
    public void testModelIOConfigEncoderDecoderDefaults() {
        ModelIOConfig config = ModelIOConfig.builder().build();
        assertEquals("encoder_hidden_states", config.getEncoderHiddenStatesName());
        assertEquals("encoder_attention_mask", config.getEncoderAttentionMaskName());
        assertFalse(config.isEncoderDecoder());
    }

    @Test
    public void testModelIOConfigEncoderDecoderEnabled() {
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .encoderHiddenStatesName("encoder_hidden_states")
                .encoderAttentionMaskName("encoder_attention_mask")
                .build();
        assertTrue(config.isEncoderDecoder());
        assertTrue(config.isEncoderHiddenStates("encoder_hidden_states"));
        assertFalse(config.isEncoderHiddenStates("inputs_embeds"));
        assertTrue(config.isEncoderAttentionMask("encoder_attention_mask"));
        assertFalse(config.isEncoderAttentionMask("attention_mask"));
        assertTrue(config.isEncoderInput("encoder_hidden_states"));
        assertTrue(config.isEncoderInput("encoder_attention_mask"));
        assertFalse(config.isEncoderInput("input_ids"));
    }

    @Test
    public void testModelIOConfigEncoderInputsNotDisposable() {
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .build();
        // Encoder inputs should NOT be disposable per step
        assertFalse(config.isPerStepDisposableInput("encoder_hidden_states"));
        assertFalse(config.isPerStepDisposableInput("encoder_attention_mask"));
        // But attention_mask (decoder's) IS disposable
        assertTrue(config.isPerStepDisposableInput("attention_mask"));
    }

    @Test
    public void testModelIOConfigCustomEncoderNames() {
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .encoderHiddenStatesName("encoder_output")
                .encoderAttentionMaskName("enc_mask")
                .build();
        assertTrue(config.isEncoderHiddenStates("encoder_output"));
        assertFalse(config.isEncoderHiddenStates("encoder_hidden_states"));
        assertTrue(config.isEncoderAttentionMask("enc_mask"));
        assertTrue(config.isEncoderInput("encoder_output"));
        assertTrue(config.isEncoderInput("enc_mask"));
    }

    // ========== DecoderInputBuilder encoder-decoder input map tests ==========

    @Test
    public void testBuildDecoderInputMapWithEncoderOutputs() {
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .inputEmbeddingsName("inputs_embeds")
                .encoderHiddenStatesName("encoder_hidden_states")
                .encoderAttentionMaskName("encoder_attention_mask")
                .build();

        List<String> inputNames = Arrays.asList(
                "inputs_embeds", "encoder_hidden_states", "encoder_attention_mask"
        );

        INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 512);
        INDArray encoderOutputs = Nd4j.randn(DataType.FLOAT, 1, 1500, 512);

        Map<String, INDArray> result = DecoderInputBuilder.buildDecoderInputMap(
                config, inputNames, null,
                embeddings, null,
                0, 1,
                null, 0, 0,
                false, 512,
                null, false,
                encoderOutputs, null);

        // Should contain encoder outputs
        assertNotNull(result.get("encoder_hidden_states"));
        assertSame(encoderOutputs, result.get("encoder_hidden_states"));

        // Should contain auto-generated encoder attention mask (all ones)
        assertNotNull(result.get("encoder_attention_mask"));
        INDArray mask = result.get("encoder_attention_mask");
        assertEquals(2, mask.rank());
        assertEquals(1, mask.size(0));
        assertEquals(1500, mask.size(1));
        // All ones
        assertEquals(1500.0, mask.sumNumber().doubleValue(), 1e-6);

        // Should contain embeddings
        assertNotNull(result.get("inputs_embeds"));
    }

    @Test
    public void testBuildDecoderInputMapWithExplicitEncoderMask() {
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .inputEmbeddingsName("inputs_embeds")
                .encoderHiddenStatesName("encoder_hidden_states")
                .encoderAttentionMaskName("encoder_attention_mask")
                .build();

        List<String> inputNames = Arrays.asList(
                "inputs_embeds", "encoder_hidden_states", "encoder_attention_mask"
        );

        INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 512);
        INDArray encoderOutputs = Nd4j.randn(DataType.FLOAT, 1, 1500, 512);
        INDArray encoderMask = Nd4j.ones(DataType.LONG, 1, 1200); // partial mask

        Map<String, INDArray> result = DecoderInputBuilder.buildDecoderInputMap(
                config, inputNames, null,
                embeddings, null,
                0, 1,
                null, 0, 0,
                false, 512,
                null, false,
                encoderOutputs, encoderMask);

        // Should use the explicit mask, not auto-generated
        assertSame(encoderMask, result.get("encoder_attention_mask"));
    }

    @Test
    public void testBuildDecoderInputMapWithoutEncoderOutputs() {
        // Decoder-only mode: encoder inputs should be absent
        ModelIOConfig config = ModelIOConfig.builder()
                .inputEmbeddingsName("inputs_embeds")
                .build();

        List<String> inputNames = Arrays.asList("inputs_embeds");
        INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 512);

        Map<String, INDArray> result = DecoderInputBuilder.buildDecoderInputMap(
                config, inputNames, null,
                embeddings, null,
                0, 1,
                null, 0, 0,
                false, 512,
                null, false,
                null, null);

        assertNotNull(result.get("inputs_embeds"));
        assertNull(result.get("encoder_hidden_states"));
        assertNull(result.get("encoder_attention_mask"));
    }

    @Test
    public void testBuildDecoderInputMapEncoderOutputsConstantAcrossSteps() {
        // Verify encoder outputs are the same object across multiple "decode steps"
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .inputEmbeddingsName("inputs_embeds")
                .encoderHiddenStatesName("encoder_hidden_states")
                .build();

        List<String> inputNames = Arrays.asList("inputs_embeds", "encoder_hidden_states");
        INDArray encoderOutputs = Nd4j.randn(DataType.FLOAT, 1, 1500, 512);

        // Step 1
        INDArray emb1 = Nd4j.zeros(DataType.FLOAT, 1, 5, 512);
        Map<String, INDArray> result1 = DecoderInputBuilder.buildDecoderInputMap(
                config, inputNames, null,
                emb1, null, 0, 5,
                null, 0, 0, false, 512,
                null, false,
                encoderOutputs, null);

        // Step 2
        INDArray emb2 = Nd4j.zeros(DataType.FLOAT, 1, 1, 512);
        Map<String, INDArray> result2 = DecoderInputBuilder.buildDecoderInputMap(
                config, inputNames, null,
                emb2, null, 5, 1,
                null, 0, 0, false, 512,
                null, false,
                encoderOutputs, null);

        // Same encoder output object in both steps
        assertSame(result1.get("encoder_hidden_states"), result2.get("encoder_hidden_states"));
        assertSame(encoderOutputs, result1.get("encoder_hidden_states"));
    }

    @Test
    public void testBuildDecoderInputMapMixedEncoderAndKvCache() {
        // Full Whisper-like decoder input: embeddings + encoder_hidden + attention_mask
        // (KV cache creation requires a non-null decoder for shape inference, so we test
        // the encoder-related inputs alongside standard inputs without KV cache)
        ModelIOConfig config = ModelIOConfig.builder()
                .encoderDecoder(true)
                .inputEmbeddingsName("inputs_embeds")
                .inputIdsName("input_ids")
                .attentionMaskName("attention_mask")
                .encoderHiddenStatesName("encoder_hidden_states")
                .kvCachePrefix("past_key_values.")
                .build();

        List<String> inputNames = Arrays.asList(
                "inputs_embeds", "input_ids", "attention_mask",
                "encoder_hidden_states"
        );

        INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 512);
        INDArray inputIds = Nd4j.createFromArray(new long[][]{{42}});
        INDArray encoderOutputs = Nd4j.randn(DataType.FLOAT, 1, 1500, 512);

        Map<String, INDArray> result = DecoderInputBuilder.buildDecoderInputMap(
                config, inputNames, null,
                embeddings, inputIds,
                5, 1,
                null, 0, 0,
                false, 512,
                null, false,
                encoderOutputs, null);

        assertNotNull(result.get("inputs_embeds"));
        assertNotNull(result.get("input_ids"));
        assertNotNull(result.get("attention_mask"));
        assertNotNull(result.get("encoder_hidden_states"));
        assertSame(encoderOutputs, result.get("encoder_hidden_states"));

        // attention_mask should be [1, 6] (pastSeqLen=5 + currentSeqLen=1)
        INDArray mask = result.get("attention_mask");
        assertEquals(6, mask.size(1));
    }

    // ========== Native whisper_mel_spectrogram op tests ==========

    @Test
    public void testWhisperMelSpectrogramOpBasic() {
        // Generate a simple sine wave at 440Hz, 16kHz sample rate, 1 second
        int sampleRate = 16000;
        int numSamples = sampleRate; // 1 second
        float[] samples = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            samples[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate);
        }
        INDArray audio = Nd4j.createFromArray(samples).reshape(1, numSamples);

        int nFft = 400;
        int hopLength = 160;
        int numMelBins = 80;
        int targetFrames = 100; // small for test speed

        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                audio, sampleRate, nFft, hopLength, numMelBins, targetFrames,
                0.0, (double) sampleRate / 2.0
        ))[0];

        // Verify output shape: [1, numMelBins, targetFrames]
        assertEquals(3, result.rank());
        assertEquals(1, result.size(0));
        assertEquals(numMelBins, result.size(1));
        assertEquals(targetFrames, result.size(2));

        // Values should be in a reasonable range after log normalization
        // Whisper normalization: (log10(max(x,1e-10)) + 4) / 4
        // Range should be roughly [-1, 1] for typical audio
        double min = result.minNumber().doubleValue();
        double max = result.maxNumber().doubleValue();
        assertTrue(min >= -2.0, "Min value " + min + " should be >= -2.0");
        assertTrue(max <= 2.0, "Max value " + max + " should be <= 2.0");
        // Should not be all zeros (signal has energy)
        assertTrue(result.sumNumber().doubleValue() != 0.0, "Result should not be all zeros");
    }

    @Test
    public void testWhisperMelSpectrogramOpPadding() {
        // Short audio that needs padding to targetFrames
        int sampleRate = 16000;
        int numSamples = 1600; // only 0.1 seconds
        float[] samples = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            samples[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate);
        }
        INDArray audio = Nd4j.createFromArray(samples).reshape(1, numSamples);

        int targetFrames = 100;

        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                audio, sampleRate, 400, 160, 80, targetFrames,
                0.0, 8000.0
        ))[0];

        assertEquals(3, result.rank());
        assertEquals(80, result.size(1));
        assertEquals(targetFrames, result.size(2));
    }

    @Test
    public void testWhisperMelSpectrogramOpWhisperDefaults() {
        // Test with Whisper's actual defaults: 16kHz, N_FFT=400, hop=160, 80 bins, 3000 frames
        int sampleRate = 16000;
        int chunkSamples = 480000; // 30 seconds
        // Generate white noise
        INDArray audio = Nd4j.randn(DataType.FLOAT, 1, chunkSamples).muli(0.1f);

        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                audio, 16000, 400, 160, 80, 3000,
                0.0, 8000.0
        ))[0];

        assertEquals(3, result.rank());
        assertEquals(1, result.size(0));
        assertEquals(80, result.size(1));
        assertEquals(3000, result.size(2));
    }

    @Test
    public void testWhisperMelSpectrogramOp128Bins() {
        // Whisper large-v3 uses 128 mel bins
        int sampleRate = 16000;
        int numSamples = 16000;
        INDArray audio = Nd4j.randn(DataType.FLOAT, 1, numSamples).muli(0.1f);

        int numMelBins = 128;
        int targetFrames = 100;

        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                audio, sampleRate, 400, 160, numMelBins, targetFrames,
                0.0, 8000.0
        ))[0];

        assertEquals(numMelBins, result.size(1));
        assertEquals(targetFrames, result.size(2));
    }

    @Test
    public void testWhisperMelSpectrogramOpRank1Input() {
        // Test with rank-1 (unbatched) input
        int numSamples = 16000;
        INDArray audio = Nd4j.randn(DataType.FLOAT, numSamples).muli(0.1f);

        int targetFrames = 50;

        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                audio, 16000, 400, 160, 80, targetFrames,
                0.0, 8000.0
        ))[0];

        // Rank-1 input → rank-2 output: [numMelBins, targetFrames]
        assertEquals(2, result.rank());
        assertEquals(80, result.size(0));
        assertEquals(targetFrames, result.size(1));
    }

    @Test
    public void testWhisperMelSpectrogramLogNormalization() {
        // Verify the log normalization produces values consistent with Whisper's formula:
        // log10(max(mel, 1e-10)), clamp to max-8, then (x+4)/4
        int sampleRate = 16000;
        int numSamples = 16000;
        // Constant amplitude sine → predictable mel energy
        float[] samples = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            samples[i] = 0.5f * (float) Math.sin(2.0 * Math.PI * 1000.0 * i / sampleRate);
        }
        INDArray audio = Nd4j.createFromArray(samples).reshape(1, numSamples);

        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                audio, sampleRate, 400, 160, 80, 50,
                0.0, 8000.0
        ))[0];

        // After Whisper normalization, values should be finite and bounded
        double min = result.minNumber().doubleValue();
        double max = result.maxNumber().doubleValue();
        assertTrue(Double.isFinite(min), "Min should be finite");
        assertTrue(Double.isFinite(max), "Max should be finite");

        // The normalization is (log10(x) + 4) / 4, with clamping
        // For zero-padded frames, log10(1e-10) = -10, clamped to max-8
        // So minimum possible is around (max-8+4)/4 = (max-4)/4
        // Maximum is around (max+4)/4
        // The range should be within [-1, 1.5] for typical audio
        assertTrue(min >= -2.5, "Min " + min + " too low");
        assertTrue(max <= 2.5, "Max " + max + " too high");
    }
}
