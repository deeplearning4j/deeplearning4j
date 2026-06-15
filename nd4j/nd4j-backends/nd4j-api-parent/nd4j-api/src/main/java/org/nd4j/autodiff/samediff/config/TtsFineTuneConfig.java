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

package org.nd4j.autodiff.samediff.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.base.Preconditions;

import java.io.Serializable;

/**
 * Configuration for Text-to-Speech (TTS) model finetuning.
 *
 * Covers mel spectrogram extraction parameters, audio preprocessing settings,
 * and TTS-specific training options such as voice cloning and speaker embedding training.
 *
 * Example usage:
 * <pre>
 * // Voice cloning: freeze text encoder, train speaker embedding
 * TtsFineTuneConfig config = TtsFineTuneConfig.voiceCloning();
 *
 * // Full finetune: unfreeze text encoder
 * TtsFineTuneConfig config = TtsFineTuneConfig.fullFinetune();
 *
 * // Custom
 * TtsFineTuneConfig config = TtsFineTuneConfig.builder()
 *     .sampleRate(22050)
 *     .numMelBins(80)
 *     .fftSize(1024)
 *     .hopLength(256)
 *     .normalization(TtsFineTuneConfig.AudioNormalization.RMS)
 *     .trainSpeakerEmbedding(true)
 *     .build();
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TtsFineTuneConfig implements Serializable {

    /**
     * Audio sampling rate in Hz.
     * Common values: 22050 (standard), 24000 (high-quality TTS).
     * Default: 24000
     */
    @Builder.Default
    private int sampleRate = 24000;

    /**
     * Number of mel filter banks (mel bins) for spectrogram extraction.
     * Common values: 80 (standard), 100 (high-quality).
     * Default: 80
     */
    @Builder.Default
    private int numMelBins = 80;

    /**
     * FFT window size in samples.
     * Must be a power of 2 for efficiency. Common values: 512, 1024, 2048.
     * Default: 1024
     */
    @Builder.Default
    private int fftSize = 1024;

    /**
     * Hop length (stride) between consecutive FFT windows in samples.
     * Determines the time resolution of the spectrogram.
     * numFrames = floor((samples - fftSize) / hopLength) + 1
     * Default: 256
     */
    @Builder.Default
    private int hopLength = 256;

    /**
     * Window length for the FFT analysis window.
     * Must be <= fftSize. If < fftSize, the window is zero-padded.
     * Default: 1024
     */
    @Builder.Default
    private int winLength = 1024;

    /**
     * Minimum frequency (Hz) for the mel filter bank.
     * Frequencies below this threshold are excluded.
     * Default: 0.0
     */
    @Builder.Default
    private double fMin = 0.0;

    /**
     * Maximum frequency (Hz) for the mel filter bank.
     * Should be at most sampleRate / 2 (Nyquist frequency).
     * Default: 8000.0
     */
    @Builder.Default
    private double fMax = 8000.0;

    /**
     * Log offset added before logarithm to prevent log(0).
     * output = log(melEnergy + logOffset)
     * Default: 1e-6
     */
    @Builder.Default
    private double logOffset = 1e-6;

    /**
     * Whether to freeze the text encoder during TTS finetuning.
     * Freezing reduces trainable parameters and is recommended for voice cloning.
     * Default: true
     */
    @Builder.Default
    private boolean freezeTextEncoder = true;

    /**
     * Whether to train speaker embeddings during finetuning.
     * Enable for voice cloning or multi-speaker TTS.
     * Default: false
     */
    @Builder.Default
    private boolean trainSpeakerEmbedding = false;

    /**
     * Dimensionality of speaker embedding vectors.
     * Only used when trainSpeakerEmbedding is true.
     * Default: 256
     */
    @Builder.Default
    private int speakerEmbeddingDim = 256;

    /**
     * LoRA configuration for the TTS decoder.
     * When null, the decoder is fully finetuned (or frozen based on other flags).
     * When set, the decoder uses LoRA with the specified rank and scaling.
     * Default: null (full finetune)
     */
    private LoraConfig decoderLoraConfig;

    /**
     * Maximum audio duration in seconds.
     * Audio samples longer than this will be truncated during preprocessing.
     * Default: 30.0
     */
    @Builder.Default
    private double maxAudioDuration = 30.0;

    /**
     * Audio normalization strategy applied before mel spectrogram extraction.
     * Default: PEAK
     */
    @Builder.Default
    private AudioNormalization normalization = AudioNormalization.PEAK;

    /**
     * Audio normalization modes.
     */
    public enum AudioNormalization {
        /** Normalize to peak amplitude: output = input / max(abs(input)). */
        PEAK,
        /** Normalize to target RMS level. */
        RMS,
        /** No normalization applied. */
        NONE
    }

    /**
     * Create a configuration preset for voice cloning.
     *
     * Freezes the text encoder and enables speaker embedding training.
     * The acoustic model adapts to a target voice while preserving text understanding.
     */
    public static TtsFineTuneConfig voiceCloning() {
        return builder()
                .trainSpeakerEmbedding(true)
                .freezeTextEncoder(true)
                .build();
    }

    /**
     * Create a configuration preset for full model finetuning.
     *
     * Unfreezes all components including the text encoder.
     * Use when you have sufficient data and want maximum model adaptation.
     */
    public static TtsFineTuneConfig fullFinetune() {
        return builder()
                .freezeTextEncoder(false)
                .build();
    }

    /**
     * Create a configuration preset using LoRA for parameter-efficient finetuning.
     *
     * @param loraRank LoRA rank for the decoder (e.g., 16)
     */
    public static TtsFineTuneConfig withLora(int loraRank) {
        LoraConfig lora = LoraConfig.builder()
                .r(loraRank)
                .loraAlpha(loraRank * 2)
                .loraDropout(0.05)
                .build();
        return builder()
                .decoderLoraConfig(lora)
                .freezeTextEncoder(true)
                .build();
    }

    /**
     * Validate the configuration, throwing IllegalStateException on invalid values.
     */
    public void validate() {
        Preconditions.checkState(sampleRate > 0,
                "sampleRate must be positive, got: %s", sampleRate);
        Preconditions.checkState(numMelBins > 0,
                "numMelBins must be positive, got: %s", numMelBins);
        Preconditions.checkState(fftSize > 0,
                "fftSize must be positive, got: %s", fftSize);
        Preconditions.checkState(hopLength > 0,
                "hopLength must be positive, got: %s", hopLength);
        Preconditions.checkState(winLength > 0 && winLength <= fftSize,
                "winLength must be in (0, fftSize], got: %s (fftSize=%s)", winLength, fftSize);
        Preconditions.checkState(fMin >= 0,
                "fMin must be >= 0, got: %s", fMin);
        Preconditions.checkState(fMax > fMin,
                "fMax must be > fMin, got fMax=%s, fMin=%s", fMax, fMin);
        Preconditions.checkState(fMax <= sampleRate / 2.0,
                "fMax must be <= sampleRate/2 (Nyquist), got fMax=%s, Nyquist=%s", fMax, sampleRate / 2.0);
        Preconditions.checkState(maxAudioDuration > 0,
                "maxAudioDuration must be positive, got: %s", maxAudioDuration);
        Preconditions.checkState(speakerEmbeddingDim > 0,
                "speakerEmbeddingDim must be positive, got: %s", speakerEmbeddingDim);
        Preconditions.checkState(logOffset > 0,
                "logOffset must be positive, got: %s", logOffset);
        if (decoderLoraConfig != null) {
            decoderLoraConfig.validate();
        }
    }

    /**
     * Return the maximum number of audio samples based on sampleRate and maxAudioDuration.
     */
    public int getMaxSamples() {
        return (int) (sampleRate * maxAudioDuration);
    }

    /**
     * Compute the number of mel spectrogram frames for a given sample count.
     *
     * @param numSamples number of audio samples
     * @return number of output frames
     */
    public int computeNumFrames(int numSamples) {
        if (numSamples < fftSize) return 0;
        return (numSamples - fftSize) / hopLength + 1;
    }

    /**
     * Return normalization mode as integer for the audio_normalize op.
     * 0 = PEAK, 1 = RMS.
     */
    public int getNormalizationMode() {
        if (normalization == AudioNormalization.RMS) return 1;
        return 0;  // PEAK or NONE both map to 0; NONE is handled by skipping the op
    }
}
