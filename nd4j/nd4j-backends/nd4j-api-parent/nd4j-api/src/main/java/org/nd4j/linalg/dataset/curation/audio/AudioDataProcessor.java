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

package org.nd4j.linalg.dataset.curation.audio;

import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.config.TtsFineTuneConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.audio.AudioNormalize;
import org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Audio preprocessing pipeline for TTS finetuning.
 *
 * Wraps the mel_spectrogram and audio_normalize native ops with a convenient
 * Java API driven by {@link TtsFineTuneConfig}.
 *
 * The full pipeline:
 * <ol>
 *   <li>Optionally normalize the waveform (PEAK or RMS)</li>
 *   <li>Compute log mel spectrogram</li>
 * </ol>
 *
 * Example:
 * <pre>
 * TtsFineTuneConfig config = TtsFineTuneConfig.voiceCloning();
 * AudioDataProcessor processor = new AudioDataProcessor(config);
 *
 * INDArray rawAudio = Nd4j.rand(DataType.FLOAT, 1, 24000); // 1 second at 24kHz
 * INDArray melSpec = processor.process(rawAudio);
 * // melSpec shape: [1, numMelBins, numFrames]
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TtsFineTuneConfig
 * @see MelSpectrogram
 * @see AudioNormalize
 */
public class AudioDataProcessor {

    @Getter
    private final TtsFineTuneConfig config;

    /**
     * Create an AudioDataProcessor with the given TTS configuration.
     *
     * @param config TTS finetuning configuration (must not be null)
     */
    public AudioDataProcessor(@NonNull TtsFineTuneConfig config) {
        config.validate();
        this.config = config;
    }

    /**
     * Compute the log mel spectrogram from a raw audio waveform.
     *
     * @param waveform input waveform [batch, samples] or [samples] as FLOAT
     * @return mel spectrogram [batch, numMelBins, numFrames]
     */
    public INDArray computeMelSpectrogram(@NonNull INDArray waveform) {
        Preconditions.checkState(waveform.rank() == 1 || waveform.rank() == 2,
                "waveform must be rank 1 or 2, got rank %s", waveform.rank());

        return Nd4j.exec(new MelSpectrogram(
                waveform,
                config.getSampleRate(),
                config.getFftSize(),
                config.getHopLength(),
                config.getNumMelBins(),
                config.getFMin(),
                config.getFMax(),
                2.0  // power=2 (power spectrogram)
        ))[0];
    }

    /**
     * Normalize an audio waveform according to the configured normalization strategy.
     *
     * If normalization is NONE, the original waveform is returned unchanged.
     *
     * @param waveform input waveform [batch, samples] or [samples]
     * @return normalized waveform (same shape)
     */
    public INDArray normalizeAudio(@NonNull INDArray waveform) {
        Preconditions.checkState(waveform.rank() == 1 || waveform.rank() == 2,
                "waveform must be rank 1 or 2, got rank %s", waveform.rank());

        if (config.getNormalization() == TtsFineTuneConfig.AudioNormalization.NONE) {
            return waveform;
        }

        boolean useRms = (config.getNormalization() == TtsFineTuneConfig.AudioNormalization.RMS);
        double targetLevel = 1.0;

        return Nd4j.exec(new AudioNormalize(waveform, targetLevel, useRms))[0];
    }

    /**
     * Run the full audio preprocessing pipeline.
     *
     * Pipeline: normalize -> mel spectrogram
     *
     * @param rawAudio raw audio waveform [batch, samples] or [samples]
     * @return mel spectrogram [batch, numMelBins, numFrames]
     */
    public INDArray process(@NonNull INDArray rawAudio) {
        INDArray normalized = normalizeAudio(rawAudio);
        return computeMelSpectrogram(normalized);
    }

    /**
     * Truncate a waveform to the maximum number of samples based on maxAudioDuration.
     *
     * @param waveform input waveform [batch, samples] or [samples]
     * @return truncated waveform (or original if already within limit)
     */
    public INDArray truncate(@NonNull INDArray waveform) {
        int maxSamples = config.getMaxSamples();
        int rank = waveform.rank();

        long samples = rank == 2 ? waveform.size(1) : waveform.size(0);
        if (samples <= maxSamples) {
            return waveform;
        }

        if (rank == 2) {
            return waveform.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, maxSamples));
        } else {
            return waveform.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.interval(0, maxSamples));
        }
    }

    /**
     * Compute number of output mel frames for a given sample count using the current config.
     *
     * @param numSamples number of audio samples
     * @return number of mel spectrogram frames
     */
    public int computeNumFrames(int numSamples) {
        return config.computeNumFrames(numSamples);
    }
}
