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

package org.eclipse.deeplearning4j.audio.feature;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.audio.io.AudioLoader;
import org.eclipse.deeplearning4j.audio.transform.AudioPreprocessor;
import org.eclipse.deeplearning4j.audio.whisper.WhisperConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.audio.WhisperMelSpectrogramOp;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

/**
 * Whisper-specific mel spectrogram extraction using native C++ op.
 * <p>
 * Delegates to the native {@code whisper_mel_spectrogram} op which combines
 * STFT + mel filterbank + Whisper-specific log normalization in a single kernel:
 * <ul>
 *   <li>16kHz sample rate, N_FFT=400, HOP_LENGTH=160</li>
 *   <li>80 or 128 mel bins (depending on model version)</li>
 *   <li>Pad/trim to 3000 frames (30-second chunks)</li>
 *   <li>log10(max(mel, 1e-10)), clamp to max-8.0, then (x + 4.0) / 4.0</li>
 * </ul>
 */
@Slf4j
public class WhisperMelSpectrogram {

    private final WhisperConfig config;

    public WhisperMelSpectrogram(WhisperConfig config) {
        this.config = config;
    }

    /**
     * Extract Whisper mel spectrogram features from raw audio samples.
     *
     * @param audio Audio samples as FLOAT INDArray of shape [samples], at 16kHz
     * @return Mel spectrogram of shape [1, numMelBins, 3000]
     */
    public INDArray extractFeatures(INDArray audio) {
        // Pad or trim to exactly 30 seconds (480000 samples at 16kHz)
        int chunkSamples = config.getChunkLengthSamples();
        INDArray padded = AudioPreprocessor.padOrTrim(audio, chunkSamples);

        // Ensure rank 2 for batch processing: [1, samples]
        if (padded.rank() == 1) {
            padded = padded.reshape(1, padded.length());
        }

        // Run native whisper_mel_spectrogram op (mel + log normalization in one call)
        INDArray result = Nd4j.exec(new WhisperMelSpectrogramOp(
                padded,
                config.getSampleRate(),
                config.getNFft(),
                config.getHopLength(),
                config.getNumMelBins(),
                config.getNumFrames(),
                0.0,   // lower edge Hz
                (double) config.getSampleRate() / 2.0  // upper edge Hz (Nyquist)
        ))[0];

        // Ensure shape is [1, numMelBins, numFrames]
        if (result.rank() == 2) {
            result = result.reshape(1, result.size(0), result.size(1));
        }

        return result.castTo(DataType.FLOAT);
    }

    /**
     * Extract features from a WAV file.
     *
     * @param wavFile Path to the WAV file
     * @return Mel spectrogram of shape [1, numMelBins, 3000]
     * @throws IOException if the file cannot be read
     */
    public INDArray extractFeaturesFromFile(File wavFile) throws IOException {
        INDArray audio = AudioLoader.loadWav(wavFile);
        int sourceSampleRate = AudioLoader.getSampleRate(wavFile);

        // Resample to 16kHz if needed
        if (sourceSampleRate != config.getSampleRate()) {
            AudioPreprocessor preprocessor = AudioPreprocessor.builder()
                    .targetSampleRate(config.getSampleRate())
                    .normalize(false)
                    .applyPreEmphasis(false)
                    .build();
            audio = preprocessor.process(audio, sourceSampleRate);
        }

        return extractFeatures(audio);
    }
}
