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

import lombok.Builder;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.audio.MFCC;
import org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram;
import org.nd4j.linalg.api.ops.impl.audio.PreEmphasis;
import org.nd4j.linalg.factory.Nd4j;

/**
 * High-level audio feature extraction pipeline.
 * Provides convenient methods for extracting common audio features
 * (MFCC, mel spectrogram) with configurable parameters.
 */
@Builder
@Getter
public class AudioFeatureExtractor {

    @Builder.Default
    private int sampleRate = 22050;

    @Builder.Default
    private int fftSize = 2048;

    @Builder.Default
    private int hopLength = 512;

    @Builder.Default
    private int numMelBins = 128;

    @Builder.Default
    private int numMfcc = 13;

    @Builder.Default
    private double lowerEdgeHz = 0.0;

    @Builder.Default
    private double upperEdgeHz = 8000.0;

    @Builder.Default
    private boolean applyPreEmphasis = true;

    @Builder.Default
    private double preEmphasisCoeff = 0.97;

    /**
     * Extract MFCC features from an audio waveform.
     *
     * @param audio Audio waveform as INDArray of shape [batch, samples] or [samples]
     * @return MFCC coefficients of shape [batch, numMfcc, numFrames] or [numMfcc, numFrames]
     */
    public INDArray extractMfcc(INDArray audio) {
        INDArray input = audio;
        if (applyPreEmphasis) {
            input = Nd4j.exec(new PreEmphasis(audio, preEmphasisCoeff))[0];
        }

        return Nd4j.exec(new MFCC(input, sampleRate, fftSize, hopLength,
                numMelBins, numMfcc, lowerEdgeHz, upperEdgeHz))[0];
    }

    /**
     * Extract mel spectrogram from an audio waveform.
     *
     * @param audio Audio waveform as INDArray of shape [batch, samples] or [samples]
     * @return Mel spectrogram of shape [batch, numMelBins, numFrames] or [numMelBins, numFrames]
     */
    public INDArray extractMelSpectrogram(INDArray audio) {
        INDArray input = audio;
        if (applyPreEmphasis) {
            input = Nd4j.exec(new PreEmphasis(audio, preEmphasisCoeff))[0];
        }

        return Nd4j.exec(new MelSpectrogram(input, sampleRate, fftSize, hopLength,
                numMelBins, lowerEdgeHz, upperEdgeHz, 2.0))[0];
    }

    /**
     * Extract log mel spectrogram (mel spectrogram with log scaling).
     *
     * @param audio Audio waveform as INDArray
     * @return Log mel spectrogram
     */
    public INDArray extractLogMelSpectrogram(INDArray audio) {
        INDArray melSpec = extractMelSpectrogram(audio);
        return Nd4j.math().log(melSpec.add(1e-10));
    }
}
