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

package org.eclipse.deeplearning4j.audio.transform;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.audio.AudioNormalize;
import org.nd4j.linalg.api.ops.impl.audio.AudioResample;
import org.nd4j.linalg.api.ops.impl.audio.PreEmphasis;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Audio preprocessing pipeline for preparing audio data for ML models.
 * Supports normalization, resampling, pre-emphasis, and windowing.
 */
@Builder
@Getter
public class AudioPreprocessor {

    @Builder.Default
    private int targetSampleRate = 16000;

    @Builder.Default
    private boolean normalize = true;

    @Builder.Default
    private double normalizeLevel = 1.0;

    @Builder.Default
    private boolean normalizeRms = false;

    @Builder.Default
    private boolean applyPreEmphasis = true;

    @Builder.Default
    private double preEmphasisCoeff = 0.97;

    /**
     * Apply the preprocessing pipeline to an audio waveform.
     *
     * @param audio Audio waveform as INDArray
     * @param sourceSampleRate Sample rate of the input audio
     * @return Preprocessed audio waveform
     */
    public INDArray process(INDArray audio, int sourceSampleRate) {
        INDArray result = audio;

        // Resample if needed
        if (sourceSampleRate != targetSampleRate) {
            result = Nd4j.exec(new AudioResample(result, sourceSampleRate, targetSampleRate))[0];
        }

        // Normalize
        if (normalize) {
            result = Nd4j.exec(new AudioNormalize(result, normalizeLevel, normalizeRms))[0];
        }

        // Pre-emphasis
        if (applyPreEmphasis) {
            result = Nd4j.exec(new PreEmphasis(result, preEmphasisCoeff))[0];
        }

        return result;
    }

    /**
     * Pad or trim audio to a fixed length.
     *
     * @param audio Audio waveform
     * @param targetLength Target number of samples
     * @return Audio padded with zeros or trimmed to targetLength
     */
    public static INDArray padOrTrim(INDArray audio, long targetLength) {
        long currentLength = audio.length();

        if (currentLength == targetLength) {
            return audio;
        } else if (currentLength > targetLength) {
            // Trim
            return audio.get(NDArrayIndex.interval(0, targetLength));
        } else {
            // Pad with zeros
            INDArray padded = Nd4j.zeros(audio.dataType(), targetLength);
            padded.get(NDArrayIndex.interval(0, currentLength)).assign(audio);
            return padded;
        }
    }
}
