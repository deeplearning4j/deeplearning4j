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

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A single (audio waveform, transcript) training example for TTS fine-tuning.
 *
 * <p>Holds the raw audio waveform (time domain), the ground-truth text transcript,
 * an optional pre-computed speaker embedding for multi-speaker models, and the
 * sample rate of the audio.
 *
 * <p>Example:
 * <pre>
 * INDArray waveform = AudioLoader.load(new File("sample.wav"));
 * TtsTrainingExample ex = TtsTrainingExample.builder()
 *     .audioWaveform(waveform)
 *     .text("Hello, this is a test.")
 *     .sampleRate(22050)
 *     .build();
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TtsTrainingPipeline
 */
@Data
@Builder
public class TtsTrainingExample {

    /**
     * Raw audio waveform in time domain.
     * Shape: [samples] (rank 1) or [1, samples] (rank 2, batch of 1).
     * Must not be null.
     */
    @NonNull
    private final INDArray audioWaveform;

    /**
     * Ground-truth text transcript corresponding to the audio.
     * Must not be null.
     */
    @NonNull
    private final String text;

    /**
     * Optional pre-computed speaker embedding for multi-speaker TTS.
     * Shape: [embeddingDim].
     * When null, the pipeline uses a default speaker or ignores speaker conditioning.
     */
    private final INDArray speakerEmbedding;

    /**
     * Audio sample rate in Hz.
     * The pipeline will resample if this differs from the target sample rate in
     * {@link org.nd4j.autodiff.samediff.config.TtsFineTuneConfig#getSampleRate()}.
     * Default: 22050
     */
    @Builder.Default
    private final int sampleRate = 22050;
}
