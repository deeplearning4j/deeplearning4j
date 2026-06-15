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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Utility for extracting fixed-dimensional speaker embeddings from audio.
 *
 * <p>Wraps a {@link SameDiff} speaker encoder model (e.g. a d-vector or
 * x-vector network) and exposes a simple {@link #encode(INDArray)} API.
 * The underlying model may be any SameDiff graph whose first input accepts
 * audio waveforms or mel spectrograms and whose first output is a speaker
 * embedding vector.
 *
 * <p>Usage:
 * <pre>
 * SameDiff encoderModel = SameDiff.load(new File("speaker_encoder.sd"), true);
 * SpeakerEncoder encoder = SpeakerEncoder.create(encoderModel, 256);
 *
 * INDArray waveform = Nd4j.rand(DataType.FLOAT, 1, 16000); // 1 sec at 16kHz
 * INDArray embedding = encoder.encode(waveform); // [1, 256]
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TtsTrainingPipeline
 */
@Slf4j
public class SpeakerEncoder {

    /** The underlying SameDiff encoder model. */
    @Getter
    private final SameDiff model;

    /** Dimensionality of the output embedding vector. */
    @Getter
    private final int embeddingDim;

    /** Name of the model's primary input (auto-detected from the graph). */
    private final String inputName;

    /** Name of the model's primary output (auto-detected from the graph). */
    private final String outputName;

    /**
     * Private constructor — use {@link #create(SameDiff, int)}.
     */
    private SpeakerEncoder(SameDiff model, int embeddingDim) {
        this.model = model;
        this.embeddingDim = embeddingDim;

        List<String> inputs = model.inputs();
        Preconditions.checkState(!inputs.isEmpty(),
                "Speaker encoder model has no inputs");
        this.inputName = inputs.get(0);

        List<String> outputs = model.outputs();
        Preconditions.checkState(!outputs.isEmpty(),
                "Speaker encoder model has no outputs");
        this.outputName = outputs.get(0);

        log.debug("SpeakerEncoder: input='{}' output='{}' embeddingDim={}",
                inputName, outputName, embeddingDim);
    }

    /**
     * Create a SpeakerEncoder from a SameDiff model.
     *
     * @param model        SameDiff speaker encoder model. The first graph input
     *                     must accept audio waveform [batch, samples] and the
     *                     first graph output must be the embedding [batch, embeddingDim].
     * @param embeddingDim Expected embedding dimensionality.
     * @return A new SpeakerEncoder wrapping the model.
     */
    public static SpeakerEncoder create(SameDiff model, int embeddingDim) {
        Preconditions.checkNotNull(model, "model must not be null");
        Preconditions.checkState(embeddingDim > 0,
                "embeddingDim must be positive, got: %s", embeddingDim);
        return new SpeakerEncoder(model, embeddingDim);
    }

    /**
     * Encode a raw audio waveform into a speaker embedding.
     *
     * @param audioWaveform Raw waveform [batch, samples] or [samples] as FLOAT.
     * @return Speaker embedding [batch, embeddingDim].
     */
    public INDArray encode(INDArray audioWaveform) {
        Preconditions.checkNotNull(audioWaveform, "audioWaveform must not be null");
        INDArray input = audioWaveform.rank() == 1
                ? audioWaveform.reshape(1, audioWaveform.length())
                : audioWaveform;

        log.debug("SpeakerEncoder.encode: input shape {}", input.shapeInfoToString());

        Map<String, INDArray> result = model.output(
                Collections.singletonMap(inputName, input), outputName);
        INDArray embedding = result.get(outputName);

        log.debug("SpeakerEncoder.encode: embedding shape {}", embedding.shapeInfoToString());
        return embedding;
    }

    /**
     * Encode from either a raw waveform or a pre-computed mel spectrogram.
     *
     * @param input       Audio waveform [batch, samples] or mel spectrogram
     *                    [batch, melBins, frames] as FLOAT.
     * @param preComputed {@code true} if {@code input} is already a mel spectrogram;
     *                    {@code false} if it is a raw waveform.
     * @return Speaker embedding [batch, embeddingDim].
     */
    public INDArray encode(INDArray input, boolean preComputed) {
        Preconditions.checkNotNull(input, "input must not be null");

        INDArray modelInput;
        if (!preComputed && input.rank() == 1) {
            modelInput = input.reshape(1, input.length());
        } else {
            modelInput = input;
        }

        log.debug("SpeakerEncoder.encode(preComputed={}): input shape {}", preComputed,
                modelInput.shapeInfoToString());

        Map<String, INDArray> result = model.output(
                Collections.singletonMap(inputName, modelInput), outputName);
        return result.get(outputName);
    }
}
