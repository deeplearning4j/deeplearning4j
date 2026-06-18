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

package org.eclipse.deeplearning4j.vlm.training;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * A single training example for Vision-Language Model (VLM) fine-tuning.
 *
 * <p>Each example consists of:
 * <ul>
 *   <li><b>imageFeatures</b> — the raw image tensor ({@code [C, H, W]}) or pre-encoded
 *       vision features ({@code [numPatches, featureDim]}). The pipeline will pass
 *       raw images through the vision encoder if the shape indicates pixel data
 *       (rank 3 with channels in the first dimension), or use pre-encoded features
 *       directly (rank 2).</li>
 *   <li><b>prompt</b> — the user-side text prompt (instruction / question).</li>
 *   <li><b>response</b> — the expected assistant response, which is the training
 *       target. Loss is computed only on response tokens, not on prompt tokens.</li>
 *   <li><b>systemMessage</b> — optional system message prepended before the user
 *       prompt. When null, no system message is included.</li>
 * </ul>
 *
 * <p>Example:</p>
 * <pre>{@code
 * INDArray image = Nd4j.rand(DataType.FLOAT, 3, 224, 224);
 *
 * VlmTrainingExample example = VlmTrainingExample.builder()
 *     .imageFeatures(image)
 *     .prompt("Describe what you see in this image.")
 *     .response("The image shows a white cat sitting on a wooden floor.")
 *     .systemMessage("You are a helpful vision assistant.")
 *     .build();
 * }</pre>
 *
 * @author Adam Gibson
 * @see VlmTrainingPipeline
 */
@Data
@Builder
public class VlmTrainingExample implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Raw image tensor in {@code [channels, height, width]} layout, or pre-encoded
     * patch features in {@code [numPatches, featureDim]} layout.
     *
     * <p>When the rank is 3 the pipeline treats this as pixel data and passes it
     * through the vision encoder. When the rank is 2 the pipeline treats it as
     * already-encoded patch features and skips the vision encoder.</p>
     *
     * <p>Must not be null.</p>
     */
    @NonNull
    private final INDArray imageFeatures;

    /**
     * User-side text prompt (instruction or question).
     * This is formatted according to the chat template and its tokens are excluded
     * from the training loss (loss mask = 0 on prompt tokens).
     *
     * <p>Must not be null or empty.</p>
     */
    @NonNull
    private final String prompt;

    /**
     * Expected assistant response — the training target.
     * Loss is computed only on these tokens (loss mask = 1 on response tokens).
     *
     * <p>Must not be null or empty.</p>
     */
    @NonNull
    private final String response;

    /**
     * Optional system message prepended before the user prompt.
     * When null or empty, no system message is included in the formatted input.
     */
    private final String systemMessage;

    /**
     * Return whether this example contains pre-encoded patch features rather than
     * raw pixel data. Pre-encoded features have rank 2; raw pixel images have rank 3.
     *
     * @return true if {@link #imageFeatures} is already encoded (rank-2 tensor)
     */
    public boolean isPreEncoded() {
        return imageFeatures.rank() == 2;
    }

    /**
     * Return whether this example has a system message.
     *
     * @return true if {@code systemMessage} is non-null and non-empty
     */
    public boolean hasSystemMessage() {
        return systemMessage != null && !systemMessage.isEmpty();
    }
}
