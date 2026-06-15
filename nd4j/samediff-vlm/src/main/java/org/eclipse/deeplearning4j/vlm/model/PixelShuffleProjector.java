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

package org.eclipse.deeplearning4j.vlm.model;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.PixelShuffle;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Pixel shuffle projector for vision-language models.
 *
 * <p>Applies spatial compression to vision encoder outputs using pixel unshuffle
 * (space_to_depth), reducing the number of vision tokens by a factor of r^2.
 * This is a key component in several video VLMs:</p>
 * <ul>
 *   <li><b>SmolVLM2</b>: r=3, reducing 729 tokens to 81 per frame (9x compression)</li>
 *   <li><b>Qwen-VL</b>: r=2, reducing tokens by 4x per frame</li>
 * </ul>
 *
 * <p>The pixel unshuffle operation rearranges spatial patches into the channel dimension:
 * [B, C, H, W] -> [B, C*r*r, H/r, W/r], then the result is reshaped into a sequence
 * of compressed tokens for the language model.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * PixelShuffleProjector projector = PixelShuffleProjector.builder()
 *     .shuffleFactor(3)
 *     .build();
 *
 * // visionOutput: [1, tokens, hidden] from vision encoder
 * // Reshape to spatial: [1, hidden, H_patches, W_patches]
 * // Apply pixel unshuffle: [1, hidden*9, H_patches/3, W_patches/3]
 * // Flatten back to sequence: [1, compressed_tokens, hidden*9]
 * INDArray compressed = projector.compress(visionOutput, patchGridH, patchGridW);
 * }</pre>
 *
 * @see PixelShuffle
 */
@Slf4j
@Builder
@Getter
public class PixelShuffleProjector {

    @Builder.Default
    private int shuffleFactor = 3;

    /**
     * Compress vision encoder output using pixel unshuffle.
     *
     * <p>Takes vision encoder output in sequence format [1, seqLen, hidden] where
     * seqLen = patchGridH * patchGridW, reshapes to spatial, applies pixel unshuffle,
     * then reshapes back to sequence format with fewer, wider tokens.</p>
     *
     * @param visionOutput vision encoder output [1, seqLen, hidden]
     * @param patchGridH number of patch rows (H / patch_size)
     * @param patchGridW number of patch columns (W / patch_size)
     * @return compressed tokens [1, seqLen/(r*r), hidden*(r*r)]
     */
    public INDArray compress(INDArray visionOutput, int patchGridH, int patchGridW) {
        long[] shape = visionOutput.shape();
        int batch = (int) shape[0];
        int seqLen = (int) shape[1];
        int hidden = (int) shape[2];

        if (patchGridH % shuffleFactor != 0 || patchGridW % shuffleFactor != 0) {
            throw new IllegalArgumentException(
                    "Patch grid (" + patchGridH + "x" + patchGridW +
                    ") must be divisible by shuffle factor " + shuffleFactor);
        }

        // Reshape from [B, H*W, C] to [B, C, H, W]
        INDArray spatial = visionOutput.reshape(batch, patchGridH, patchGridW, hidden)
                .permute(0, 3, 1, 2);  // [B, hidden, patchGridH, patchGridW]

        // Pixel unshuffle: [B, hidden, H, W] -> [B, hidden*r*r, H/r, W/r]
        PixelShuffle unshuffle = new PixelShuffle(spatial, shuffleFactor, true);
        INDArray[] results = Nd4j.getExecutioner().exec(unshuffle);
        INDArray unshuffled = results[0];

        int newH = patchGridH / shuffleFactor;
        int newW = patchGridW / shuffleFactor;
        int newHidden = hidden * shuffleFactor * shuffleFactor;
        int newSeqLen = newH * newW;

        // Reshape back to sequence: [B, newHidden, newH, newW] -> [B, newSeqLen, newHidden]
        INDArray result = unshuffled.permute(0, 2, 3, 1)  // [B, newH, newW, newHidden]
                .reshape(batch, newSeqLen, newHidden);

        log.info("Pixel unshuffle r={}: [1, {}, {}] -> [1, {}, {}] ({}x token reduction)",
                shuffleFactor, seqLen, hidden, newSeqLen, newHidden,
                shuffleFactor * shuffleFactor);

        return result;
    }

    /**
     * Build pixel unshuffle into a SameDiff graph.
     *
     * @param sd the SameDiff instance
     * @param visionOutput the vision encoder output variable [B, seqLen, hidden]
     * @param patchGridH number of patch rows
     * @param patchGridW number of patch columns
     * @param name output variable name
     * @return the compressed output variable [B, seqLen/(r*r), hidden*(r*r)]
     */
    public SDVariable compressGraph(SameDiff sd, SDVariable visionOutput,
                                     int patchGridH, int patchGridW, String name) {
        int newH = patchGridH / shuffleFactor;
        int newW = patchGridW / shuffleFactor;

        // Reshape to spatial: [B, H*W, C] -> [B, C, H, W]
        SDVariable reshaped = sd.reshape(visionOutput, -1, patchGridH, patchGridW, 0);
        SDVariable spatial = sd.permute(reshaped, 0, 3, 1, 2);

        // space_to_depth: [B, C, H, W] -> [B, C*r*r, H/r, W/r]
        SDVariable unshuffled = sd.cnn().spaceToDepth(spatial, shuffleFactor,
                org.nd4j.enums.DataFormat.NCHW);

        // Reshape back: [B, C*r*r, H/r, W/r] -> [B, (H/r)*(W/r), C*r*r]
        SDVariable permuted = sd.permute(unshuffled, 0, 2, 3, 1);
        return sd.reshape(name, permuted, -1, newH * newW, 0);
    }

}
