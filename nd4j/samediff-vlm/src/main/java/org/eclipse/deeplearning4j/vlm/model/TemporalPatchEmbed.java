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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Temporal patch embedding for video Vision Transformers.
 *
 * <p>Extracts 3D patches from video frames by combining temporal and spatial
 * patch extraction. Used by Qwen3-VL with temporal_patch_size=2, which means
 * pairs of consecutive frames are merged into single patch embeddings.</p>
 *
 * <p>The operation:</p>
 * <pre>
 * Input:  [batch, T, C, H, W]
 * Output: [batch, T/temporal_patch, num_spatial_patches, embed_dim]
 *
 * where:
 *   num_spatial_patches = (H / spatial_patch) * (W / spatial_patch)
 *   embed_dim = C * temporal_patch * spatial_patch * spatial_patch
 * </pre>
 *
 * <p>This is equivalent to a Conv3D with kernel=[temporal_patch, spatial_patch, spatial_patch]
 * and stride=[temporal_patch, spatial_patch, spatial_patch].</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * TemporalPatchEmbed tpe = TemporalPatchEmbed.builder()
 *     .temporalPatchSize(2).spatialPatchSize(16).channels(3).build();
 *
 * // frames: [1, 16, 3, 384, 384]
 * INDArray patches = tpe.extractPatches(frames);
 * // patches: [1, 8, 576, 1536] (8 temporal groups, 24*24 spatial patches, 3*2*16*16 dim)
 * }</pre>
 *
 * @see VideoPreprocessor
 */
@Slf4j
@Builder
@Getter
public class TemporalPatchEmbed {

    @Builder.Default
    private int temporalPatchSize = 2;

    @Builder.Default
    private int spatialPatchSize = 16;

    @Builder.Default
    private int channels = 3;

    /**
     * Extract 3D patches from a video tensor.
     *
     * @param video input tensor [batch, T, C, H, W]
     * @return patches [batch, T/temporalPatch, numSpatialPatches, embedDim]
     */
    public INDArray extractPatches(INDArray video) {
        long[] shape = video.shape();
        int batch = (int) shape[0];
        int T = (int) shape[1];
        int C = (int) shape[2];
        int H = (int) shape[3];
        int W = (int) shape[4];

        if (T % temporalPatchSize != 0) {
            throw new IllegalArgumentException(
                    "Temporal dimension " + T + " must be divisible by temporal_patch_size " + temporalPatchSize);
        }
        if (H % spatialPatchSize != 0 || W % spatialPatchSize != 0) {
            throw new IllegalArgumentException(
                    "Spatial dimensions (" + H + "x" + W + ") must be divisible by spatial_patch_size " + spatialPatchSize);
        }

        int numTemporalGroups = T / temporalPatchSize;
        int numPatchesH = H / spatialPatchSize;
        int numPatchesW = W / spatialPatchSize;
        int numSpatialPatches = numPatchesH * numPatchesW;
        int embedDim = C * temporalPatchSize * spatialPatchSize * spatialPatchSize;

        INDArray result = Nd4j.create(DataType.FLOAT, batch, numTemporalGroups, numSpatialPatches, embedDim);

        for (int b = 0; b < batch; b++) {
            for (int tg = 0; tg < numTemporalGroups; tg++) {
                int tStart = tg * temporalPatchSize;
                int patchIdx = 0;

                for (int ph = 0; ph < numPatchesH; ph++) {
                    for (int pw = 0; pw < numPatchesW; pw++) {
                        int hStart = ph * spatialPatchSize;
                        int wStart = pw * spatialPatchSize;

                        // Extract 3D patch: [temporalPatch, C, spatialPatch, spatialPatch]
                        INDArray patch = video.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.interval(tStart, tStart + temporalPatchSize),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(hStart, hStart + spatialPatchSize),
                                NDArrayIndex.interval(wStart, wStart + spatialPatchSize)
                        );

                        // Flatten to embedDim
                        INDArray flat = patch.reshape(embedDim);

                        result.get(
                                NDArrayIndex.point(b),
                                NDArrayIndex.point(tg),
                                NDArrayIndex.point(patchIdx),
                                NDArrayIndex.all()
                        ).assign(flat);

                        patchIdx++;
                    }
                }
            }
        }

        log.info("Temporal patch embed: [{}, {}, {}, {}, {}] -> [{}, {}, {}, {}]",
                batch, T, C, H, W,
                batch, numTemporalGroups, numSpatialPatches, embedDim);

        return result;
    }

    /**
     * Build temporal patch embedding into a SameDiff graph using Conv3D.
     *
     * @param sd the SameDiff instance
     * @param video input variable [batch, T, C, H, W]
     * @param weight Conv3D weight [embedDim, C, temporalPatch, spatialPatch, spatialPatch]
     * @param name output variable name
     * @return output variable [batch, numTemporalGroups, numSpatialPatches, embedDim]
     */
    public SDVariable buildGraph(SameDiff sd, SDVariable video, SDVariable weight, String name) {
        // Reshape for Conv3D: [B, C, T, H, W] (Conv3D expects channels-first)
        SDVariable permuted = sd.permute(video, 0, 2, 1, 3, 4);

        // Conv3D with stride = kernel size = patch sizes
        // This is the standard way to implement patch extraction
        Conv3DConfig conv3dConfig = Conv3DConfig.builder()
                .dataFormat(Conv3DConfig.NCDHW)
                .kD(temporalPatchSize).kH(spatialPatchSize).kW(spatialPatchSize)
                .sD(temporalPatchSize).sH(spatialPatchSize).sW(spatialPatchSize)
                .pD(0).pH(0).pW(0)
                .dD(1).dH(1).dW(1)
                .build();
        SDVariable conv = sd.cnn().conv3d(name + "_conv3d", permuted, weight, conv3dConfig);

        return conv;
    }

    /**
     * Get the number of output patches for given input dimensions.
     */
    public int getNumPatches(int T, int H, int W) {
        int temporalGroups = T / temporalPatchSize;
        int spatialPatches = (H / spatialPatchSize) * (W / spatialPatchSize);
        return temporalGroups * spatialPatches;
    }

    /**
     * Get the embedding dimension.
     */
    public int getEmbedDim() {
        return channels * temporalPatchSize * spatialPatchSize * spatialPatchSize;
    }

}
