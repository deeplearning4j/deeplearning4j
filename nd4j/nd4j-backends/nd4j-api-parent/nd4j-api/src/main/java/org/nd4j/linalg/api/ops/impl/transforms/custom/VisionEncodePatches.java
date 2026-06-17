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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * ViT-style patch extraction for Vision-Language Model training.
 *
 * <p>Splits an image into non-overlapping square patches and flattens each patch
 * into a 1-D feature vector. This is the first step of vision encoding used in
 * VLMs such as LLaVA, Idefics, and SmolDocling (ViT-style backbones).</p>
 *
 * <p>Given an image of shape {@code [batch, channels, height, width]} and a
 * {@code patchSize}, the output is:</p>
 * <ul>
 *   <li>Output 0: {@code [batch, numPatches, patchDim]} where
 *       {@code numPatches = (height/patchSize) * (width/patchSize)} and
 *       {@code patchDim = channels * patchSize * patchSize}</li>
 *   <li>Output 1: INT64 scalar — {@code numPatches} per image</li>
 * </ul>
 *
 * <p>The image height and width must both be divisible by {@code patchSize}.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * // [1, 3, 224, 224] image with patchSize=14 → [1, 256, 588] patches
 * // numPatches = (224/14)^2 = 256, patchDim = 3*14*14 = 588
 * INDArray image = Nd4j.rand(DataType.FLOAT, 1, 3, 224, 224);
 * INDArray[] out = Nd4j.exec(new VisionEncodePatches(image, 14));
 * INDArray patches = out[0];  // shape [1, 256, 588]
 * INDArray nPatches = out[1]; // scalar 256
 *
 * // In SameDiff:
 * SameDiff sd = SameDiff.create();
 * SDVariable img = sd.placeHolder("image", DataType.FLOAT, 1, 3, 224, 224);
 * SDVariable[] out = new VisionEncodePatches(sd, img, 14).outputVariables();
 * SDVariable patches = out[0];
 * SDVariable numPatches = out[1];
 * }</pre>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class VisionEncodePatches extends DynamicCustomOp {

    private int patchSize = 14;
    private int hiddenDim = 0;

    /**
     * Construct a patch-extraction op.
     *
     * @param image     input image {@code [batch, channels, height, width]}
     * @param patchSize size of each square patch (e.g. 14); height and width must
     *                  each be divisible by this value
     */
    public VisionEncodePatches(INDArray image, int patchSize) {
        this(image, patchSize, 0);
    }

    /**
     * Construct a patch-extraction op with an explicit hidden dimension hint.
     *
     * @param image     input image {@code [batch, channels, height, width]}
     * @param patchSize size of each square patch (e.g. 14)
     * @param hiddenDim reserved for downstream projection ops (pass 0 if unused)
     */
    public VisionEncodePatches(INDArray image, int patchSize, int hiddenDim) {
        super(wrapFilterNull(image), null);
        this.patchSize = patchSize;
        this.hiddenDim = hiddenDim;
        addIArgument(patchSize, hiddenDim);
    }

    /**
     * Construct a SameDiff patch-extraction op.
     *
     * @param sd        the SameDiff instance
     * @param image     image variable {@code [batch, channels, height, width]}
     * @param patchSize size of each square patch (e.g. 14)
     */
    public VisionEncodePatches(SameDiff sd, SDVariable image, int patchSize) {
        this(sd, image, patchSize, 0);
    }

    /**
     * Construct a SameDiff patch-extraction op with a hidden dimension hint.
     *
     * @param sd        the SameDiff instance
     * @param image     image variable {@code [batch, channels, height, width]}
     * @param patchSize size of each square patch (e.g. 14)
     * @param hiddenDim reserved for downstream projection ops (pass 0 if unused)
     */
    public VisionEncodePatches(SameDiff sd, SDVariable image, int patchSize, int hiddenDim) {
        super(null, sd, new SDVariable[]{image}, false);
        this.patchSize = patchSize;
        this.hiddenDim = hiddenDim;
        addIArgument(patchSize, hiddenDim);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.patchSize = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.hiddenDim = iArguments.get(1).intValue();
        }
    }

    @Override
    public String opName() {
        return "vision_encode_patches";
    }

    /**
     * Returns the patch size used by this op.
     *
     * @return patch size
     */
    public int getPatchSize() {
        return patchSize;
    }

    /**
     * Returns the hidden dim hint (may be 0 if unused).
     *
     * @return hidden dim
     */
    public int getHiddenDim() {
        return hiddenDim;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(!dataTypes.isEmpty(),
                "Expected at least 1 input data type, got: %s", dataTypes);
        DataType inputType = dataTypes.get(0);
        Preconditions.checkState(inputType.isFPType(),
                "Input data type must be a floating point type for vision_encode_patches, got: %s",
                inputType);
        return Arrays.asList(inputType, DataType.INT64);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
