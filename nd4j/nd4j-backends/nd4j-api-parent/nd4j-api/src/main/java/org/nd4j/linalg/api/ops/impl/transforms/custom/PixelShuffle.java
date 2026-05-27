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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * PixelShuffle op — rearranges elements in a tensor of shape [B, C*r^2, H, W]
 * to [B, C, H*r, W*r] (upscale), or the inverse (downscale/unshuffle).
 *
 * <p>This is equivalent to PyTorch's {@code nn.PixelShuffle} and {@code nn.PixelUnshuffle}.
 * Internally delegates to the existing {@code depth_to_space} (upscale) or
 * {@code space_to_depth} (downscale) C++ ops.</p>
 *
 * <p>Used by video VLMs for spatial compression of vision tokens:</p>
 * <ul>
 *   <li>SmolVLM2: r=3 downscale (9x spatial compression after vision encoder)</li>
 *   <li>Qwen-VL: r=2 downscale (4x spatial compression)</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * // Upscale: [B, C*r*r, H, W] -> [B, C, H*r, W*r]
 * PixelShuffle upscale = new PixelShuffle(input, 3, false);
 *
 * // Downscale (unshuffle): [B, C, H, W] -> [B, C*r*r, H/r, W/r]
 * PixelShuffle downscale = new PixelShuffle(input, 3, true);
 * }</pre>
 */
@NoArgsConstructor
public class PixelShuffle extends DynamicCustomOp {

    private int blockSize;
    private boolean inverse;

    public PixelShuffle(SameDiff sameDiff, SDVariable input, int blockSize, boolean inverse) {
        super(null, sameDiff, new SDVariable[]{input});
        this.blockSize = blockSize;
        this.inverse = inverse;
        addIArgument(blockSize);
        addIArgument(0);  // NCHW format (isNHWC = false)
    }

    public PixelShuffle(INDArray input, int blockSize, boolean inverse) {
        super(new INDArray[]{input}, null);
        this.blockSize = blockSize;
        this.inverse = inverse;
        addIArgument(blockSize);
        addIArgument(0);  // NCHW format
    }

    public PixelShuffle(INDArray input, INDArray output, int blockSize, boolean inverse) {
        super(null, new INDArray[]{input}, new INDArray[]{output});
        this.blockSize = blockSize;
        this.inverse = inverse;
        addIArgument(blockSize);
        addIArgument(0);  // NCHW format
    }

    @Override
    public String opName() {
        return inverse ? "space_to_depth" : "depth_to_space";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
