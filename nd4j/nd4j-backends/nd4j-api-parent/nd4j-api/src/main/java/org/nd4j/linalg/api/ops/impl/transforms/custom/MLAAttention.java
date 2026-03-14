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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Multi-head Latent Attention (MLA) operation.
 * <p>
 * Implements MLA as used in DeepSeek-V2 and similar models. Instead of storing
 * separate K and V caches per layer, a single shared latent (compressed) KV cache
 * is stored and decompressed per-layer via learned down-projection weights.
 * This dramatically reduces KV cache memory during inference.
 * <p>
 * Algorithm:
 * <ol>
 *   <li>Decompress: latentKVCache @ kvDownProj -> K, V</li>
 *   <li>Attention: softmax(Q @ K^T * scale) @ V</li>
 *   <li>GQA head mapping if numHeads != numKvHeads</li>
 * </ol>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: query [B, 1, numHeads, headDim]</li>
 *   <li>1: latentKVCache [B, maxSeqLen, latentDim] - shared across layers</li>
 *   <li>2: kvDownProj [latentDim, 2 * numKvHeads * headDim] - per-layer decompression</li>
 *   <li>3: ropeCache [B, maxSeqLen, ropeHeadDim] (optional)</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: numHeads (number of query heads)</li>
 *   <li>1: numKvHeads (number of KV heads for GQA)</li>
 *   <li>2: headDim (dimension per head)</li>
 *   <li>3: latentDim (compressed latent dimension)</li>
 *   <li>4: ropeHeadDim (RoPE portion dimension, 0 if none)</li>
 *   <li>5: seqLen (current valid sequence length in cache)</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: attScale (attention scale, 0 = auto 1/sqrt(headDim))</li>
 * </ul>
 * <p>
 * Output:
 * <ul>
 *   <li>0: output [B, 1, numHeads, headDim]</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class MLAAttention extends DynamicCustomOp {

    @Getter private int numHeads = 8;
    @Getter private int numKvHeads = 8;
    @Getter private int headDim = 64;
    @Getter private int latentDim = 512;
    @Getter private int ropeHeadDim = 0;
    @Getter private int seqLen = 1;
    @Getter private float attScale = 0.0f;

    /**
     * SameDiff constructor with required inputs and all configuration.
     *
     * @param sameDiff      SameDiff instance
     * @param query         [B, 1, numHeads, headDim]
     * @param latentKVCache [B, maxSeqLen, latentDim]
     * @param kvDownProj    [latentDim, 2 * numKvHeads * headDim]
     * @param numHeads      number of query heads
     * @param numKvHeads    number of KV heads (for GQA)
     * @param headDim       dimension per head
     * @param latentDim     compressed latent dimension
     * @param ropeHeadDim   RoPE portion dimension (0 if none)
     * @param seqLen        current valid sequence length
     * @param attScale      attention scale (0 = auto)
     */
    public MLAAttention(SameDiff sameDiff, SDVariable query, SDVariable latentKVCache,
                        SDVariable kvDownProj, int numHeads, int numKvHeads, int headDim,
                        int latentDim, int ropeHeadDim, int seqLen, float attScale) {
        super(null, sameDiff, new SDVariable[]{query, latentKVCache, kvDownProj}, false);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.latentDim = latentDim;
        this.ropeHeadDim = ropeHeadDim;
        this.seqLen = seqLen;
        this.attScale = attScale;
        addArgs();
    }

    /**
     * SameDiff constructor with optional RoPE cache.
     */
    public MLAAttention(SameDiff sameDiff, SDVariable query, SDVariable latentKVCache,
                        SDVariable kvDownProj, SDVariable ropeCache,
                        int numHeads, int numKvHeads, int headDim,
                        int latentDim, int ropeHeadDim, int seqLen, float attScale) {
        super(null, sameDiff, new SDVariable[]{query, latentKVCache, kvDownProj, ropeCache}, false);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.latentDim = latentDim;
        this.ropeHeadDim = ropeHeadDim;
        this.seqLen = seqLen;
        this.attScale = attScale;
        addArgs();
    }

    /**
     * INDArray constructor for direct execution.
     *
     * @param query         [B, 1, numHeads, headDim]
     * @param latentKVCache [B, maxSeqLen, latentDim]
     * @param kvDownProj    [latentDim, 2 * numKvHeads * headDim]
     * @param output        [B, 1, numHeads, headDim] (may be null for auto-allocation)
     * @param numHeads      number of query heads
     * @param numKvHeads    number of KV heads
     * @param headDim       dimension per head
     * @param latentDim     compressed latent dimension
     * @param ropeHeadDim   RoPE portion dimension (0 if none)
     * @param seqLen        current valid sequence length
     * @param attScale      attention scale (0 = auto)
     */
    public MLAAttention(INDArray query, INDArray latentKVCache, INDArray kvDownProj,
                        INDArray output, int numHeads, int numKvHeads, int headDim,
                        int latentDim, int ropeHeadDim, int seqLen, float attScale) {
        super(null, new INDArray[]{query, latentKVCache, kvDownProj},
              output != null ? new INDArray[]{output} : null);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.latentDim = latentDim;
        this.ropeHeadDim = ropeHeadDim;
        this.seqLen = seqLen;
        this.attScale = attScale;
        addArgs();
    }

    /**
     * INDArray constructor with optional RoPE cache.
     */
    public MLAAttention(INDArray query, INDArray latentKVCache, INDArray kvDownProj,
                        INDArray ropeCache, INDArray output,
                        int numHeads, int numKvHeads, int headDim,
                        int latentDim, int ropeHeadDim, int seqLen, float attScale) {
        super(null,
              ropeCache != null ? new INDArray[]{query, latentKVCache, kvDownProj, ropeCache}
                                : new INDArray[]{query, latentKVCache, kvDownProj},
              output != null ? new INDArray[]{output} : null);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.latentDim = latentDim;
        this.ropeHeadDim = ropeHeadDim;
        this.seqLen = seqLen;
        this.attScale = attScale;
        addArgs();
    }

    private void addArgs() {
        addIArgument((long) numHeads, (long) numKvHeads, (long) headDim,
                     (long) latentDim, (long) ropeHeadDim, (long) seqLen);
        addTArgument((double) attScale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.numHeads = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.numKvHeads = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.headDim = iArguments.get(2).intValue();
        if (iArguments.size() > 3) this.latentDim = iArguments.get(3).intValue();
        if (iArguments.size() > 4) this.ropeHeadDim = iArguments.get(4).intValue();
        if (iArguments.size() > 5) this.seqLen = iArguments.get(5).intValue();
        if (tArguments.size() > 0) this.attScale = tArguments.get(0).floatValue();
    }

    @Override
    public String opName() {
        return "mla_attention";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
