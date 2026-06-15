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
 * Fused Multimodal Rotary Position Embedding (M-RoPE).
 *
 * <p>Applies rotary position embeddings with separate temporal/height/width position
 * encodings for multimodal models like Qwen3-VL and Qwen2.5-VL. The head dimension
 * is split into 3 frequency band sections, each rotated using its own position vector.</p>
 *
 * <p>For Qwen3-VL with head_dim=64, the default sections are [24, 20, 20]:
 * <ul>
 *   <li>Temporal band: dims [0..24) rotated by temporal positions</li>
 *   <li>Height band: dims [24..44) rotated by spatial height positions</li>
 *   <li>Width band: dims [44..64) rotated by spatial width positions</li>
 * </ul>
 *
 * <p>Interleaved mode (Qwen3-VL): instead of contiguous sections, frequencies are
 * distributed round-robin across all 3 position types.</p>
 *
 * <p>Input shapes:</p>
 * <ul>
 *   <li>input: [batch, seq_len, num_heads, head_dim]</li>
 *   <li>position_t: [batch, seq_len] - temporal positions</li>
 *   <li>position_h: [batch, seq_len] - height positions</li>
 *   <li>position_w: [batch, seq_len] - width positions</li>
 * </ul>
 *
 * <p>Output: [batch, seq_len, num_heads, head_dim]</p>
 */
@NoArgsConstructor
public class FusedMRoPE extends DynamicCustomOp {

    @Getter private int sectionT = 24;
    @Getter private int sectionH = 20;
    @Getter private int sectionW = 20;
    @Getter private boolean interleaved = false;
    @Getter private double freqBase = 10000.0;

    /**
     * Create M-RoPE for SameDiff.
     *
     * @param sameDiff SameDiff instance
     * @param input input tensor [batch, seq, heads, head_dim]
     * @param positionT temporal positions [batch, seq]
     * @param positionH height positions [batch, seq]
     * @param positionW width positions [batch, seq]
     * @param sectionT temporal section size
     * @param sectionH height section size
     * @param sectionW width section size
     * @param interleaved whether to interleave frequency bands
     * @param freqBase base frequency (default 10000.0)
     */
    public FusedMRoPE(SameDiff sameDiff, SDVariable input,
                      SDVariable positionT, SDVariable positionH, SDVariable positionW,
                      int sectionT, int sectionH, int sectionW,
                      boolean interleaved, double freqBase) {
        super(null, sameDiff, new SDVariable[]{input, positionT, positionH, positionW});
        this.sectionT = sectionT;
        this.sectionH = sectionH;
        this.sectionW = sectionW;
        this.interleaved = interleaved;
        this.freqBase = freqBase;
        addArgs();
    }

    /**
     * Create M-RoPE for INDArray execution.
     *
     * @param input input tensor [batch, seq, heads, head_dim]
     * @param positionT temporal positions [batch, seq]
     * @param positionH height positions [batch, seq]
     * @param positionW width positions [batch, seq]
     * @param sectionT temporal section size
     * @param sectionH height section size
     * @param sectionW width section size
     * @param interleaved whether to interleave frequency bands
     * @param freqBase base frequency
     */
    public FusedMRoPE(INDArray input, INDArray positionT, INDArray positionH, INDArray positionW,
                      int sectionT, int sectionH, int sectionW,
                      boolean interleaved, double freqBase) {
        super(new INDArray[]{input, positionT, positionH, positionW}, null);
        this.sectionT = sectionT;
        this.sectionH = sectionH;
        this.sectionW = sectionW;
        this.interleaved = interleaved;
        this.freqBase = freqBase;
        addArgs();
    }

    /**
     * Create M-RoPE with Qwen3-VL defaults (sections=[24,20,20], interleaved=true).
     */
    public static FusedMRoPE forQwen3VL(SameDiff sameDiff, SDVariable input,
                                         SDVariable posT, SDVariable posH, SDVariable posW) {
        return new FusedMRoPE(sameDiff, input, posT, posH, posW, 24, 20, 20, true, 5000000.0);
    }

    /**
     * Create M-RoPE with Qwen2.5-VL defaults (sections=[24,20,20], interleaved=false).
     */
    public static FusedMRoPE forQwen25VL(SameDiff sameDiff, SDVariable input,
                                          SDVariable posT, SDVariable posH, SDVariable posW) {
        return new FusedMRoPE(sameDiff, input, posT, posH, posW, 24, 20, 20, false, 10000.0);
    }

    private void addArgs() {
        iArguments.clear();
        tArguments.clear();
        addIArgument(sectionT);
        addIArgument(sectionH);
        addIArgument(sectionW);
        addIArgument(interleaved ? 1 : 0);
        addTArgument(freqBase);
    }

    @Override
    public String opName() {
        return "fused_mrope";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
