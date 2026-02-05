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
 * Fused Rotary Position Embedding (RoPE)
 *
 * Applies rotary position embeddings with optimized sin/cos computation.
 * Supports multiple RoPE variants used in modern LLMs:
 * - Standard (LLaMA, Mistral) - ropeType=0
 * - NeoX (GPT-NeoX, Pythia) - ropeType=1
 * - GPT-J - ropeType=2
 *
 * Input shape: [batch, seq_len, num_heads, head_dim]
 * Output shape: [batch, seq_len, num_heads, head_dim]
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedRoPE extends DynamicCustomOp {

    public static final int ROPE_TYPE_STANDARD = 0;  // LLaMA, Mistral
    public static final int ROPE_TYPE_NEOX = 1;       // GPT-NeoX, Pythia
    public static final int ROPE_TYPE_GPTJ = 2;       // GPT-J

    @Getter private int ropeType = ROPE_TYPE_STANDARD;
    @Getter private int positionOffset = 0;
    @Getter private double freqBase = 10000.0;
    @Getter private double freqScale = 1.0;

    /**
     * Create a fused RoPE operation.
     *
     * @param sameDiff The SameDiff instance
     * @param input Input tensor [batch, seq_len, num_heads, head_dim]
     * @param ropeType RoPE variant (0=standard, 1=neox, 2=gptj)
     * @param positionOffset Starting position for KV cache continuation
     * @param freqBase Base frequency (default 10000.0)
     * @param freqScale Frequency scale factor (default 1.0)
     */
    public FusedRoPE(SameDiff sameDiff, SDVariable input, int ropeType, int positionOffset,
                     double freqBase, double freqScale) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.ropeType = ropeType;
        this.positionOffset = positionOffset;
        this.freqBase = freqBase;
        this.freqScale = freqScale;

        addIArgument(ropeType, positionOffset);
        addTArgument(freqBase, freqScale);
    }

    public FusedRoPE(SameDiff sameDiff, SDVariable input) {
        this(sameDiff, input, ROPE_TYPE_STANDARD, 0, 10000.0, 1.0);
    }

    public FusedRoPE(INDArray input, INDArray output, int ropeType, int positionOffset,
                     double freqBase, double freqScale) {
        super(new INDArray[]{input}, output != null ? new INDArray[]{output} : null);
        this.ropeType = ropeType;
        this.positionOffset = positionOffset;
        this.freqBase = freqBase;
        this.freqScale = freqScale;

        addIArgument(ropeType, positionOffset);
        addTArgument(freqBase, freqScale);
    }

    public FusedRoPE(INDArray input) {
        this(input, null, ROPE_TYPE_STANDARD, 0, 10000.0, 1.0);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.ropeType = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.positionOffset = iArguments.get(1).intValue();
        }
        if (tArguments.size() > 0) {
            this.freqBase = tArguments.get(0);
        }
        if (tArguments.size() > 1) {
            this.freqScale = tArguments.get(1);
        }
    }

    @Override
    public String opName() {
        return "fused_rope";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Collections.singletonList(
            new FusedRoPEBp(sameDiff, arg(0), gradients.get(0), ropeType, positionOffset, freqBase, freqScale).outputVariable()
        );
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
