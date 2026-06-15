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

import java.util.Arrays;
import java.util.List;

/**
 * Decoder masked multi-head attention for autoregressive inference.
 * <p>
 * Performs fused QKV projection, rotary positional encoding, and
 * masked multi-head attention in a single op for decoder-only models.
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: hiddenStates [B, 1, H] - current step hidden states</li>
 *   <li>1: qkvWeight [H, 3H] - fused QKV projection weight</li>
 *   <li>2: outWeight [H, H] - output projection weight</li>
 *   <li>3: pastKey [B, heads, seq, dim] - cached keys from previous steps</li>
 *   <li>4: pastValue [B, heads, seq, dim] - cached values from previous steps</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: numHeads (default 32)</li>
 *   <li>1: numKvHeads - number of key/value heads (for GQA/MQA)</li>
 *   <li>2: headDim - dimension per head</li>
 *   <li>3: useRoPE (0=no, 1=yes, default 1)</li>
 *   <li>4: ropeBase (default 10000)</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: maskFilterValue (default -3.4028235e+38f)</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: output [B, 1, H]</li>
 *   <li>1: presentKey [B, heads, seq+1, dim]</li>
 *   <li>2: presentValue [B, heads, seq+1, dim]</li>
 * </ul>
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class DecoderMaskedMha extends DynamicCustomOp {

    @Getter private int numHeads = 32;
    @Getter private int numKvHeads = 0;
    @Getter private int headDim = 0;
    @Getter private int useRoPE = 1;
    @Getter private int ropeBase = 10000;
    @Getter private float maskFilterValue = -3.4028235e+38f;

    /**
     * SameDiff constructor with required inputs and default options.
     */
    public DecoderMaskedMha(SameDiff sameDiff, SDVariable hiddenStates, SDVariable qkvWeight,
                            SDVariable outWeight, SDVariable pastKey, SDVariable pastValue) {
        super(null, sameDiff, new SDVariable[]{hiddenStates, qkvWeight, outWeight, pastKey, pastValue}, false);
        addIArgument((long) numHeads, (long) numKvHeads, (long) headDim, (long) useRoPE, (long) ropeBase);
        addTArgument((double) maskFilterValue);
    }

    /**
     * SameDiff constructor with boolean useRoPE (uses default ropeBase and maskFilterValue).
     */
    public DecoderMaskedMha(SameDiff sameDiff, SDVariable hiddenStates, SDVariable qkvWeight,
                            SDVariable outWeight, SDVariable pastKey, SDVariable pastValue,
                            int numHeads, int numKvHeads, int headDim, boolean useRoPE) {
        this(sameDiff, hiddenStates, qkvWeight, outWeight, pastKey, pastValue,
                numHeads, numKvHeads, headDim, useRoPE ? 1 : 0, 10000, -3.4028235e+38f);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public DecoderMaskedMha(SameDiff sameDiff, SDVariable hiddenStates, SDVariable qkvWeight,
                            SDVariable outWeight, SDVariable pastKey, SDVariable pastValue,
                            int numHeads, int numKvHeads, int headDim,
                            int useRoPE, int ropeBase, float maskFilterValue) {
        super(null, sameDiff, new SDVariable[]{hiddenStates, qkvWeight, outWeight, pastKey, pastValue}, false);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.useRoPE = useRoPE;
        this.ropeBase = ropeBase;
        this.maskFilterValue = maskFilterValue;
        addIArgument((long) numHeads, (long) numKvHeads, (long) headDim, (long) useRoPE, (long) ropeBase);
        addTArgument((double) maskFilterValue);
    }

    /**
     * SameDiff convenience constructor with query/key/value inputs.
     */
    public DecoderMaskedMha(SameDiff sameDiff, SDVariable query, SDVariable key, SDVariable value,
                            int numHeads, boolean isCausal) {
        super(null, sameDiff, new SDVariable[]{query, key, value}, false);
        this.numHeads = numHeads;
        addIArgument((long) numHeads, (long) numKvHeads, (long) headDim, isCausal ? 1L : 0L, (long) ropeBase);
        addTArgument((double) maskFilterValue);
    }

    /**
     * INDArray convenience constructor with query/key/value inputs.
     */
    public DecoderMaskedMha(INDArray query, INDArray key, INDArray value,
                            int numHeads, boolean isCausal) {
        super(null, new INDArray[]{query, key, value}, null);
        this.numHeads = numHeads;
        addIArgument((long) numHeads, (long) numKvHeads, (long) headDim, isCausal ? 1L : 0L, (long) ropeBase);
        addTArgument((double) maskFilterValue);
    }

    /**
     * INDArray constructor.
     */
    public DecoderMaskedMha(INDArray hiddenStates, INDArray qkvWeight, INDArray outWeight,
                            INDArray pastKey, INDArray pastValue,
                            INDArray output, INDArray presentKey, INDArray presentValue,
                            int numHeads, int numKvHeads, int headDim,
                            int useRoPE, int ropeBase, float maskFilterValue) {
        super(null, new INDArray[]{hiddenStates, qkvWeight, outWeight, pastKey, pastValue},
                output != null ? new INDArray[]{output, presentKey, presentValue} : null);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.useRoPE = useRoPE;
        this.ropeBase = ropeBase;
        this.maskFilterValue = maskFilterValue;
        addIArgument((long) numHeads, (long) numKvHeads, (long) headDim, (long) useRoPE, (long) ropeBase);
        addTArgument((double) maskFilterValue);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.numHeads = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.numKvHeads = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.headDim = iArguments.get(2).intValue();
        if (iArguments.size() > 3) this.useRoPE = iArguments.get(3).intValue();
        if (iArguments.size() > 4) this.ropeBase = iArguments.get(4).intValue();
        if (tArguments.size() > 0) this.maskFilterValue = tArguments.get(0).floatValue();
    }

    @Override
    public String opName() {
        return "decoder_masked_mha";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }
}
