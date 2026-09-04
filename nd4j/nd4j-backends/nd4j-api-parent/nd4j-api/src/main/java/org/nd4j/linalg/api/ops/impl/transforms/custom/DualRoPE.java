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
 * Dual Rotary Position Embedding (Dual RoPE) for Gemma 4.
 *
 * Applies two independent sets of rotary embeddings with different frequency
 * bases and scales. Gemma 4 uses dual RoPE to support both local (sliding-window)
 * and global attention patterns within the same model:
 *
 * <ul>
 *   <li><b>Local RoPE</b>: Higher frequency base for fine-grained local position encoding
 *       in sliding-window attention layers.</li>
 *   <li><b>Global RoPE</b>: Lower frequency base for coarse global position encoding
 *       in full-context attention layers.</li>
 * </ul>
 *
 * The attention type parameter selects which RoPE configuration is applied.
 *
 * Input shape: [batch, seq_len, num_heads, head_dim]
 * Output shape: [batch, seq_len, num_heads, head_dim]
 *
 * iArgs:
 *   0: attentionType   - 0 = local (sliding-window), 1 = global (full-context)
 *   1: positionOffset  - starting position for KV cache continuation
 *
 * tArgs:
 *   0: localFreqBase   - frequency base for local RoPE (e.g. 10000.0)
 *   1: globalFreqBase  - frequency base for global RoPE (e.g. 1000000.0)
 *   2: localFreqScale  - frequency scale for local RoPE (default 1.0)
 *   3: globalFreqScale - frequency scale for global RoPE (default 1.0)
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class DualRoPE extends DynamicCustomOp {

    public static final int ATTENTION_TYPE_LOCAL = 0;
    public static final int ATTENTION_TYPE_GLOBAL = 1;

    @Getter private int attentionType = ATTENTION_TYPE_LOCAL;
    @Getter private int positionOffset = 0;
    @Getter private double localFreqBase = 10000.0;
    @Getter private double globalFreqBase = 1000000.0;
    @Getter private double localFreqScale = 1.0;
    @Getter private double globalFreqScale = 1.0;

    public DualRoPE(INDArray input, int attentionType, int positionOffset,
                    double localFreqBase, double globalFreqBase,
                    double localFreqScale, double globalFreqScale) {
        super(new INDArray[]{input}, null);
        this.attentionType = attentionType;
        this.positionOffset = positionOffset;
        this.localFreqBase = localFreqBase;
        this.globalFreqBase = globalFreqBase;
        this.localFreqScale = localFreqScale;
        this.globalFreqScale = globalFreqScale;
        addIArgument(attentionType, positionOffset);
        addTArgument(localFreqBase, globalFreqBase, localFreqScale, globalFreqScale);
    }

    public DualRoPE(SameDiff sd, SDVariable input, int attentionType, int positionOffset,
                    double localFreqBase, double globalFreqBase,
                    double localFreqScale, double globalFreqScale) {
        super(null, sd, new SDVariable[]{input}, false);
        this.attentionType = attentionType;
        this.positionOffset = positionOffset;
        this.localFreqBase = localFreqBase;
        this.globalFreqBase = globalFreqBase;
        this.localFreqScale = localFreqScale;
        this.globalFreqScale = globalFreqScale;
        addIArgument(attentionType, positionOffset);
        addTArgument(localFreqBase, globalFreqBase, localFreqScale, globalFreqScale);
    }

    /**
     * Dynamic-position constructor: the base position is supplied as an SDVariable
     * input (typically a scalar INT64 placeholder) instead of a static iArg. This
     * lets decode loops advance positions by feeding the placeholder without
     * rebuilding the graph. iArguments contains only attentionType in this form.
     */
    public DualRoPE(SameDiff sd, SDVariable input, SDVariable position, int attentionType,
                    double localFreqBase, double globalFreqBase,
                    double localFreqScale, double globalFreqScale) {
        super(null, sd, new SDVariable[]{input, position}, false);
        this.attentionType = attentionType;
        this.localFreqBase = localFreqBase;
        this.globalFreqBase = globalFreqBase;
        this.localFreqScale = localFreqScale;
        this.globalFreqScale = globalFreqScale;
        addIArgument(attentionType);
        addTArgument(localFreqBase, globalFreqBase, localFreqScale, globalFreqScale);
    }

    /**
     * Convenience constructor for local (sliding-window) attention layers.
     */
    public DualRoPE(SameDiff sd, SDVariable input, double localFreqBase, double globalFreqBase) {
        this(sd, input, ATTENTION_TYPE_LOCAL, 0, localFreqBase, globalFreqBase, 1.0, 1.0);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.attentionType = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.positionOffset = iArguments.get(1).intValue();
        if (tArguments.size() > 0) this.localFreqBase = tArguments.get(0);
        if (tArguments.size() > 1) this.globalFreqBase = tArguments.get(1);
        if (tArguments.size() > 2) this.localFreqScale = tArguments.get(2);
        if (tArguments.size() > 3) this.globalFreqScale = tArguments.get(3);
    }

    @Override
    public String opName() {
        return "dual_rope";
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
