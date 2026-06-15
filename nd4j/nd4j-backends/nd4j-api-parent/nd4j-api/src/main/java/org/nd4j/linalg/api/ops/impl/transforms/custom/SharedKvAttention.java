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
 * Shared Key-Value Attention for Gemma 4.
 *
 * Performs multi-head attention where key and value projections are shared
 * across a group of layers. This reduces parameter count and memory by
 * reusing K/V computations from a designated source layer.
 *
 * Supports both global (full-context) and sliding-window (local) attention
 * patterns, configured via the causal and slidingWindowSize parameters.
 *
 * Inputs:
 *   0: query       [batch, seq_len, num_heads, head_dim]    - query projection
 *   1: sharedKey   [batch, seq_len, num_kv_heads, head_dim] - shared key projection
 *   2: sharedValue [batch, seq_len, num_kv_heads, head_dim] - shared value projection
 *   3: mask        [batch, 1, seq_len, seq_len] (optional)  - attention mask
 *
 * Outputs:
 *   0: output [batch, seq_len, num_heads, head_dim] - attention output
 *
 * iArgs:
 *   0: numHeads          - number of query attention heads
 *   1: numKvHeads        - number of key/value heads
 *   2: causal            - 1 for causal (autoregressive) masking, 0 for bidirectional
 *   3: slidingWindowSize - sliding window size (0 = full context / global attention)
 *
 * tArgs:
 *   0: scale - attention score scaling factor (typically 1/sqrt(head_dim))
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class SharedKvAttention extends DynamicCustomOp {

    @Getter private int numHeads;
    @Getter private int numKvHeads;
    @Getter private int causal = 1;
    @Getter private int slidingWindowSize = 0;
    @Getter private double scale = 0.0;

    public SharedKvAttention(INDArray query, INDArray sharedKey, INDArray sharedValue,
                             int numHeads, int numKvHeads, int causal, int slidingWindowSize, double scale) {
        super(new INDArray[]{query, sharedKey, sharedValue}, null);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.causal = causal;
        this.slidingWindowSize = slidingWindowSize;
        this.scale = scale;
        addIArgument(numHeads, numKvHeads, causal, slidingWindowSize);
        addTArgument(scale);
    }

    public SharedKvAttention(INDArray query, INDArray sharedKey, INDArray sharedValue, INDArray mask,
                             int numHeads, int numKvHeads, int causal, int slidingWindowSize, double scale) {
        super(mask != null
                ? new INDArray[]{query, sharedKey, sharedValue, mask}
                : new INDArray[]{query, sharedKey, sharedValue}, null);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.causal = causal;
        this.slidingWindowSize = slidingWindowSize;
        this.scale = scale;
        addIArgument(numHeads, numKvHeads, causal, slidingWindowSize);
        addTArgument(scale);
    }

    public SharedKvAttention(SameDiff sd, SDVariable query, SDVariable sharedKey, SDVariable sharedValue,
                             int numHeads, int numKvHeads, int causal, int slidingWindowSize, double scale) {
        super(null, sd, new SDVariable[]{query, sharedKey, sharedValue}, false);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.causal = causal;
        this.slidingWindowSize = slidingWindowSize;
        this.scale = scale;
        addIArgument(numHeads, numKvHeads, causal, slidingWindowSize);
        addTArgument(scale);
    }

    public SharedKvAttention(SameDiff sd, SDVariable query, SDVariable sharedKey, SDVariable sharedValue,
                             SDVariable mask, int numHeads, int numKvHeads, int causal,
                             int slidingWindowSize, double scale) {
        super(null, sd, mask != null
                ? new SDVariable[]{query, sharedKey, sharedValue, mask}
                : new SDVariable[]{query, sharedKey, sharedValue}, false);
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.causal = causal;
        this.slidingWindowSize = slidingWindowSize;
        this.scale = scale;
        addIArgument(numHeads, numKvHeads, causal, slidingWindowSize);
        addTArgument(scale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.numHeads = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.numKvHeads = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.causal = iArguments.get(2).intValue();
        if (iArguments.size() > 3) this.slidingWindowSize = iArguments.get(3).intValue();
        if (tArguments.size() > 0) this.scale = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "shared_kv_attention";
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
