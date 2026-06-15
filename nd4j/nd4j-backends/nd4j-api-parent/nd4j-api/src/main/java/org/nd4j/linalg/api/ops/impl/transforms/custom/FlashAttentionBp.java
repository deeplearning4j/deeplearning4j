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
 * Flash Attention Backward Pass.
 *
 * Computes gradients for query, key, and value tensors.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FlashAttentionBp extends DynamicCustomOp {
    private boolean isCausal = true;
    private double scale = 0.0;

    /**
     * Create a flash attention backward pass operation.
     *
     * @param query Query tensor [batch, seq_len, num_heads, head_dim]
     * @param key Key tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param value Value tensor [batch, kv_len, num_kv_heads, head_dim]
     * @param gradOutput Gradient of loss w.r.t. output [batch, seq_len, num_heads, head_dim]
     * @param isCausal Whether causal mask was applied in forward pass
     * @param scale Scale factor used in forward pass
     */
    public FlashAttentionBp(INDArray query, INDArray key, INDArray value, INDArray gradOutput,
                            boolean isCausal, double scale) {
        super(wrapFilterNull(query, key, value, gradOutput), null);
        this.isCausal = isCausal;
        this.scale = scale;
        addIArgument(isCausal ? 1 : 0);
        addTArgument(scale);
    }

    /**
     * Create a flash attention backward pass operation for SameDiff.
     */
    public FlashAttentionBp(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                            SDVariable gradOutput, boolean isCausal, double scale) {
        super(null, sd, new SDVariable[]{query, key, value, gradOutput}, false);
        this.isCausal = isCausal;
        this.scale = scale;
        addIArgument(isCausal ? 1 : 0);
        addTArgument(scale);
    }

    /**
     * Create a flash attention backward pass with forward output and LSE for efficiency.
     *
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param gradOutput Gradient from upstream
     * @param forwardOutput Output from forward pass (avoids recomputation)
     * @param softmaxLse Log-sum-exp from forward pass (avoids recomputation)
     * @param isCausal Whether causal mask was applied
     * @param scale Scale factor used
     */
    public FlashAttentionBp(SameDiff sd, SDVariable query, SDVariable key, SDVariable value,
                            SDVariable gradOutput, SDVariable forwardOutput, SDVariable softmaxLse,
                            boolean isCausal, double scale) {
        super(null, sd, new SDVariable[]{query, key, value, gradOutput, forwardOutput, softmaxLse}, false);
        this.isCausal = isCausal;
        this.scale = scale;
        addIArgument(isCausal ? 1 : 0);
        addTArgument(scale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) {
            this.isCausal = iArguments.get(0) != 0;
        }
        if (!tArguments.isEmpty()) {
            this.scale = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "flash_attention_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        DataType first = dataTypes.get(0);
        Preconditions.checkState(first.isFPType(),
            "Input datatype must be a floating point type, got %s", first);
        // Returns gradients for query, key, value
        return Arrays.asList(first, first, first);
    }

    @Override
    public int getNumOutputs() {
        return 3;  // gradQuery, gradKey, gradValue
    }
}
