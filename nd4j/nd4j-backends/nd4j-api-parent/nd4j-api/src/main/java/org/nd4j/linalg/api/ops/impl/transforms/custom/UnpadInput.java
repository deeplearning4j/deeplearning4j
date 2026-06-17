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
 * Removes padding tokens from a batch of sequences using an attention mask,
 * producing a compact (padding-free) representation together with the metadata
 * needed to reverse the operation via {@link PadInput}.
 *
 * <p>Inputs:
 * <ol>
 *   <li>hidden_states [batch, max_seq_len, hidden_dim] — padded input (float)</li>
 *   <li>attention_mask [batch, max_seq_len] — 1=real token, 0=padding (INT32 or BOOL)</li>
 * </ol>
 *
 * <p>Outputs:
 * <ol>
 *   <li>unpadded [total_tokens, hidden_dim] — concatenated real tokens</li>
 *   <li>indices [total_tokens] INT64 — flat index into padded batch for each real token</li>
 *   <li>cu_seqlens [batch+1] INT64 — cumulative sequence lengths (starts with 0)</li>
 *   <li>max_seqlen_in_batch [1] INT64 scalar</li>
 * </ol>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class UnpadInput extends DynamicCustomOp {

    public UnpadInput(INDArray hiddenStates, INDArray attentionMask) {
        super(new INDArray[]{hiddenStates, attentionMask}, null);
    }

    public UnpadInput(INDArray hiddenStates, INDArray attentionMask, INDArray[] outputs) {
        super(new INDArray[]{hiddenStates, attentionMask},
              outputs != null ? outputs : null);
    }

    public UnpadInput(SameDiff sameDiff, SDVariable hiddenStates, SDVariable attentionMask) {
        super(null, sameDiff, new SDVariable[]{hiddenStates, attentionMask}, false);
    }

    @Override
    public String opName() {
        return "unpad_input";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 2,
                "UnpadInput: expected at least 2 input data types, got %s", inputDataTypes);
        DataType hiddenDtype = inputDataTypes.get(0);
        Preconditions.checkState(hiddenDtype.isFPType(),
                "UnpadInput: hidden_states must be a floating-point type, got %s", hiddenDtype);
        // unpadded has same float dtype; indices, cu_seqlens, max_seqlen are INT64
        return Arrays.asList(hiddenDtype, DataType.INT64, DataType.INT64, DataType.INT64);
    }

    @Override
    public int getNumOutputs() {
        return 4;
    }
}
