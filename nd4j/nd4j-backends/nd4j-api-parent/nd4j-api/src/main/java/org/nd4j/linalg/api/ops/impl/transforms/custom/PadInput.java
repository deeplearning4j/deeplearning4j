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
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Reverses {@link UnpadInput}: scatters unpadded tokens back into padded positions,
 * producing a zero-padded [batch, max_seq_len, hidden_dim] tensor.
 *
 * <p>Inputs:
 * <ol>
 *   <li>unpadded [total_tokens, hidden_dim]</li>
 *   <li>indices [total_tokens] INT64 — from {@link UnpadInput} output 1</li>
 *   <li>batch_size INT64 scalar</li>
 *   <li>max_seq_len INT64 scalar</li>
 * </ol>
 *
 * <p>Output:
 * <ol>
 *   <li>padded [batch, max_seq_len, hidden_dim] — zero-filled with real tokens placed back</li>
 * </ol>
 *
 * <p>The backward pass is implemented by {@link PadInputBp}.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class PadInput extends DynamicCustomOp {

    public PadInput(INDArray unpadded, INDArray indices, long batchSize, long maxSeqLen) {
        super(new INDArray[]{
                unpadded,
                indices,
                Nd4j.scalar(DataType.INT64, batchSize),
                Nd4j.scalar(DataType.INT64, maxSeqLen)
        }, null);
    }

    public PadInput(INDArray unpadded, INDArray indices, INDArray batchSize, INDArray maxSeqLen) {
        super(new INDArray[]{unpadded, indices, batchSize, maxSeqLen}, null);
    }

    public PadInput(SameDiff sameDiff,
                    SDVariable unpadded, SDVariable indices,
                    SDVariable batchSize, SDVariable maxSeqLen) {
        super(null, sameDiff,
              new SDVariable[]{unpadded, indices, batchSize, maxSeqLen}, false);
    }

    @Override
    public String opName() {
        return "pad_input";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "PadInput: expected at least 1 input data type, got %s", inputDataTypes);
        DataType dt = inputDataTypes.get(0);
        Preconditions.checkState(dt.isFPType(),
                "PadInput: unpadded must be a floating-point type, got %s", dt);
        return Collections.singletonList(dt);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
