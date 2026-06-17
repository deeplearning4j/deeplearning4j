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

import java.util.Collections;
import java.util.List;

/**
 * Backward pass for {@link PadInput}: gathers gradients from padded positions
 * back to unpadded positions.
 *
 * <p>This is simply a gather: grad_unpadded[i] = grad_padded[indices[i]].
 *
 * <p>Inputs:
 * <ol>
 *   <li>grad_padded [batch, max_seq_len, hidden_dim]</li>
 *   <li>indices [total_tokens] INT64</li>
 * </ol>
 *
 * <p>Output:
 * <ol>
 *   <li>grad_unpadded [total_tokens, hidden_dim]</li>
 * </ol>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class PadInputBp extends DynamicCustomOp {

    public PadInputBp(INDArray gradPadded, INDArray indices) {
        super(new INDArray[]{gradPadded, indices}, null);
    }

    public PadInputBp(SameDiff sameDiff, SDVariable gradPadded, SDVariable indices) {
        super(null, sameDiff, new SDVariable[]{gradPadded, indices}, false);
    }

    @Override
    public String opName() {
        return "pad_input_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "PadInputBp: expected at least 1 input data type, got %s", inputDataTypes);
        DataType dt = inputDataTypes.get(0);
        Preconditions.checkState(dt.isFPType(),
                "PadInputBp: grad_padded must be a floating-point type, got %s", dt);
        return Collections.singletonList(dt);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
