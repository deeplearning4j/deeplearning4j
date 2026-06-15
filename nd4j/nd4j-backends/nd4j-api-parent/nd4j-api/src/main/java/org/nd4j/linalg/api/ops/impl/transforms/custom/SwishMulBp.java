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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Backward pass for SwishMul operation.
 *
 * Computes gradients for swish_mul(x, y) = silu(x) * y
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class SwishMulBp extends DynamicCustomOp {

    /**
     * Create a swish_mul_bp operation for SameDiff.
     *
     * @param sameDiff The SameDiff instance
     * @param x Input tensor for swish activation
     * @param y Gate tensor
     * @param gradOut Gradient of the output
     */
    public SwishMulBp(SameDiff sameDiff, SDVariable x, SDVariable y, SDVariable gradOut) {
        super(null, sameDiff, new SDVariable[]{x, y, gradOut}, false);
    }

    /**
     * Create a swish_mul_bp operation for INDArray execution.
     *
     * @param x Input array for swish activation
     * @param y Gate array
     * @param gradOut Gradient of the output
     * @param gradX Gradient w.r.t. x (output)
     * @param gradY Gradient w.r.t. y (output)
     */
    public SwishMulBp(INDArray x, INDArray y, INDArray gradOut, INDArray gradX, INDArray gradY) {
        super(new INDArray[]{x, y, gradOut},
              gradX != null && gradY != null ? new INDArray[]{gradX, gradY} : null);
    }

    @Override
    public String opName() {
        return "swish_mul_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(1));
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
