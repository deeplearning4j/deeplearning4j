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
import java.util.Collections;
import java.util.List;

/**
 * SwiGLU Component Operation: swish(x) * y
 *
 * Computes silu(x) * y where silu(x) = x * sigmoid(x).
 * This is the gated activation function used in LLaMA, Mistral, and other modern LLMs
 * for their feed-forward network.
 *
 * The SwiGLU activation pattern is: silu(x @ W_gate) * (x @ W_up)
 * This op implements the element-wise part: silu(gate_output) * up_output
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class SwishMul extends DynamicCustomOp {

    /**
     * Create a swish_mul operation for SameDiff.
     *
     * @param sameDiff The SameDiff instance
     * @param x Input tensor for swish activation
     * @param y Gate tensor to multiply with
     */
    public SwishMul(SameDiff sameDiff, SDVariable x, SDVariable y) {
        super(null, sameDiff, new SDVariable[]{x, y}, false);
    }

    /**
     * Create a swish_mul operation for INDArray execution.
     *
     * @param x Input array for swish activation
     * @param y Gate array to multiply with
     * @param output Output array (optional)
     */
    public SwishMul(INDArray x, INDArray y, INDArray output) {
        super(new INDArray[]{x, y}, output != null ? new INDArray[]{output} : null);
    }

    /**
     * Create a swish_mul operation for INDArray execution.
     *
     * @param x Input array for swish activation
     * @param y Gate array to multiply with
     */
    public SwishMul(INDArray x, INDArray y) {
        this(x, y, null);
    }

    @Override
    public String opName() {
        return "swish_mul";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        return Arrays.asList(new SwishMulBp(sameDiff, arg(0), arg(1), gradients.get(0)).outputVariables());
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
