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

import java.util.Collections;
import java.util.List;

/**
 * Fused SiLU (Swish) activation with element-wise multiply (SwiGLU).
 *
 * Computes: output = silu(gate) * up = (gate * sigmoid(gate)) * up
 *
 * This is the core MLP computation in LLaMA, Qwen, Gemma, and Mistral.
 *
 * Inputs:
 *   0: gate [batch, ..., dim] — gate projection output
 *   1: up   [batch, ..., dim] — up projection output (same shape)
 *
 * Output:
 *   0: result [batch, ..., dim]
 */
@NoArgsConstructor
public class SiluAndMul extends DynamicCustomOp {

    public SiluAndMul(INDArray gate, INDArray up) {
        super(new INDArray[]{gate, up}, null);
    }

    public SiluAndMul(SameDiff sd, SDVariable gate, SDVariable up) {
        super(null, sd, new SDVariable[]{gate, up}, false);
    }

    @Override
    public String opName() {
        return "silu_and_mul";
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
