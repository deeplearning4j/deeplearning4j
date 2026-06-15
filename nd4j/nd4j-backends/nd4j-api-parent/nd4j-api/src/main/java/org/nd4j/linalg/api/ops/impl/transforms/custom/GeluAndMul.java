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
 * Fused GELU activation with element-wise multiply (GEGLU).
 *
 * Computes: output = gelu(gate) * up
 *
 * Used in GPT-NeoX, Phi, and other GEGLU architectures.
 *
 * Inputs:
 *   0: gate [batch, ..., dim] — gate projection output
 *   1: up   [batch, ..., dim] — up projection output (same shape)
 *
 * Output:
 *   0: result [batch, ..., dim]
 *
 * Integer args:
 *   0: useTanhApprox (0=exact GELU, 1=tanh approximation, default: 0)
 */
@NoArgsConstructor
public class GeluAndMul extends DynamicCustomOp {

    @Getter private int useTanhApprox = 0;

    public GeluAndMul(INDArray gate, INDArray up) {
        this(gate, up, false);
    }

    public GeluAndMul(INDArray gate, INDArray up, boolean useTanhApprox) {
        super(new INDArray[]{gate, up}, null);
        this.useTanhApprox = useTanhApprox ? 1 : 0;
        addIArgument((long) this.useTanhApprox);
    }

    public GeluAndMul(SameDiff sd, SDVariable gate, SDVariable up, boolean useTanhApprox) {
        super(null, sd, new SDVariable[]{gate, up}, false);
        this.useTanhApprox = useTanhApprox ? 1 : 0;
        addIArgument((long) this.useTanhApprox);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.useTanhApprox = iArguments.get(0).intValue();
    }

    @Override
    public String opName() {
        return "gelu_and_mul";
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
