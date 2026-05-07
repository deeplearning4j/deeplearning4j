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
 * Lightning Attention — O(N) linear attention using cumulative sum recurrence.
 *
 * Instead of the full N×N attention matrix, uses a running state:
 *   S_i = sum_{j<=i} (K_j^T * V_j)
 *   O_i = scale * Q_i @ S_i
 *
 * Complexity: O(N * d²) vs O(N² * d) for standard attention.
 *
 * Inputs:
 *   0: query  [batch, heads, seqLen, headDim]
 *   1: key    [batch, heads, seqLen, headDim]
 *   2: value  [batch, heads, seqLen, headDim]
 *
 * Output:
 *   0: attention output [batch, heads, seqLen, headDim]
 *
 * Float args:
 *   0: scale (default: 1/sqrt(headDim))
 */
@NoArgsConstructor
public class LightningAttention extends DynamicCustomOp {

    @Getter private double scale = 0.0;

    public LightningAttention(INDArray query, INDArray key, INDArray value) {
        super(new INDArray[]{query, key, value}, null);
    }

    public LightningAttention(INDArray query, INDArray key, INDArray value, double scale) {
        super(new INDArray[]{query, key, value}, null);
        this.scale = scale;
        addTArgument(scale);
    }

    public LightningAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
    }

    public LightningAttention(SameDiff sd, SDVariable query, SDVariable key, SDVariable value, double scale) {
        super(null, sd, new SDVariable[]{query, key, value}, false);
        this.scale = scale;
        addTArgument(scale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.scale = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "lightning_attention";
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
