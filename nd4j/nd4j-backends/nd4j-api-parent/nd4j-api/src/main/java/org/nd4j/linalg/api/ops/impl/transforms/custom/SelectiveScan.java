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
 * Selective scan (S6/Mamba) operation.
 * <p>
 * Implements the selective state space model scan used in Mamba architectures:
 * <pre>
 *   h_t = A_t * h_{t-1} + B_t * x_t
 *   y_t = C_t * h_t + D * x_t
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: x [B, L, D] - input sequence</li>
 *   <li>1: A [B, L, S] - state transition (discretized)</li>
 *   <li>2: B [B, L, S] - input projection</li>
 *   <li>3: C [B, L, S] - output projection</li>
 *   <li>4: D [D] - skip connection weight</li>
 *   <li>5: h0 (optional) - initial hidden state [B, D, S]</li>
 * </ul>
 * <p>
 * Output: [B, L, D]
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class SelectiveScan extends DynamicCustomOp {

    /**
     * SameDiff constructor with required inputs.
     */
    public SelectiveScan(SameDiff sameDiff, SDVariable x, SDVariable a, SDVariable b,
                         SDVariable c, SDVariable d) {
        super(null, sameDiff, new SDVariable[]{x, a, b, c, d}, false);
    }

    /**
     * SameDiff constructor with initial hidden state.
     */
    public SelectiveScan(SameDiff sameDiff, SDVariable x, SDVariable a, SDVariable b,
                         SDVariable c, SDVariable d, SDVariable h0) {
        super(null, sameDiff, new SDVariable[]{x, a, b, c, d, h0}, false);
    }

    /**
     * INDArray constructor without output pre-allocation.
     */
    public SelectiveScan(INDArray x, INDArray a, INDArray b, INDArray c, INDArray d,
                         INDArray h0) {
        this(x, a, b, c, d, h0, null);
    }

    /**
     * INDArray constructor.
     */
    public SelectiveScan(INDArray x, INDArray a, INDArray b, INDArray c, INDArray d,
                         INDArray h0, INDArray output) {
        super(null, h0 != null ?
                new INDArray[]{x, a, b, c, d, h0} :
                new INDArray[]{x, a, b, c, d},
                output != null ? new INDArray[]{output} : null);
    }

    public SelectiveScan(SameDiff sd, SDVariable input) {
        super(null, sd, new SDVariable[]{input}, false);
    }

    public SelectiveScan(INDArray input) {
        super(new INDArray[]{input}, null);
    }

    @Override
    public String opName() {
        return "selective_scan";
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
