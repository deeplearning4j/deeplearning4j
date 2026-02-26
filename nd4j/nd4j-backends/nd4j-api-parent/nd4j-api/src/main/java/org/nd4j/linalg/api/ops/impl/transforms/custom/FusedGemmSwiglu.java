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
 * Fused GEMM + SwiGLU activation.
 * <p>
 * Computes the gated linear unit with SiLU (Swish) activation in a single
 * fused kernel, commonly used in LLM feed-forward blocks:
 * <pre>
 *   output = (input @ wGate) * silu(input @ wUp)
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [M, K]</li>
 *   <li>1: wGate [K, N] - gate projection weight</li>
 *   <li>2: wUp [K, N] - up projection weight</li>
 * </ul>
 * <p>
 * Output: [M, N]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class FusedGemmSwiglu extends DynamicCustomOp {

    /**
     * SameDiff constructor.
     */
    public FusedGemmSwiglu(SameDiff sameDiff, SDVariable input, SDVariable wGate, SDVariable wUp) {
        super(null, sameDiff, new SDVariable[]{input, wGate, wUp}, false);
    }

    /**
     * INDArray constructor.
     */
    public FusedGemmSwiglu(INDArray input, INDArray wGate, INDArray wUp, INDArray output) {
        super(null, new INDArray[]{input, wGate, wUp},
                output != null ? new INDArray[]{output} : null);
    }

    @Override
    public String opName() {
        return "fused_gemm_swiglu";
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
