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
 * TurboQuant asymmetric attention operation.
 *
 * <p>Computes scaled dot-product attention using compressed key representations
 * from TurboQuant's two-stage quantization. The asymmetric inner product estimator
 * combines MSE reconstruction with QJL correction:</p>
 *
 * <pre>{@code
 * score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs>
 * }</pre>
 *
 * <h3>Inputs</h3>
 * <ol start="0">
 *   <li>Q — [B, H, Sq, D] query tensor</li>
 *   <li>K_mse — [B, H, Sk, D] pre-reconstructed MSE keys (FLOAT16)</li>
 *   <li>QJL_signs — [B, H, Sk, D] sign bits (INT8)</li>
 *   <li>Residual_norms — [B, H, Sk] residual norms (FLOAT16)</li>
 *   <li>QJL_matrix — [D, D] projection matrix (FLOAT32)</li>
 *   <li>V — [B, H, Sk, D] dequantized values</li>
 *   <li>Attention_mask — [B, 1, 1, Sk] or compatible shape</li>
 * </ol>
 *
 * <h3>Integer Args</h3>
 * <ul>
 *   <li>0: numHeads</li>
 *   <li>1: headDim</li>
 * </ul>
 *
 * <h3>Float Args</h3>
 * <ul>
 *   <li>0: scale (default 1/√headDim)</li>
 * </ul>
 *
 * <h3>Output</h3>
 * <p>[B, H, Sq, D] attention output</p>
 *
 * @see org.eclipse.deeplearning4j.llm.generation.TurboQuantKvCacheManager
 */
@NoArgsConstructor
public class TurboQuantAttention extends DynamicCustomOp {

    /**
     * INDArray constructor for direct execution.
     *
     * @param query          [B, H, Sq, D] query tensor
     * @param kMse           [B, H, Sk, D] MSE-reconstructed keys
     * @param qjlSigns       [B, H, Sk, D] QJL sign bits
     * @param residualNorms  [B, H, Sk] residual norms
     * @param qjlMatrix      [D, D] QJL projection matrix
     * @param values         [B, H, Sk, D] dequantized values
     * @param attentionMask  [B, 1, 1, Sk] attention mask
     * @param numHeads       number of attention heads
     * @param headDim        dimension per head
     * @param scale          attention scale (0 = auto: 1/sqrt(headDim))
     */
    public TurboQuantAttention(INDArray query, INDArray kMse, INDArray qjlSigns,
                                INDArray residualNorms, INDArray qjlMatrix,
                                INDArray values, INDArray attentionMask,
                                int numHeads, int headDim, double scale) {
        super(null, new INDArray[]{query, kMse, qjlSigns, residualNorms,
                qjlMatrix, values, attentionMask}, null);
        addIArgument(numHeads, headDim);
        addTArgument(scale);
    }

    /**
     * SameDiff constructor for graph execution.
     */
    public TurboQuantAttention(SameDiff sameDiff, SDVariable query, SDVariable kMse,
                                SDVariable qjlSigns, SDVariable residualNorms,
                                SDVariable qjlMatrix, SDVariable values,
                                SDVariable attentionMask,
                                int numHeads, int headDim, double scale) {
        super(null, sameDiff, new SDVariable[]{query, kMse, qjlSigns, residualNorms,
                qjlMatrix, values, attentionMask}, false);
        addIArgument(numHeads, headDim);
        addTArgument(scale);
    }

    @Override
    public String opName() {
        return "turbo_quant_attention";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // Output type matches query type
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
