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

import java.util.Arrays;
import java.util.List;

/**
 * Backprop op for {@link MultiLoraMatmul}.
 *
 * <p>Computes gradients for the trainable inputs of multi_lora_matmul:
 * <ul>
 *   <li>dInput  [B, I]     — gradient w.r.t. input activations</li>
 *   <li>dLoraA  [A, I, R]  — gradient w.r.t. per-adapter down-projections</li>
 *   <li>dLoraB  [A, R, O]  — gradient w.r.t. per-adapter up-projections</li>
 * </ul>
 * baseWeight and adapterIds are frozen/discrete; their gradients are not produced here.
 *
 * <b>Inputs (positional):</b>
 * <ol>
 *   <li>input       [B, I]</li>
 *   <li>baseWeight  [I, O]  (frozen — used for computing dInput but not updated)</li>
 *   <li>loraA       [A, I, R]</li>
 *   <li>loraB       [A, R, O]</li>
 *   <li>adapterIds  [B] INT64</li>
 *   <li>gradOut     [B, O]  upstream gradient</li>
 * </ol>
 *
 * <b>Float args:</b>
 * <ol>
 *   <li>alpha — LoRA scaling</li>
 * </ol>
 *
 * <b>Outputs:</b> [dInput, dLoraA, dLoraB]
 */
@NoArgsConstructor
public class MultiLoraMatmulBp extends DynamicCustomOp {

    private float alpha = 1.0f;

    /**
     * SameDiff constructor.
     */
    public MultiLoraMatmulBp(SameDiff sameDiff,
                             SDVariable input, SDVariable baseWeight,
                             SDVariable loraA, SDVariable loraB,
                             SDVariable adapterIds, SDVariable gradOut,
                             float alpha) {
        super(null, sameDiff,
            new SDVariable[]{input, baseWeight, loraA, loraB, adapterIds, gradOut}, false);
        this.alpha = alpha;
        addTArgument((double) alpha);
    }

    /**
     * INDArray constructor.
     *
     * @return INDArray[3] = {dInput, dLoraA, dLoraB}
     */
    public MultiLoraMatmulBp(INDArray input, INDArray baseWeight, INDArray loraA,
                             INDArray loraB, INDArray adapterIds, INDArray gradOut,
                             float alpha) {
        super(null,
            new INDArray[]{input, baseWeight, loraA, loraB, adapterIds, gradOut}, null);
        this.alpha = alpha;
        addTArgument((double) alpha);
    }

    /**
     * Eager convenience method.
     */
    public static INDArray[] exec(INDArray input, INDArray baseWeight, INDArray loraA,
                                  INDArray loraB, INDArray adapterIds, INDArray gradOut,
                                  float alpha) {
        MultiLoraMatmulBp op = new MultiLoraMatmulBp(
            input, baseWeight, loraA, loraB, adapterIds, gradOut, alpha);
        return Nd4j.exec(op);
    }

    @Override
    public String opName() {
        return "multi_lora_matmul_bp";
    }

    @Override
    public int getNumOutputs() {
        return 3;  // dInput, dLoraA, dLoraB
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 6,
            "Expected 6 input data types, got %s", inputDataTypes);
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt, dt);
    }
}
