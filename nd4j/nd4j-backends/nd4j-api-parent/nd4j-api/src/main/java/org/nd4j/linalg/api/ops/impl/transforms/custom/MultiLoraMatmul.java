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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Multi-LoRA matrix multiplication for batched adapter inference.
 * <p>
 * Applies per-sample LoRA adapters during batched inference:
 * <pre>
 *   output[i] = input[i] @ baseWeight + alpha * input[i] @ loraA[adapterIds[i]] @ loraB[adapterIds[i]]
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [B, I] - input activations</li>
 *   <li>1: baseWeight [I, O] - base model weight</li>
 *   <li>2: loraA [A, I, R] - LoRA down-projection weights (A adapters)</li>
 *   <li>3: loraB [A, R, O] - LoRA up-projection weights (A adapters)</li>
 *   <li>4: adapterIds [B] INT64 - per-sample adapter index</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: alpha (default 1.0) - LoRA scaling factor</li>
 * </ul>
 * <p>
 * Output: [B, O]
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class MultiLoraMatmul extends DynamicCustomOp {

    @Getter private float alpha = 1.0f;

    /**
     * SameDiff constructor with default alpha.
     */
    public MultiLoraMatmul(SameDiff sameDiff, SDVariable input, SDVariable baseWeight,
                           SDVariable loraA, SDVariable loraB, SDVariable adapterIds) {
        super(null, sameDiff, new SDVariable[]{input, baseWeight, loraA, loraB, adapterIds}, false);
        addTArgument((double) alpha);
    }

    /**
     * SameDiff constructor with double alpha.
     */
    public MultiLoraMatmul(SameDiff sameDiff, SDVariable input, SDVariable baseWeight,
                           SDVariable loraA, SDVariable loraB, SDVariable adapterIds,
                           double alpha) {
        this(sameDiff, input, baseWeight, loraA, loraB, adapterIds, (float) alpha);
    }

    /**
     * Full SameDiff constructor with alpha.
     */
    public MultiLoraMatmul(SameDiff sameDiff, SDVariable input, SDVariable baseWeight,
                           SDVariable loraA, SDVariable loraB, SDVariable adapterIds,
                           float alpha) {
        super(null, sameDiff, new SDVariable[]{input, baseWeight, loraA, loraB, adapterIds}, false);
        this.alpha = alpha;
        addTArgument((double) alpha);
    }

    /**
     * INDArray convenience constructor without output.
     */
    public MultiLoraMatmul(INDArray input, INDArray baseWeight, INDArray loraA, INDArray loraB,
                           INDArray adapterIds, double alpha) {
        this(input, baseWeight, loraA, loraB, adapterIds, null, (float) alpha);
    }

    /**
     * INDArray constructor.
     */
    public MultiLoraMatmul(INDArray input, INDArray baseWeight, INDArray loraA, INDArray loraB,
                           INDArray adapterIds, INDArray output, float alpha) {
        super(null, new INDArray[]{input, baseWeight, loraA, loraB, adapterIds},
                output != null ? new INDArray[]{output} : null);
        this.alpha = alpha;
        addTArgument((double) alpha);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.alpha = tArguments.get(0).floatValue();
    }

    @Override
    public String opName() {
        return "multi_lora_matmul";
    }

    /**
     * Backprop for multi_lora_matmul.
     *
     * <p>Inputs:
     * <ul>
     *   <li>0: input       [B, I]</li>
     *   <li>1: baseWeight  [I, O]  (frozen)</li>
     *   <li>2: loraA       [A, I, R]</li>
     *   <li>3: loraB       [A, R, O]</li>
     *   <li>4: adapterIds  [B] INT64</li>
     * </ul>
     *
     * <p>Returns 5 gradients (one per input).  baseWeight and adapterIds are
     * frozen/discrete — their slots receive zeros.  The fused native bp op
     * {@code multi_lora_matmul_bp} computes dInput, dLoraA, dLoraB.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable gradOut    = gradients.get(0);  // [B, O]
        SDVariable input      = arg(0);  // [B, I]
        SDVariable baseWeight = arg(1);  // [I, O]
        SDVariable loraA      = arg(2);  // [A, I, R]
        SDVariable loraB      = arg(3);  // [A, R, O]
        SDVariable adapterIds = arg(4);  // [B] INT64

        double[] tArgsArr = tArgs();
        double sc = (tArgsArr != null && tArgsArr.length > 0)
            ? tArgsArr[0] : (double) alpha;

        MultiLoraMatmulBp bpOp = new MultiLoraMatmulBp(
            sameDiff, input, baseWeight, loraA, loraB, adapterIds, gradOut, (float) sc);
        SDVariable[] bpOuts = bpOp.outputVariables();

        // bpOuts[0]=dInput, bpOuts[1]=dLoraA, bpOuts[2]=dLoraB
        SDVariable dInput  = bpOuts[0];
        SDVariable dLoraA  = bpOuts[1];
        SDVariable dLoraB  = bpOuts[2];

        // baseWeight frozen; adapterIds is INT64 (no float gradient)
        SDVariable dBase       = sameDiff.zerosLike(baseWeight);
        SDVariable dAdapterIds = sameDiff.zerosLike(adapterIds);

        return Arrays.asList(dInput, dBase, dLoraA, dLoraB, dAdapterIds);
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
