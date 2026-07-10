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
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * LoRA (Low-Rank Adaptation) fused matrix multiplication operation.
 * <p>
 * Computes: output = input @ weight + scaling * (input @ A^T @ B^T)
 * <p>
 * This fused operation is more efficient than computing the two matmuls
 * separately, and supports automatic differentiation for training LoRA adapters.
 * <p>
 * Input shapes:
 * <ul>
 *   <li>input: [batch, in_features]</li>
 *   <li>weight: [out_features, in_features] (frozen base weight)</li>
 *   <li>loraA: [r, in_features] (trainable)</li>
 *   <li>loraB: [out_features, r] (trainable, initialized to zeros)</li>
 * </ul>
 * <p>
 * Output shape: [batch, out_features]
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class LoraMatMul extends DynamicCustomOp {

    private double scaling = 1.0;
    private boolean transposeWeight = true;
    private double dropout = 0.0;

    public LoraMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable loraA,
                      @NonNull SDVariable loraB, double scaling) {
        super(null, sameDiff, new SDVariable[]{input, weight, loraA, loraB}, false);
        this.scaling = scaling;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LoraMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable loraA,
                      @NonNull SDVariable loraB, double scaling, double dropout) {
        super(null, sameDiff, new SDVariable[]{input, weight, loraA, loraB}, false);
        this.scaling = scaling;
        this.dropout = dropout;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LoraMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable loraA,
                      @NonNull SDVariable loraB, double scaling, boolean transposeWeight) {
        super(null, sameDiff, new SDVariable[]{input, weight, loraA, loraB}, false);
        this.scaling = scaling;
        this.transposeWeight = transposeWeight;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LoraMatMul(@NonNull INDArray input, @NonNull INDArray weight,
                      @NonNull INDArray loraA, @NonNull INDArray loraB,
                      double scaling, boolean transposeWeight) {
        super(null, new INDArray[]{input, weight, loraA, loraB}, null);
        this.scaling = scaling;
        this.transposeWeight = transposeWeight;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LoraMatMul(@NonNull INDArray input, @NonNull INDArray weight,
                      @NonNull INDArray loraA, @NonNull INDArray loraB,
                      double scaling, INDArray output) {
        super(null, new INDArray[]{input, weight, loraA, loraB},
              output == null ? null : new INDArray[]{output});
        this.scaling = scaling;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    @Override
    public String opName() {
        return "lora_matmul";
    }

    /**
     * Compute gradients for LoRA matmul.
     *
     * <p>Handles rank-2 [batch, in_features] and rank-3 [B, S, in_features] activations.
     * For rank-3, activations are flattened to [B*S, in_features] before the 2D grad
     * math, then the result is reshaped back to the original leading dims.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable grad   = gradients.get(0); // [batch, out_features] or [B,S,out_features]
        SDVariable input  = arg(0);            // [batch, in_features]  or [B,S,in_features]
        SDVariable weight = arg(1);            // [out_features, in_features]
        SDVariable loraA  = arg(2);            // [r, in_features]
        SDVariable loraB  = arg(3);            // [out_features, r]

        // Detect rank-3 (e.g. [B, S, F]) — flatten to 2D for the matmul grads
        long[] inputShape = input.getShape();
        boolean rank3 = (inputShape != null && inputShape.length == 3);

        SDVariable inputFlat;
        SDVariable gradFlat;
        long[] leadDims = null;
        if (rank3) {
            leadDims  = new long[]{inputShape[0], inputShape[1]};
            long inF  = inputShape[2];
            inputFlat = input.reshape(-1, inF);
            // out_features must come from the frozen weight [out,in] (always concrete), NOT
            // from the incoming gradient's static shape — which is frequently unknown (-1) at
            // doDiff time. Using grad.getShape() there produced reshape(-1, -1) (two unknown
            // dims), which the reshape op rejects.
            long[] wShape = weight.getShape();
            long outF = (wShape != null && wShape.length == 2) ? wShape[0] : loraB.getShape()[0];
            gradFlat  = grad.reshape(-1, outF);
        } else {
            inputFlat = input;
            gradFlat  = grad;
        }

        // grad_input_flat = gradFlat @ weight + scaling * (gradFlat @ loraB) @ loraA
        SDVariable weightGradPart = sameDiff.mmul(gradFlat, weight);
        SDVariable loraGradPart   = sameDiff.mmul(
            sameDiff.mmul(gradFlat, loraB), loraA).mul(scaling);
        SDVariable gradInputFlat  = weightGradPart.add(loraGradPart);

        SDVariable gradInput;
        if (rank3 && leadDims != null && inputShape != null) {
            gradInput = gradInputFlat.reshape(leadDims[0], leadDims[1], inputShape[2]);
        } else {
            gradInput = gradInputFlat;
        }

        // Weight frozen
        SDVariable gradWeight = sameDiff.zerosLike(weight);

        // grad_A = scaling * (gradFlat @ loraB)^T @ inputFlat
        SDVariable gradTimesB = sameDiff.mmul(gradFlat, loraB); // [M, r]
        SDVariable gradA = sameDiff.mmul(
            sameDiff.transpose(gradTimesB),  // [r, M]
            inputFlat                         // [M, inF]
        ).mul(scaling);                       // [r, inF]

        // grad_B = scaling * gradFlat^T @ (inputFlat @ loraA^T)
        SDVariable inputTimesAT = sameDiff.mmul(inputFlat, sameDiff.transpose(loraA)); // [M, r]
        SDVariable gradB = sameDiff.mmul(
            sameDiff.transpose(gradFlat),  // [outF, M]
            inputTimesAT                    // [M, r]
        ).mul(scaling);                     // [outF, r]

        return Arrays.asList(gradInput, gradWeight, gradA, gradB);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 4,
            "Expected 4 input data types, got %s", inputDataTypes);
        // Output type matches input type (or promoted type)
        DataType outType = inputDataTypes.get(0);
        if (inputDataTypes.get(1) == DataType.DOUBLE || inputDataTypes.get(2) == DataType.DOUBLE) {
            outType = DataType.DOUBLE;
        } else if (inputDataTypes.get(1) == DataType.FLOAT || inputDataTypes.get(2) == DataType.FLOAT) {
            outType = DataType.FLOAT;
        }
        return Collections.singletonList(outType);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public String tensorflowName() {
        return "LoraMatMul";
    }

    @Override
    public String onnxName() {
        return "LoraMatMul";
    }
}
