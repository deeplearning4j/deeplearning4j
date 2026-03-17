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

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable grad = gradients.get(0); // [batch, out_features]

        SDVariable input = arg(0);  // [batch, in_features]
        SDVariable weight = arg(1); // [out_features, in_features]
        SDVariable loraA = arg(2);  // [r, in_features]
        SDVariable loraB = arg(3);  // [out_features, r]

        // Gradient w.r.t. input:
        // d(output)/d(input) = weight^T + scaling * A^T @ B^T
        // grad_input = grad @ (weight + scaling * B @ A)
        SDVariable weightGradPart = sameDiff.mmul(grad, weight); // grad @ weight
        SDVariable loraGradPart = sameDiff.mmul(
            sameDiff.mmul(grad, loraB), // grad @ B: [batch, r]
            loraA                        // @ A: [batch, in_features]
        ).mul(scaling);
        SDVariable gradInput = weightGradPart.add(loraGradPart);

        // Gradient w.r.t. weight (frozen, return zeros or skip)
        SDVariable gradWeight = sameDiff.zerosLike(weight);

        // Gradient w.r.t. loraA:
        // d(output)/d(A) involves: scaling * input^T @ (grad @ B)
        // grad_A = scaling * (grad @ B)^T @ input = scaling * B^T @ grad^T @ input
        SDVariable gradTimesB = sameDiff.mmul(grad, loraB); // [batch, r]
        SDVariable gradA = sameDiff.mmul(
            sameDiff.transpose(gradTimesB), // [r, batch]
            input                            // [batch, in_features]
        ).mul(scaling); // [r, in_features]

        // Gradient w.r.t. loraB:
        // d(output)/d(B) = scaling * grad^T @ (input @ A^T)
        SDVariable inputTimesA = sameDiff.mmul(input, sameDiff.transpose(loraA)); // [batch, r]
        SDVariable gradB = sameDiff.mmul(
            sameDiff.transpose(grad), // [out_features, batch]
            inputTimesA               // [batch, r]
        ).mul(scaling); // [out_features, r]

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
