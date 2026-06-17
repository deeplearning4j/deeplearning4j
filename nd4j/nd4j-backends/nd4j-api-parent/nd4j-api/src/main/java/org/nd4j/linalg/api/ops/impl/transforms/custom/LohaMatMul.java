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
 * LoHa (Low-Rank Hadamard Product) fused matrix multiplication operation.
 * <p>
 * Computes: output = input @ weight^T + scaling * input @ ((B1 @ A1) ⊙ (B2 @ A2))^T
 * <p>
 * where ⊙ is the Hadamard (element-wise) product. LoHa uses two low-rank decompositions
 * whose Hadamard product can represent higher effective ranks. With dimension d,
 * the effective rank can be up to d².
 * <p>
 * Input shapes:
 * <ul>
 *   <li>input: [batch, in_features]</li>
 *   <li>weight: [out_features, in_features] (frozen base weight)</li>
 *   <li>lohaA1: [dim, in_features] (trainable)</li>
 *   <li>lohaB1: [out_features, dim] (trainable)</li>
 *   <li>lohaA2: [dim, in_features] (trainable)</li>
 *   <li>lohaB2: [out_features, dim] (trainable)</li>
 * </ul>
 * <p>
 * Output shape: [batch, out_features]
 *
 * @author Adam Gibson
 * @see LoraMatMul
 */
@NoArgsConstructor
public class LohaMatMul extends DynamicCustomOp {

    private double scaling = 1.0;
    private boolean transposeWeight = true;
    private double dropout = 0.0;

    public LohaMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable lohaA1,
                      @NonNull SDVariable lohaB1, @NonNull SDVariable lohaA2,
                      @NonNull SDVariable lohaB2, double scaling) {
        super(null, sameDiff, new SDVariable[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2}, false);
        this.scaling = scaling;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LohaMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable lohaA1,
                      @NonNull SDVariable lohaB1, @NonNull SDVariable lohaA2,
                      @NonNull SDVariable lohaB2, double scaling, double dropout) {
        super(null, sameDiff, new SDVariable[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2}, false);
        this.scaling = scaling;
        this.dropout = dropout;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LohaMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable lohaA1,
                      @NonNull SDVariable lohaB1, @NonNull SDVariable lohaA2,
                      @NonNull SDVariable lohaB2, double scaling, boolean transposeWeight) {
        super(null, sameDiff, new SDVariable[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2}, false);
        this.scaling = scaling;
        this.transposeWeight = transposeWeight;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LohaMatMul(@NonNull INDArray input, @NonNull INDArray weight,
                      @NonNull INDArray lohaA1, @NonNull INDArray lohaB1,
                      @NonNull INDArray lohaA2, @NonNull INDArray lohaB2,
                      double scaling, boolean transposeWeight) {
        super(null, new INDArray[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2}, null);
        this.scaling = scaling;
        this.transposeWeight = transposeWeight;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    public LohaMatMul(@NonNull INDArray input, @NonNull INDArray weight,
                      @NonNull INDArray lohaA1, @NonNull INDArray lohaB1,
                      @NonNull INDArray lohaA2, @NonNull INDArray lohaB2,
                      double scaling, INDArray output) {
        super(null, new INDArray[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2},
              output == null ? null : new INDArray[]{output});
        this.scaling = scaling;
        addTArgument(scaling, dropout);
        addBArgument(transposeWeight);
    }

    @Override
    public String opName() {
        return "loha_matmul";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable grad = gradients.get(0); // [batch, out_features]

        SDVariable input = arg(0);   // [batch, in_features]
        SDVariable weight = arg(1);  // [out_features, in_features]
        SDVariable lohaA1 = arg(2);  // [dim, in_features]
        SDVariable lohaB1 = arg(3);  // [out_features, dim]
        SDVariable lohaA2 = arg(4);  // [dim, in_features]
        SDVariable lohaB2 = arg(5);  // [out_features, dim]

        // Compute intermediate products for gradients
        // prod1 = B1 @ A1: [out_features, in_features]
        // prod2 = B2 @ A2: [out_features, in_features]
        SDVariable prod1 = sameDiff.mmul(lohaB1, lohaA1);
        SDVariable prod2 = sameDiff.mmul(lohaB2, lohaA2);

        // lohaDelta = prod1 * prod2 (Hadamard product)
        SDVariable lohaDelta = prod1.mul(prod2);

        // Gradient w.r.t. input:
        // grad_input = grad @ weight + scaling * grad @ lohaDelta
        SDVariable gradInputBase = sameDiff.mmul(grad, weight);
        SDVariable gradInputLoha = sameDiff.mmul(grad, lohaDelta).mul(scaling);
        SDVariable gradInput = gradInputBase.add(gradInputLoha);

        // Gradient w.r.t. weight (frozen)
        SDVariable gradWeight = sameDiff.zerosLike(weight);

        // For LoHa, the delta is (B1 @ A1) ⊙ (B2 @ A2)
        // Gradient flows through both decompositions

        // Gradient w.r.t. A1, B1 (through prod1, with prod2 as coefficient)
        // d/dA1: scaling * (grad @ (delta * prod2))^T @ ... but delta uses prod1
        // Actually: d(output)/d(prod1) = scaling * prod2 (element-wise)
        // Then d(prod1)/dA1 and d(prod1)/dB1

        // Simplified gradient computation using chain rule:
        // gradProd1 = scaling * grad^T @ input @ A1^T ⊙ prod2 for B1
        // This is an approximation - full gradient is more complex

        SDVariable inputTimesGrad = sameDiff.mmul(sameDiff.transpose(grad), input); // [out, in]
        SDVariable gradProd1Contribution = inputTimesGrad.mul(prod2).mul(scaling);
        SDVariable gradProd2Contribution = inputTimesGrad.mul(prod1).mul(scaling);

        // Gradient w.r.t. A1: B1^T @ gradProd1
        SDVariable gradA1 = sameDiff.mmul(sameDiff.transpose(lohaB1), gradProd1Contribution);

        // Gradient w.r.t. B1: gradProd1 @ A1^T
        SDVariable gradB1 = sameDiff.mmul(gradProd1Contribution, sameDiff.transpose(lohaA1));

        // Gradient w.r.t. A2: B2^T @ gradProd2
        SDVariable gradA2 = sameDiff.mmul(sameDiff.transpose(lohaB2), gradProd2Contribution);

        // Gradient w.r.t. B2: gradProd2 @ A2^T
        SDVariable gradB2 = sameDiff.mmul(gradProd2Contribution, sameDiff.transpose(lohaA2));

        return Arrays.asList(gradInput, gradWeight, gradA1, gradB1, gradA2, gradB2);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 6,
            "Expected 6 input data types, got %s", inputDataTypes);
        DataType outType = inputDataTypes.get(0);
        for (int i = 1; i < inputDataTypes.size(); i++) {
            if (inputDataTypes.get(i) == DataType.DOUBLE) {
                outType = DataType.DOUBLE;
                break;
            } else if (inputDataTypes.get(i) == DataType.FLOAT && outType != DataType.DOUBLE) {
                outType = DataType.FLOAT;
            }
        }
        return Collections.singletonList(outType);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public String tensorflowName() {
        return "LohaMatMul";
    }

    @Override
    public String onnxName() {
        return "LohaMatMul";
    }
}
