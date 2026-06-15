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
import java.util.List;

/**
 * Backpropagation op for LoHa (Low-Rank Hadamard Product) fused matrix multiplication.
 * <p>
 * Computes gradients for all trainable parameters of the LoHa forward pass.
 * <p>
 * Inputs:
 * <ul>
 *   <li>input: [batch, in_features]</li>
 *   <li>weight: [out_features, in_features] (frozen base weight)</li>
 *   <li>lohaA1: [dim, in_features] (trainable)</li>
 *   <li>lohaB1: [out_features, dim] (trainable)</li>
 *   <li>lohaA2: [dim, in_features] (trainable)</li>
 *   <li>lohaB2: [out_features, dim] (trainable)</li>
 *   <li>dLdOut: [batch, out_features] (upstream gradient)</li>
 * </ul>
 * <p>
 * Outputs (6 gradients):
 * <ul>
 *   <li>dLdInput: [batch, in_features]</li>
 *   <li>dLdWeight: [out_features, in_features]</li>
 *   <li>dLdLohaA1: [dim, in_features]</li>
 *   <li>dLdLohaB1: [out_features, dim]</li>
 *   <li>dLdLohaA2: [dim, in_features]</li>
 *   <li>dLdLohaB2: [out_features, dim]</li>
 * </ul>
 *
 * @author Adam Gibson
 * @see LohaMatMul
 */
@NoArgsConstructor
public class LohaMatMulBp extends DynamicCustomOp {

    private double scaling = 1.0;
    private boolean transposeWeight = true;

    public LohaMatMulBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                        @NonNull SDVariable weight, @NonNull SDVariable lohaA1,
                        @NonNull SDVariable lohaB1, @NonNull SDVariable lohaA2,
                        @NonNull SDVariable lohaB2, @NonNull SDVariable dLdOut,
                        double scaling) {
        super(null, sameDiff, new SDVariable[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2, dLdOut}, false);
        this.scaling = scaling;
        addTArgument(scaling);
        addBArgument(transposeWeight);
    }

    public LohaMatMulBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                        @NonNull SDVariable weight, @NonNull SDVariable lohaA1,
                        @NonNull SDVariable lohaB1, @NonNull SDVariable lohaA2,
                        @NonNull SDVariable lohaB2, @NonNull SDVariable dLdOut,
                        double scaling, boolean transposeWeight) {
        super(null, sameDiff, new SDVariable[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2, dLdOut}, false);
        this.scaling = scaling;
        this.transposeWeight = transposeWeight;
        addTArgument(scaling);
        addBArgument(transposeWeight);
    }

    public LohaMatMulBp(@NonNull INDArray input, @NonNull INDArray weight,
                        @NonNull INDArray lohaA1, @NonNull INDArray lohaB1,
                        @NonNull INDArray lohaA2, @NonNull INDArray lohaB2,
                        @NonNull INDArray dLdOut, double scaling) {
        super(null, new INDArray[]{input, weight, lohaA1, lohaB1, lohaA2, lohaB2, dLdOut}, null);
        this.scaling = scaling;
        addTArgument(scaling);
        addBArgument(transposeWeight);
    }

    @Override
    public String opName() {
        return "loha_matmul_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 7,
            "Expected 7 input data types, got %s", inputDataTypes);
        DataType outType = inputDataTypes.get(0);
        for (int i = 1; i < inputDataTypes.size(); i++) {
            if (inputDataTypes.get(i) == DataType.DOUBLE) {
                outType = DataType.DOUBLE;
                break;
            } else if (inputDataTypes.get(i) == DataType.FLOAT && outType != DataType.DOUBLE) {
                outType = DataType.FLOAT;
            }
        }
        return Arrays.asList(outType, outType, outType, outType, outType, outType);
    }

    @Override
    public int getNumOutputs() {
        return 6;
    }
}
