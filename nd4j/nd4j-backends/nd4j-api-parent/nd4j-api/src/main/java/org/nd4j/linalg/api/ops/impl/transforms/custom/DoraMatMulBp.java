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
 * Backpropagation op for DoRA (Weight-Decomposed Low-Rank Adaptation) fused matrix multiplication.
 * <p>
 * Computes gradients for all trainable parameters of the DoRA forward pass.
 * <p>
 * Inputs:
 * <ul>
 *   <li>input: [batch, in_features]</li>
 *   <li>weight: [out_features, in_features] (frozen base weight)</li>
 *   <li>loraA: [r, in_features] (trainable)</li>
 *   <li>loraB: [out_features, r] (trainable)</li>
 *   <li>magnitude: [out_features] or [out_features, 1] (trainable)</li>
 *   <li>dLdOut: [batch, out_features] (upstream gradient)</li>
 * </ul>
 * <p>
 * Outputs (5 gradients):
 * <ul>
 *   <li>dLdInput: [batch, in_features]</li>
 *   <li>dLdWeight: [out_features, in_features]</li>
 *   <li>dLdLoraA: [r, in_features]</li>
 *   <li>dLdLoraB: [out_features, r]</li>
 *   <li>dLdMagnitude: [out_features]</li>
 * </ul>
 *
 * @author Adam Gibson
 * @see DoraMatMul
 */
@NoArgsConstructor
public class DoraMatMulBp extends DynamicCustomOp {

    private double scaling = 1.0;
    private double eps = 1e-8;

    public DoraMatMulBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                        @NonNull SDVariable weight, @NonNull SDVariable loraA,
                        @NonNull SDVariable loraB, @NonNull SDVariable magnitude,
                        @NonNull SDVariable dLdOut, double scaling, double eps) {
        super(null, sameDiff, new SDVariable[]{input, weight, loraA, loraB, magnitude, dLdOut}, false);
        this.scaling = scaling;
        this.eps = eps;
        addTArgument(scaling, 0.0, eps);
        addBArgument(true);
    }

    public DoraMatMulBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                        @NonNull SDVariable weight, @NonNull SDVariable loraA,
                        @NonNull SDVariable loraB, @NonNull SDVariable magnitude,
                        @NonNull SDVariable dLdOut, double scaling) {
        this(sameDiff, input, weight, loraA, loraB, magnitude, dLdOut, scaling, 1e-8);
    }

    public DoraMatMulBp(@NonNull INDArray input, @NonNull INDArray weight,
                        @NonNull INDArray loraA, @NonNull INDArray loraB,
                        @NonNull INDArray magnitude, @NonNull INDArray dLdOut,
                        double scaling, double eps) {
        super(null, new INDArray[]{input, weight, loraA, loraB, magnitude, dLdOut}, null);
        this.scaling = scaling;
        this.eps = eps;
        addTArgument(scaling, 0.0, eps);
        addBArgument(true);
    }

    @Override
    public String opName() {
        return "dora_matmul_bp";
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
        return Arrays.asList(outType, outType, outType, outType, outType);
    }

    @Override
    public int getNumOutputs() {
        return 5;
    }
}
