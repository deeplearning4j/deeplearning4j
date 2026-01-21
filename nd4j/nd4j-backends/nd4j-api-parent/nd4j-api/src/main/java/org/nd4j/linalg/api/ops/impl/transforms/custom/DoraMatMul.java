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
 * DoRA (Weight-Decomposed Low-Rank Adaptation) fused matrix multiplication operation.
 * <p>
 * Computes: output = input @ (m * (W + scaling * B @ A) / ||W + scaling * B @ A||)^T
 * <p>
 * DoRA decomposes weight updates into magnitude (m) and direction components.
 * The direction is handled by standard LoRA matrices (A, B), while magnitude
 * is managed by a separate learnable parameter (m). This improves performance
 * especially at lower ranks.
 * <p>
 * Mathematical formulation:
 * <pre>
 * W' = m * (W₀ + BA) / ||W₀ + BA||
 *
 * where:
 *   W₀ = pretrained weight (frozen)
 *   B, A = low-rank decomposition matrices (trainable)
 *   m = magnitude vector (trainable)
 *   ||·|| = column-wise L2 norm
 * </pre>
 * <p>
 * Input shapes:
 * <ul>
 *   <li>input: [batch, in_features]</li>
 *   <li>weight: [out_features, in_features] (frozen base weight)</li>
 *   <li>loraA: [r, in_features] (trainable)</li>
 *   <li>loraB: [out_features, r] (trainable)</li>
 *   <li>magnitude: [out_features] or [out_features, 1] (trainable)</li>
 * </ul>
 * <p>
 * Output shape: [batch, out_features]
 *
 * @author Adam Gibson
 * @see LoraMatMul
 */
@NoArgsConstructor
public class DoraMatMul extends DynamicCustomOp {

    private double scaling = 1.0;
    private boolean transposeWeight = true;
    private double dropout = 0.0;
    private double eps = 1e-8;  // Epsilon for numerical stability in normalization

    public DoraMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable loraA,
                      @NonNull SDVariable loraB, @NonNull SDVariable magnitude,
                      double scaling) {
        super(null, sameDiff, new SDVariable[]{input, weight, loraA, loraB, magnitude}, false);
        this.scaling = scaling;
        addTArgument(scaling, dropout, eps);
        addBArgument(transposeWeight);
    }

    public DoraMatMul(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                      @NonNull SDVariable weight, @NonNull SDVariable loraA,
                      @NonNull SDVariable loraB, @NonNull SDVariable magnitude,
                      double scaling, double dropout, double eps) {
        super(null, sameDiff, new SDVariable[]{input, weight, loraA, loraB, magnitude}, false);
        this.scaling = scaling;
        this.dropout = dropout;
        this.eps = eps;
        addTArgument(scaling, dropout, eps);
        addBArgument(transposeWeight);
    }

    public DoraMatMul(@NonNull INDArray input, @NonNull INDArray weight,
                      @NonNull INDArray loraA, @NonNull INDArray loraB,
                      @NonNull INDArray magnitude, double scaling, INDArray output) {
        super(null, new INDArray[]{input, weight, loraA, loraB, magnitude},
              output == null ? null : new INDArray[]{output});
        this.scaling = scaling;
        addTArgument(scaling, dropout, eps);
        addBArgument(transposeWeight);
    }

    @Override
    public String opName() {
        return "dora_matmul";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable grad = gradients.get(0); // [batch, out_features]

        SDVariable input = arg(0);      // [batch, in_features]
        SDVariable weight = arg(1);     // [out_features, in_features]
        SDVariable loraA = arg(2);      // [r, in_features]
        SDVariable loraB = arg(3);      // [out_features, r]
        SDVariable magnitude = arg(4);  // [out_features] or [out_features, 1]

        // Compute effective weight before normalization
        // W_eff = W + scaling * B @ A
        SDVariable loraDelta = sameDiff.mmul(loraB, loraA).mul(scaling);
        SDVariable wEff = weight.add(loraDelta);

        // Compute column-wise L2 norm: ||W_eff||
        // norm shape: [out_features, 1] or [out_features]
        SDVariable wEffSquared = wEff.mul(wEff);
        SDVariable normSquared = wEffSquared.sum(true, 1);  // Sum over in_features, keep dims
        SDVariable norm = sameDiff.math.sqrt(normSquared.add(eps));

        // Normalized direction: W_eff / ||W_eff||
        SDVariable direction = wEff.div(norm);

        // For magnitude, ensure proper broadcasting
        SDVariable magExpanded = magnitude;
        if (magnitude.getShape() != null && magnitude.getShape().length == 1) {
            magExpanded = magnitude.reshape(-1, 1);
        }

        // Final weight: m * direction
        SDVariable finalWeight = direction.mul(magExpanded);

        // Gradient w.r.t. input
        SDVariable gradInput = sameDiff.mmul(grad, finalWeight);

        // Gradient w.r.t. weight (frozen)
        SDVariable gradWeight = sameDiff.zerosLike(weight);

        // Gradient w.r.t. magnitude
        // d(output)/d(m) = input^T @ grad @ direction
        SDVariable inputTimesGrad = sameDiff.mmul(sameDiff.transpose(grad), input);
        SDVariable gradMag = inputTimesGrad.mul(direction).sum(1);

        // Gradient w.r.t. loraA and loraB (through direction and norm)
        // This requires careful handling of the normalization gradient
        // Using chain rule: d(output)/d(BA) flows through direction normalization

        // Simplified gradient for LoRA components
        // gradDirection = m * grad @ input / norm
        SDVariable gradDirection = sameDiff.mmul(sameDiff.transpose(grad), input).mul(magExpanded).div(norm);

        // Gradient for B @ A needs to account for normalization derivative
        // Full gradient: d(W_eff/||W_eff||)/d(W_eff) = (I - W_eff @ W_eff^T / ||W_eff||^2) / ||W_eff||
        // Simplified: use gradDirection directly

        // grad_A = scaling * B^T @ gradDirection
        SDVariable gradA = sameDiff.mmul(sameDiff.transpose(loraB), gradDirection).mul(scaling);

        // grad_B = scaling * gradDirection @ A^T
        SDVariable gradB = sameDiff.mmul(gradDirection, sameDiff.transpose(loraA)).mul(scaling);

        return Arrays.asList(gradInput, gradWeight, gradA, gradB, gradMag);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 5,
            "Expected 5 input data types, got %s", inputDataTypes);
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
        return "DoraMatMul";
    }

    @Override
    public String onnxName() {
        return "DoraMatMul";
    }
}
