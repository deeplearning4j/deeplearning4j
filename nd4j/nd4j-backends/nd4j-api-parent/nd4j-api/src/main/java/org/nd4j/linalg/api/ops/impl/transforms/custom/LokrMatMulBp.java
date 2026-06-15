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
 * Backpropagation op for LoKr (Low-Rank Kronecker Product) fused matrix multiplication.
 * <p>
 * Computes gradients for all trainable parameters of the LoKr forward pass.
 * <p>
 * Inputs:
 * <ul>
 *   <li>input: [batch, in_features]</li>
 *   <li>weight: [out_features, in_features] (frozen base weight)</li>
 *   <li>lokrC: [f1, f2] (Kronecker factor)</li>
 *   <li>lokrA: [dim, d2] (low-rank decomposition)</li>
 *   <li>lokrB: [d1, dim] (low-rank decomposition)</li>
 *   <li>dLdOut: [batch, out_features] (upstream gradient)</li>
 * </ul>
 * <p>
 * Outputs (5 gradients):
 * <ul>
 *   <li>dLdInput: [batch, in_features]</li>
 *   <li>dLdWeight: [out_features, in_features]</li>
 *   <li>dLdLokrC: [f1, f2]</li>
 *   <li>dLdLokrA: [dim, d2]</li>
 *   <li>dLdLokrB: [d1, dim]</li>
 * </ul>
 *
 * @author Adam Gibson
 * @see LokrMatMul
 */
@NoArgsConstructor
public class LokrMatMulBp extends DynamicCustomOp {

    private double scaling = 1.0;
    private boolean transposeWeight = true;
    private int factor1 = -1;
    private int factor2 = -1;

    public LokrMatMulBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                        @NonNull SDVariable weight, @NonNull SDVariable lokrC,
                        @NonNull SDVariable lokrA, @NonNull SDVariable lokrB,
                        @NonNull SDVariable dLdOut, double scaling,
                        int factor1, int factor2) {
        super(null, sameDiff, new SDVariable[]{input, weight, lokrC, lokrA, lokrB, dLdOut}, false);
        this.scaling = scaling;
        this.factor1 = factor1;
        this.factor2 = factor2;
        addTArgument(scaling);
        addIArgument(factor1, factor2);
        addBArgument(transposeWeight);
    }

    public LokrMatMulBp(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                        @NonNull SDVariable weight, @NonNull SDVariable lokrC,
                        @NonNull SDVariable lokrA, @NonNull SDVariable lokrB,
                        @NonNull SDVariable dLdOut, double scaling,
                        int factor1, int factor2, boolean transposeWeight) {
        super(null, sameDiff, new SDVariable[]{input, weight, lokrC, lokrA, lokrB, dLdOut}, false);
        this.scaling = scaling;
        this.factor1 = factor1;
        this.factor2 = factor2;
        this.transposeWeight = transposeWeight;
        addTArgument(scaling);
        addIArgument(factor1, factor2);
        addBArgument(transposeWeight);
    }

    public LokrMatMulBp(@NonNull INDArray input, @NonNull INDArray weight,
                        @NonNull INDArray lokrC, @NonNull INDArray lokrA,
                        @NonNull INDArray lokrB, @NonNull INDArray dLdOut,
                        double scaling, int factor1, int factor2) {
        super(null, new INDArray[]{input, weight, lokrC, lokrA, lokrB, dLdOut}, null);
        this.scaling = scaling;
        this.factor1 = factor1;
        this.factor2 = factor2;
        addTArgument(scaling);
        addIArgument(factor1, factor2);
        addBArgument(transposeWeight);
    }

    @Override
    public String opName() {
        return "lokr_matmul_bp";
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
