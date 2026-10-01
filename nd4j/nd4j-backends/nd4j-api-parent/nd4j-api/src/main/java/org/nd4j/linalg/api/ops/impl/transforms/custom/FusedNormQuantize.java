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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Normalization followed by per-row symmetric quantization.
 * <p>
 * The rows of the last axis normalize, by RMSNorm or LayerNorm, scale by gamma and shift by the optional beta. Each
 * normalized row then quantizes symmetrically onto the grid of the quantized data type:
 * <pre>
 *   scale = max|row| / qmax
 *   code  = row / scale, rounded to the nearest value of the quantized type
 * </pre>
 * where qmax is the largest finite value of the quantized type (127 for INT8, 448 for FLOAT8), so code * scale
 * reconstructs the row. The quantized type is any signed integer or floating type, FP8 included.
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [..., H] (floating); the last axis holds the features</li>
 *   <li>1: gamma [H] (floating) - normalization scale</li>
 *   <li>2: beta [H] (optional, floating) - normalization shift; an empty array stands for an absent beta</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: normalization type ({@link #NORM_RMSNORM} or {@link #NORM_LAYERNORM}, default: RMSNorm)</li>
 * </ul>
 * Floating arguments:
 * <ul>
 *   <li>0: epsilon added to the mean square of each row (default: 1e-5)</li>
 * </ul>
 * Data type arguments:
 * <ul>
 *   <li>0: quantized type (default: {@link DataType#INT8})</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: codes, in the input's shape and the quantized type</li>
 *   <li>1: scales, in the input's shape without the last axis and the input's type</li>
 * </ul>
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class FusedNormQuantize extends DynamicCustomOp {

    public static final int NORM_RMSNORM = 0;
    public static final int NORM_LAYERNORM = 1;
    public static final double DEFAULT_EPSILON = 1e-5;

    @Getter private int normType = NORM_RMSNORM;
    @Getter private DataType quantizedType = DataType.INT8;
    @Getter private double epsilon = DEFAULT_EPSILON;

    /**
     * RMSNorm into INT8 codes.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable gamma) {
        this(sameDiff, input, gamma, null, NORM_RMSNORM, DataType.INT8, DEFAULT_EPSILON);
    }

    /**
     * RMSNorm shifted by beta, into INT8 codes.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable gamma, SDVariable beta) {
        this(sameDiff, input, gamma, beta, NORM_RMSNORM, DataType.INT8, DEFAULT_EPSILON);
    }

    /**
     * RMSNorm into codes of the given type.
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable gamma, double epsilon,
                             DataType quantizedType) {
        this(sameDiff, input, gamma, null, NORM_RMSNORM, quantizedType, epsilon);
    }

    /**
     * Full SameDiff constructor.
     *
     * @param beta          normalization shift, or null for none
     * @param normType      {@link #NORM_RMSNORM} or {@link #NORM_LAYERNORM}
     * @param quantizedType data type of the codes: a signed integer or floating type
     */
    public FusedNormQuantize(SameDiff sameDiff, SDVariable input, SDVariable gamma, SDVariable beta,
                             int normType, DataType quantizedType, double epsilon) {
        super(null, sameDiff, beta != null ?
                new SDVariable[]{input, gamma, beta} :
                new SDVariable[]{input, gamma}, false);
        configure(normType, quantizedType, epsilon);
    }

    /**
     * RMSNorm into codes of the given type.
     */
    public FusedNormQuantize(INDArray input, INDArray gamma, double epsilon, DataType quantizedType) {
        this(input, gamma, null, null, null, NORM_RMSNORM, quantizedType, epsilon);
    }

    /**
     * Full INDArray constructor. The codes and scales arrays are given together or not at all; given codes must hold
     * the quantized type.
     */
    public FusedNormQuantize(INDArray input, INDArray gamma, INDArray beta, INDArray codes, INDArray scales,
                             int normType, DataType quantizedType, double epsilon) {
        super(null, beta != null ?
                new INDArray[]{input, gamma, beta} :
                new INDArray[]{input, gamma},
                codes != null ? new INDArray[]{codes, scales} : null);
        Preconditions.checkArgument((codes == null) == (scales == null),
                "fused_norm_quantize takes both output arrays or neither");
        Preconditions.checkArgument(codes == null || codes.dataType() == quantizedType,
                "fused_norm_quantize codes array holds %s but the quantized type is %s",
                codes == null ? null : codes.dataType(), quantizedType);
        configure(normType, quantizedType, epsilon);
    }

    private void configure(int normType, DataType quantizedType, double epsilon) {
        Preconditions.checkArgument(normType == NORM_RMSNORM || normType == NORM_LAYERNORM,
                "fused_norm_quantize normalization type must be NORM_RMSNORM or NORM_LAYERNORM, got %s", normType);
        Preconditions.checkArgument(quantizedType != null && quantizedType.isSigned()
                        && (quantizedType.isIntType() || quantizedType.isFPType()),
                "fused_norm_quantize codes need a signed integer or floating type, got %s", quantizedType);
        this.normType = normType;
        this.quantizedType = quantizedType;
        this.epsilon = epsilon;
        addIArgument(normType);
        addTArgument(epsilon);
        addDArgument(quantizedType);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (!iArguments.isEmpty()) this.normType = iArguments.get(0).intValue();
        if (!tArguments.isEmpty()) this.epsilon = tArguments.get(0);
        if (dArguments != null && !dArguments.isEmpty()) this.quantizedType = dArguments.get(0);
    }

    @Override
    public String opName() {
        return "fused_norm_quantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 2,
                "Expected at least 2 input data types for fused_norm_quantize, got %s", inputDataTypes);
        DataType codes = dArguments != null && !dArguments.isEmpty() ? dArguments.get(0) : DataType.INT8;
        return Arrays.asList(codes, inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
