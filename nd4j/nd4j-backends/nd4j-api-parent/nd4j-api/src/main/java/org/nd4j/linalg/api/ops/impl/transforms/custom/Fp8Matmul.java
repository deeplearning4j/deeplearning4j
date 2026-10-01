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

import java.util.Collections;
import java.util.List;

/**
 * Scaled matrix multiplication over FP8 operands.
 * <pre>
 *   C = (op(A) @ op(B)) * scale_A * scale_B + bias
 * </pre>
 * A and B keep their storage types: {@link DataType#FLOAT8} (E4M3FN), {@link DataType#FLOAT8_E5M2} or any wider
 * floating type. The storage type carries the FP8 format, and the product accumulates in the aggregate type of the
 * output.
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: A [M, K], or [K, M] when transposing A</li>
 *   <li>1: B [K, N], or [N, K] when transposing B</li>
 *   <li>2: scale_A (floating) - one dequantization scale, or one per row of op(A) [M]</li>
 *   <li>3: scale_B (floating) - one dequantization scale, or one per column of op(B) [N]</li>
 *   <li>4: bias (optional, floating) [N]; an empty array stands for an absent bias</li>
 * </ul>
 * <p>
 * Output: C [M, N], in the requested output type, else in the type of scale_A.
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: transpose_a (0=no, 1=yes, default: 0)</li>
 *   <li>1: transpose_b (0=no, 1=yes, default: 0)</li>
 * </ul>
 * Data type arguments:
 * <ul>
 *   <li>0: output type (optional, floating)</li>
 * </ul>
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class Fp8Matmul extends DynamicCustomOp {

    @Getter private boolean transposeA = false;
    @Getter private boolean transposeB = false;
    /** The requested output type; null takes the type of scale_A. */
    @Getter private DataType outputDataType;

    /**
     * SameDiff constructor with required inputs.
     *
     * @param sameDiff the SameDiff instance
     * @param a        FP8 A matrix [M, K]
     * @param b        FP8 B matrix [K, N]
     * @param scaleA   dequantization scale for A, one or one per row
     * @param scaleB   dequantization scale for B, one or one per column
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB) {
        this(sameDiff, a, b, scaleA, scaleB, null, false, false, null);
    }

    /**
     * SameDiff constructor with bias.
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB, SDVariable bias) {
        this(sameDiff, a, b, scaleA, scaleB, bias, false, false, null);
    }

    /**
     * SameDiff constructor with transposes.
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB, SDVariable bias,
                     boolean transposeA, boolean transposeB) {
        this(sameDiff, a, b, scaleA, scaleB, bias, transposeA, transposeB, null);
    }

    /**
     * Full SameDiff constructor with all options.
     *
     * @param outputDataType floating output type, or null for the type of scaleA
     */
    public Fp8Matmul(SameDiff sameDiff, SDVariable a, SDVariable b,
                     SDVariable scaleA, SDVariable scaleB, SDVariable bias,
                     boolean transposeA, boolean transposeB, DataType outputDataType) {
        super(null, sameDiff, bias != null ?
                new SDVariable[]{a, b, scaleA, scaleB, bias} :
                new SDVariable[]{a, b, scaleA, scaleB}, false);
        configure(transposeA, transposeB, outputDataType);
    }

    /**
     * INDArray convenience constructor with defaults.
     */
    public Fp8Matmul(INDArray a, INDArray b, INDArray scaleA, INDArray scaleB) {
        this(a, b, scaleA, scaleB, null, null, false, false);
    }

    /**
     * INDArray constructor. A given output array sets the output type; without one the output takes the type of
     * scaleA.
     */
    public Fp8Matmul(INDArray a, INDArray b, INDArray scaleA, INDArray scaleB,
                     INDArray bias, INDArray output,
                     boolean transposeA, boolean transposeB) {
        super(null, bias != null ?
                new INDArray[]{a, b, scaleA, scaleB, bias} :
                new INDArray[]{a, b, scaleA, scaleB},
                output != null ? new INDArray[]{output} : null);
        configure(transposeA, transposeB, output != null ? output.dataType() : null);
    }

    private void configure(boolean transposeA, boolean transposeB, DataType outputDataType) {
        Preconditions.checkArgument(outputDataType == null || outputDataType.isFPType(),
                "fp8_matmul output type must be floating, got %s", outputDataType);
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.outputDataType = outputDataType;
        addIArgument(transposeA ? 1L : 0L, transposeB ? 1L : 0L);
        if (outputDataType != null) {
            addDArgument(outputDataType);
        }
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.transposeA = iArguments.get(0) != 0;
        if (iArguments.size() > 1) this.transposeB = iArguments.get(1) != 0;
        if (dArguments != null && !dArguments.isEmpty()) this.outputDataType = dArguments.get(0);
    }

    @Override
    public String opName() {
        return "fp8_matmul";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 4,
                "Expected at least 4 input data types for fp8_matmul, got %s", inputDataTypes);
        if (dArguments != null && !dArguments.isEmpty()) {
            return Collections.singletonList(dArguments.get(0));
        }
        return Collections.singletonList(inputDataTypes.get(2));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
