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
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * AWQ (Activation-aware Weight Quantization) matrix multiplication.
 * <p>
 * Multiplies by a weight W [K, N] stored as numBits-wide unsigned codes packed along K, which dequantizes per group
 * of groupSize input channels and per output channel:
 * <pre>
 *   W[k, n] = (code(k, n) - zeros[k / groupSize, n]) * scales[k / groupSize, n]
 *   output  = input @ W + bias
 * </pre>
 * Byte (c, n) of the packed weights holds the codes of rows c * (8 / numBits) + j in bits
 * [j * numBits, (j + 1) * numBits). Without zeros the zero point is the middle of the code range, 2^(numBits - 1).
 * W dequantizes in the aggregate type of the output, where the product accumulates, so each weight rounds once.
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input (floating) [..., K] - activations</li>
 *   <li>1: weightPacked (one-byte integer codes) [ceil(K * numBits / 8), N]</li>
 *   <li>2: scales (floating) [ceil(K / groupSize), N] - per-group dequantization scales</li>
 *   <li>3: zeros (optional) - per-group zero points in the type and shape of the scales; an empty array stands for
 *   absent zeros</li>
 *   <li>4: bias (optional, floating) [N]; an empty array stands for an absent bias</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: groupSize (default 128)</li>
 *   <li>1: numBits (1, 2, 4 or 8; default 4)</li>
 * </ul>
 * <p>
 * Output: the input's shape with N in the last axis, in the input's type
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class AwqMatmul extends DynamicCustomOp {

    @Getter private int groupSize = 128;
    @Getter private int numBits = 4;

    /**
     * SameDiff constructor with required inputs (null zeros take the middle of the code range).
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, SDVariable zeros) {
        this(sameDiff, input, weightPacked, scales, zeros, null, 128, 4);
    }

    /**
     * SameDiff constructor with bias.
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, SDVariable zeros, SDVariable bias) {
        this(sameDiff, input, weightPacked, scales, zeros, bias, 128, 4);
    }

    /**
     * Full SameDiff constructor with all options. Zeros and bias may be null.
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, SDVariable zeros, SDVariable bias,
                     int groupSize, int numBits) {
        super(null, sameDiff, inputs(sameDiff, input, weightPacked, scales, zeros, bias), false);
        configure(groupSize, numBits);
    }

    /**
     * SameDiff convenience constructor (no zeros, default numBits).
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, int groupSize) {
        this(sameDiff, input, weightPacked, scales, null, null, groupSize, 4);
    }

    /**
     * INDArray convenience constructor (no zeros, default numBits).
     */
    public AwqMatmul(INDArray input, INDArray weightPacked, INDArray scales, int groupSize) {
        this(input, weightPacked, scales, null, null, null, groupSize, 4);
    }

    /**
     * INDArray constructor. Zeros, bias and output may be null.
     */
    public AwqMatmul(INDArray input, INDArray weightPacked, INDArray scales, INDArray zeros,
                     INDArray bias, INDArray output, int groupSize, int numBits) {
        super(null, inputs(input, weightPacked, scales, zeros, bias),
                output != null ? new INDArray[]{output} : null);
        configure(groupSize, numBits);
    }

    // Absent zeros ahead of a bias take an empty placeholder, which the op reads as absent.
    private static SDVariable[] inputs(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                                       SDVariable scales, SDVariable zeros, SDVariable bias) {
        if (bias == null) {
            return zeros == null ? new SDVariable[]{input, weightPacked, scales} :
                    new SDVariable[]{input, weightPacked, scales, zeros};
        }
        return new SDVariable[]{input, weightPacked, scales,
                zeros == null ? sameDiff.constant(Nd4j.empty(DataType.FLOAT)) : zeros, bias};
    }

    private static INDArray[] inputs(INDArray input, INDArray weightPacked, INDArray scales, INDArray zeros,
                                     INDArray bias) {
        if (bias == null) {
            return zeros == null ? new INDArray[]{input, weightPacked, scales} :
                    new INDArray[]{input, weightPacked, scales, zeros};
        }
        return new INDArray[]{input, weightPacked, scales, zeros == null ? Nd4j.empty(DataType.FLOAT) : zeros, bias};
    }

    private void configure(int groupSize, int numBits) {
        this.groupSize = groupSize;
        this.numBits = numBits;
        addIArgument((long) groupSize, (long) numBits);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.groupSize = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.numBits = iArguments.get(1).intValue();
    }

    @Override
    public String opName() {
        return "awq_matmul";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 3,
                "Expected at least 3 input data types for awq_matmul, got %s", inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
