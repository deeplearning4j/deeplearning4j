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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * AWQ (Activation-aware Weight Quantization) matrix multiplication.
 * <p>
 * Performs dequantization and GEMM with AWQ-packed weights:
 * <pre>
 *   output = input @ dequant(weightPacked, scales, zeros) + bias
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [M, K] - activation tensor</li>
 *   <li>1: weightPacked [K/2, N] INT8 - packed quantized weights</li>
 *   <li>2: scales [K/g, N] - per-group dequantization scales</li>
 *   <li>3: zeros [K/g, N] - per-group zero points</li>
 *   <li>4: bias [N] (optional)</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: groupSize (default 128)</li>
 *   <li>1: numBits (default 4)</li>
 * </ul>
 * <p>
 * Output: [M, N] same dtype as input
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class AwqMatmul extends DynamicCustomOp {

    @Getter private int groupSize = 128;
    @Getter private int numBits = 4;

    /**
     * SameDiff constructor with required inputs.
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, SDVariable zeros) {
        super(null, sameDiff, new SDVariable[]{input, weightPacked, scales, zeros}, false);
        addIArgument((long) groupSize, (long) numBits);
    }

    /**
     * SameDiff constructor with bias.
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, SDVariable zeros, SDVariable bias) {
        super(null, sameDiff, new SDVariable[]{input, weightPacked, scales, zeros, bias}, false);
        addIArgument((long) groupSize, (long) numBits);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, SDVariable zeros, SDVariable bias,
                     int groupSize, int numBits) {
        super(null, sameDiff, bias != null ?
                new SDVariable[]{input, weightPacked, scales, zeros, bias} :
                new SDVariable[]{input, weightPacked, scales, zeros}, false);
        this.groupSize = groupSize;
        this.numBits = numBits;
        addIArgument((long) groupSize, (long) numBits);
    }

    /**
     * SameDiff convenience constructor (no zeros, default numBits).
     */
    public AwqMatmul(SameDiff sameDiff, SDVariable input, SDVariable weightPacked,
                     SDVariable scales, int groupSize) {
        super(null, sameDiff, new SDVariable[]{input, weightPacked, scales}, false);
        this.groupSize = groupSize;
        addIArgument((long) groupSize, (long) numBits);
    }

    /**
     * INDArray convenience constructor (no zeros, default numBits).
     */
    public AwqMatmul(INDArray input, INDArray weightPacked, INDArray scales, int groupSize) {
        super(null, new INDArray[]{input, weightPacked, scales}, null);
        this.groupSize = groupSize;
        addIArgument((long) groupSize, (long) numBits);
    }

    /**
     * INDArray constructor.
     */
    public AwqMatmul(INDArray input, INDArray weightPacked, INDArray scales, INDArray zeros,
                     INDArray bias, INDArray output, int groupSize, int numBits) {
        super(null, bias != null ?
                new INDArray[]{input, weightPacked, scales, zeros, bias} :
                new INDArray[]{input, weightPacked, scales, zeros},
                output != null ? new INDArray[]{output} : null);
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
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
