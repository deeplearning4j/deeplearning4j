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

import java.util.Arrays;
import java.util.List;

/**
 * Quantize a float tensor to FP8 E4M3 representation with scale factors.
 * <p>
 * Supports three quantization modes:
 * <ul>
 *   <li>Per-tensor (mode=0): single scale for the entire tensor</li>
 *   <li>Per-token (mode=1): one scale per row (default)</li>
 *   <li>Per-group (mode=2): one scale per group of {@code groupSize} elements</li>
 * </ul>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [numTokens, hiddenDim] float/half/bf16</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: quantized [numTokens, hiddenDim] INT8 (FP8 E4M3 values)</li>
 *   <li>1: scales - [numTokens] for per-token, [numTokens, numGroups] for per-group</li>
 * </ul>
 */
@NoArgsConstructor
public class Fp8Quantize extends DynamicCustomOp {

    public static final int MODE_PER_TENSOR = 0;
    public static final int MODE_PER_TOKEN = 1;
    public static final int MODE_PER_GROUP = 2;

    public static final int FORMAT_E4M3 = 0;
    public static final int FORMAT_E5M2 = 1;

    @Getter private int mode = MODE_PER_TOKEN;
    @Getter private int groupSize = 128;
    @Getter private int fp8Format = FORMAT_E4M3;

    public Fp8Quantize(INDArray input) {
        this(input, MODE_PER_TOKEN, 128, FORMAT_E4M3);
    }

    public Fp8Quantize(INDArray input, int mode, int groupSize, int fp8Format) {
        super(new INDArray[]{input}, null);
        this.mode = mode;
        this.groupSize = groupSize;
        this.fp8Format = fp8Format;
        addIArgument((long) mode, (long) groupSize, (long) fp8Format);
    }

    public Fp8Quantize(SameDiff sameDiff, SDVariable input, int mode, int groupSize, int fp8Format) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.mode = mode;
        this.groupSize = groupSize;
        this.fp8Format = fp8Format;
        addIArgument((long) mode, (long) groupSize, (long) fp8Format);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.mode = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.groupSize = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.fp8Format = iArguments.get(2).intValue();
    }

    @Override
    public String opName() {
        return "fp8_quantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(DataType.INT8, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
