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
 * Dequantize FP8 E4M3 tensor back to float representation.
 * <p>
 * Performs: output = quantized * scales
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: quantized [numTokens, hiddenDim] INT8 (FP8 E4M3 values)</li>
 *   <li>1: scales [numTokens] or [1] FLOAT32 per-token/per-tensor scales</li>
 * </ul>
 * <p>
 * Output:
 * <ul>
 *   <li>0: dequantized [numTokens, hiddenDim] float/half</li>
 * </ul>
 * <p>
 * Integer args:
 * <ul>
 *   <li>0: outputType (0=FLOAT32, 1=FLOAT16, default: 0)</li>
 * </ul>
 */
@NoArgsConstructor
public class Fp8Dequantize extends DynamicCustomOp {

    public static final int OUTPUT_FLOAT32 = 0;
    public static final int OUTPUT_FLOAT16 = 1;

    @Getter private int outputType = OUTPUT_FLOAT32;

    public Fp8Dequantize(INDArray quantized, INDArray scales) {
        this(quantized, scales, OUTPUT_FLOAT32);
    }

    public Fp8Dequantize(INDArray quantized, INDArray scales, int outputType) {
        super(new INDArray[]{quantized, scales}, null);
        this.outputType = outputType;
        addIArgument((long) outputType);
    }

    public Fp8Dequantize(SameDiff sameDiff, SDVariable quantized, SDVariable scales, int outputType) {
        super(null, sameDiff, new SDVariable[]{quantized, scales}, false);
        this.outputType = outputType;
        addIArgument((long) outputType);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.outputType = iArguments.get(0).intValue();
    }

    @Override
    public String opName() {
        return "fp8_dequantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType outDtype = (outputType == OUTPUT_FLOAT16) ? DataType.HALF : DataType.FLOAT;
        return Collections.singletonList(outDtype);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
