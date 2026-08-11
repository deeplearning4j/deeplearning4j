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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * GGML Dequantize operation.
 *
 * <p>Dequantizes raw GGML quantized bytes to a target floating-point type
 * (FLOAT32, FLOAT16, or BFLOAT16) without intermediate allocations.
 * Uses our own standalone dequantization kernels as baseline, with an
 * optional llamacpp-accelerated platform override when available.</p>
 *
 * <p>GGML quantization types:
 * <ul>
 *   <li>0 = Q4_0</li>
 *   <li>1 = Q4_1</li>
 *   <li>2 = Q5_0</li>
 *   <li>3 = Q5_1</li>
 *   <li>4 = Q8_0</li>
 *   <li>5 = Q8_1</li>
 *   <li>6 = Q2_K</li>
 *   <li>7 = Q3_K</li>
 *   <li>8 = Q4_K</li>
 *   <li>9 = Q5_K</li>
 *   <li>10 = Q6_K</li>
 *   <li>11 = Q8_K</li>
 *   <li>12 = IQ2_XXS</li>
 *   <li>13 = IQ2_XS</li>
 *   <li>14 = IQ3_XXS</li>
 *   <li>15 = IQ1_S</li>
 *   <li>16 = IQ4_NL</li>
 *   <li>17 = IQ3_S</li>
 *   <li>18 = IQ2_S</li>
 *   <li>19 = IQ4_XS</li>
 *   <li>20 = IQ1_M</li>
 *   <li>21 = TQ1_0</li>
 *   <li>22 = TQ2_0</li>
 * </ul>
 * </p>
 *
 * <p>Output data type codes:
 * <ul>
 *   <li>0 = FLOAT32</li>
 *   <li>1 = FLOAT16</li>
 *   <li>2 = BFLOAT16</li>
 * </ul>
 * </p>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class GGMLDequantize extends DynamicCustomOp {

    // GGML quant type constants matching the C++ GgmlQuantType enum
    public static final int QUANT_Q4_0 = 0;
    public static final int QUANT_Q4_1 = 1;
    public static final int QUANT_Q5_0 = 2;
    public static final int QUANT_Q5_1 = 3;
    public static final int QUANT_Q8_0 = 4;
    public static final int QUANT_Q8_1 = 5;
    public static final int QUANT_Q2_K = 6;
    public static final int QUANT_Q3_K = 7;
    public static final int QUANT_Q4_K = 8;
    public static final int QUANT_Q5_K = 9;
    public static final int QUANT_Q6_K = 10;
    public static final int QUANT_Q8_K = 11;
    public static final int QUANT_IQ2_XXS = 12;
    public static final int QUANT_IQ2_XS = 13;
    public static final int QUANT_IQ3_XXS = 14;
    public static final int QUANT_IQ1_S = 15;
    public static final int QUANT_IQ4_NL = 16;
    public static final int QUANT_IQ3_S = 17;
    public static final int QUANT_IQ2_S = 18;
    public static final int QUANT_IQ4_XS = 19;
    public static final int QUANT_IQ1_M = 20;
    public static final int QUANT_TQ1_0 = 21;
    public static final int QUANT_TQ2_0 = 22;

    // Output data type constants matching the C++ GgmlDequantOutputType enum
    public static final int OUTPUT_F32 = 0;
    public static final int OUTPUT_F16 = 1;
    public static final int OUTPUT_BF16 = 2;

    private DataType outputDataType = DataType.FLOAT;

    /**
     * Create a GGML dequantize op for INDArray execution.
     *
     * @param rawBytes      raw quantized bytes as INT8 flat 1D array
     * @param ggmlQuantType GGML quant type (use QUANT_* constants)
     * @param outputShape   desired output shape
     */
    public GGMLDequantize(INDArray rawBytes, int ggmlQuantType, long[] outputShape) {
        this(rawBytes, ggmlQuantType, DataType.FLOAT, outputShape);
    }

    /**
     * Create a GGML dequantize op for INDArray execution with target type.
     *
     * @param rawBytes      raw quantized bytes as INT8 flat 1D array
     * @param ggmlQuantType GGML quant type (use QUANT_* constants)
     * @param targetType    output data type (FLOAT, HALF, or BFLOAT16)
     * @param outputShape   desired output shape
     */
    public GGMLDequantize(INDArray rawBytes, int ggmlQuantType, DataType targetType, long[] outputShape) {
        super(new INDArray[]{rawBytes}, null);
        this.outputDataType = targetType;

        int outputDtypeArg = dataTypeToArg(targetType);

        // iArgs: [quantType, outputDtype, shape...]
        addIArgument(ggmlQuantType);
        addIArgument(outputDtypeArg);
        for (long dim : outputShape) {
            addIArgument(dim);
        }
    }

    @Override
    public String opName() {
        return "ggml_dequantize";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(outputDataType);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    /**
     * Convert DataType to the integer argument expected by the C++ op.
     */
    private static int dataTypeToArg(DataType dt) {
        switch (dt) {
            case FLOAT:
                return OUTPUT_F32;
            case HALF:
                return OUTPUT_F16;
            case BFLOAT16:
                return OUTPUT_BF16;
            default:
                throw new IllegalArgumentException(
                        "ggml_dequantize only supports direct FLOAT, HALF, or BFLOAT16 output, got: " + dt);
        }
    }
}
