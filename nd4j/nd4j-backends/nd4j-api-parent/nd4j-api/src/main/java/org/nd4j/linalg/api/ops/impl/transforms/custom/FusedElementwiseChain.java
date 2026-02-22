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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Fused element-wise chain operation.
 * <p>
 * Executes a sequence of element-wise ops in a single kernel pass, keeping
 * intermediate values in registers instead of global memory. This replaces
 * N separate kernel launches with 1.
 * <p>
 * Op codes (iArgs):
 * <pre>
 *   0=add, 1=sub, 2=mul, 3=div,
 *   10=relu, 11=sigmoid, 12=tanh, 13=gelu,
 *   14=exp, 15=log, 16=abs, 17=neg,
 *   18=square, 19=sqrt, 20=swish, 21=silu, 22=mish,
 *   30=clip, 31=leaky_relu
 * </pre>
 * <p>
 * Usage:
 * <pre>
 *   // multiply(x, y) -> sigmoid
 *   FusedElementwiseChain.builder()
 *       .input(x)
 *       .multiply(y)
 *       .sigmoid()
 *       .build()
 *       .exec();
 * </pre>
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class FusedElementwiseChain extends DynamicCustomOp {

    // Op codes matching FusedElemOp enum in C++
    public static final int OP_ADD = 0;
    public static final int OP_SUB = 1;
    public static final int OP_MUL = 2;
    public static final int OP_DIV = 3;
    public static final int OP_RELU = 10;
    public static final int OP_SIGMOID = 11;
    public static final int OP_TANH = 12;
    public static final int OP_GELU = 13;
    public static final int OP_EXP = 14;
    public static final int OP_LOG = 15;
    public static final int OP_ABS = 16;
    public static final int OP_NEG = 17;
    public static final int OP_SQUARE = 18;
    public static final int OP_SQRT = 19;
    public static final int OP_SWISH = 20;
    public static final int OP_SILU = 21;
    public static final int OP_MISH = 22;
    public static final int OP_CLIP = 30;
    public static final int OP_LEAKY_RELU = 31;

    /**
     * Create a fused elementwise chain.
     *
     * @param inputs  Primary input at index 0, secondary inputs for binary ops at indices 1+
     * @param output  Pre-allocated output (or null)
     * @param opCodes Sequence of FusedElemOp codes
     */
    public FusedElementwiseChain(INDArray[] inputs, INDArray output, int... opCodes) {
        super(inputs, output != null ? new INDArray[]{output} : null);
        for (int code : opCodes) {
            addIArgument(code);
        }
    }

    /**
     * Constructor for codegen (NDNN): single primary input + optional secondary inputs array.
     */
    public FusedElementwiseChain(INDArray input, INDArray[] secondaryInputs, int[] opCodes) {
        this(buildInputs(input, secondaryInputs), null, opCodes);
    }

    /**
     * Constructor for codegen (SDNN): SameDiff graph mode.
     */
    public FusedElementwiseChain(SameDiff sd, SDVariable input, SDVariable[] secondaryInputs, int[] opCodes) {
        super(null, sd, buildSdInputs(input, secondaryInputs), false);
        for (int code : opCodes) {
            addIArgument(code);
        }
    }

    private static INDArray[] buildInputs(INDArray input, INDArray[] secondaryInputs) {
        if (secondaryInputs == null || secondaryInputs.length == 0) {
            return new INDArray[]{input};
        }
        INDArray[] all = new INDArray[1 + secondaryInputs.length];
        all[0] = input;
        System.arraycopy(secondaryInputs, 0, all, 1, secondaryInputs.length);
        return all;
    }

    private static SDVariable[] buildSdInputs(SDVariable input, SDVariable[] secondaryInputs) {
        if (secondaryInputs == null || secondaryInputs.length == 0) {
            return new SDVariable[]{input};
        }
        SDVariable[] all = new SDVariable[1 + secondaryInputs.length];
        all[0] = input;
        System.arraycopy(secondaryInputs, 0, all, 1, secondaryInputs.length);
        return all;
    }

    @Override
    public String opName() {
        return "fused_elementwise_chain";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    /**
     * Builder for constructing fused chains fluently.
     */
    public static ChainBuilder builder() {
        return new ChainBuilder();
    }

    public static class ChainBuilder {
        private INDArray primaryInput;
        private INDArray output;
        private java.util.List<INDArray> secondaryInputs = new java.util.ArrayList<>();
        private java.util.List<Integer> opCodes = new java.util.ArrayList<>();

        public ChainBuilder input(INDArray input) {
            this.primaryInput = input;
            return this;
        }

        public ChainBuilder output(INDArray output) {
            this.output = output;
            return this;
        }

        // Binary ops (need secondary input)
        public ChainBuilder add(INDArray secondary) { opCodes.add(OP_ADD); secondaryInputs.add(secondary); return this; }
        public ChainBuilder subtract(INDArray secondary) { opCodes.add(OP_SUB); secondaryInputs.add(secondary); return this; }
        public ChainBuilder multiply(INDArray secondary) { opCodes.add(OP_MUL); secondaryInputs.add(secondary); return this; }
        public ChainBuilder divide(INDArray secondary) { opCodes.add(OP_DIV); secondaryInputs.add(secondary); return this; }

        // Unary ops
        public ChainBuilder relu() { opCodes.add(OP_RELU); return this; }
        public ChainBuilder sigmoid() { opCodes.add(OP_SIGMOID); return this; }
        public ChainBuilder tanh() { opCodes.add(OP_TANH); return this; }
        public ChainBuilder gelu() { opCodes.add(OP_GELU); return this; }
        public ChainBuilder exp() { opCodes.add(OP_EXP); return this; }
        public ChainBuilder log() { opCodes.add(OP_LOG); return this; }
        public ChainBuilder abs() { opCodes.add(OP_ABS); return this; }
        public ChainBuilder neg() { opCodes.add(OP_NEG); return this; }
        public ChainBuilder square() { opCodes.add(OP_SQUARE); return this; }
        public ChainBuilder sqrt() { opCodes.add(OP_SQRT); return this; }
        public ChainBuilder swish() { opCodes.add(OP_SWISH); return this; }
        public ChainBuilder silu() { opCodes.add(OP_SILU); return this; }
        public ChainBuilder mish() { opCodes.add(OP_MISH); return this; }

        public ChainBuilder addOp(int opCode) { opCodes.add(opCode); return this; }
        public ChainBuilder addOp(int opCode, INDArray secondary) { opCodes.add(opCode); secondaryInputs.add(secondary); return this; }

        public FusedElementwiseChain build() {
            if (primaryInput == null) {
                throw new IllegalStateException("Primary input is required");
            }
            if (opCodes.isEmpty()) {
                throw new IllegalStateException("At least one op is required");
            }

            // Build inputs array: primary first, then secondaries
            INDArray[] inputs = new INDArray[1 + secondaryInputs.size()];
            inputs[0] = primaryInput;
            for (int i = 0; i < secondaryInputs.size(); i++) {
                inputs[i + 1] = secondaryInputs.get(i);
            }

            int[] codes = opCodes.stream().mapToInt(Integer::intValue).toArray();
            return new FusedElementwiseChain(inputs, output, codes);
        }
    }
}
