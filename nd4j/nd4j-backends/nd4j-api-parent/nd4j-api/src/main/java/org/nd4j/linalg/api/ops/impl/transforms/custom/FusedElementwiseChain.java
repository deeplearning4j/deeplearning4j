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
 * Every member computes what its eager op computes: the eager op's functor applied in the
 * storage type, with the chain value as the first operand, rounded to the storage type once
 * per member. Input, output and every secondary share one floating storage type; the output
 * has the input's shape. A secondary is a single element or broadcasts right-aligned into
 * the input shape. Chains hold 1..8 members; an unknown code fails.
 * <p>
 * Op codes (iArgs):
 * <pre>
 *   binary (one secondary input each, in chain order):
 *     0=add, 1=sub, 2=mul, 3=div, 31=leaky_relu (secondary = alpha),
 *     50=min, 51=max, 52=mod, 53=atan2, 54=floordiv, 55=reverse_div,
 *     56=reverse_sub, 57=squared_sub, 58=mul_no_nan, 59=pow
 *   unary:
 *     10=relu, 11=sigmoid, 12=tanh, 13=gelu, 14=exp, 15=log, 16=abs, 17=neg,
 *     18=square, 19=sqrt, 20=swish (legacy transform, one rounding),
 *     21=silu (declarable: sigmoid rounded, then multiplied), 22=mish,
 *     23=rsqrt, 24=reciprocal, 25=sign, 26=erf, 27=erfc, 28=log1p, 29=ceil,
 *     30=clip (tArgs [min, max], one pair per chain), 32=floor, 33=round,
 *     34=sin, 35=cos, 36=elu (alpha 1), 37=selu, 38=softplus, 39=softsign,
 *     40=hard_sigmoid, 41=hardtanh, 42=relu6
 * </pre>
 * <p>
 * Usage:
 * <pre>
 *   // multiply(x, y) -&gt; sigmoid
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
    public static final int OP_RSQRT = 23;
    public static final int OP_RECIPROCAL = 24;
    public static final int OP_SIGN = 25;
    public static final int OP_ERF = 26;
    public static final int OP_ERFC = 27;
    public static final int OP_LOG1P = 28;
    public static final int OP_CEIL = 29;
    public static final int OP_CLIP = 30;
    public static final int OP_LEAKY_RELU = 31;
    public static final int OP_FLOOR = 32;
    public static final int OP_ROUND = 33;
    public static final int OP_SIN = 34;
    public static final int OP_COS = 35;
    public static final int OP_ELU = 36;
    public static final int OP_SELU = 37;
    public static final int OP_SOFTPLUS = 38;
    public static final int OP_SOFTSIGN = 39;
    public static final int OP_HARD_SIGMOID = 40;
    public static final int OP_HARDTANH = 41;
    public static final int OP_RELU6 = 42;
    public static final int OP_MIN = 50;
    public static final int OP_MAX = 51;
    public static final int OP_MOD = 52;
    public static final int OP_ATAN2 = 53;
    public static final int OP_FLOORDIV = 54;
    public static final int OP_REVERSE_DIV = 55;
    public static final int OP_REVERSE_SUB = 56;
    public static final int OP_SQUARED_SUB = 57;
    public static final int OP_MUL_NO_NAN = 58;
    public static final int OP_POW = 59;

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
     * Constructor for codegen (NDNN): single primary input, unary ops only.
     */
    public FusedElementwiseChain(INDArray input, int... opCodes) {
        this(new INDArray[]{input}, null, opCodes);
    }

    /**
     * Constructor for codegen (NDNN): single primary input + optional secondary inputs array.
     */
    public FusedElementwiseChain(INDArray input, INDArray[] secondaryInputs, int[] opCodes) {
        this(buildInputs(input, secondaryInputs), null, opCodes);
    }

    /**
     * Constructor for codegen (SDNN): SameDiff graph mode, unary ops only.
     */
    public FusedElementwiseChain(SameDiff sd, SDVariable input, int... opCodes) {
        this(sd, input, (SDVariable[]) null, opCodes);
    }

    /**
     * Constructor for codegen (SDNN): SameDiff graph mode with secondary inputs.
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
        private double[] clipBounds;

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

        /**
         * Clamps to [min, max], like clipbyvalue, which also requires min &lt; max. A chain carries
         * one bounds pair, shared by all its clip members.
         */
        public ChainBuilder clip(double min, double max) {
            if (!(min < max)) {
                throw new IllegalArgumentException("clip needs min < max, got [" + min + ", " + max + "]");
            }
            if (clipBounds != null && (clipBounds[0] != min || clipBounds[1] != max)) {
                throw new IllegalStateException("A chain has one clip bounds pair: [" + clipBounds[0] + ", "
                        + clipBounds[1] + "] is already set, got [" + min + ", " + max + "]");
            }
            clipBounds = new double[]{min, max};
            opCodes.add(OP_CLIP);
            return this;
        }

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
            FusedElementwiseChain op = new FusedElementwiseChain(inputs, output, codes);
            if (clipBounds != null) op.addTArgument(clipBounds);
            return op;
        }
    }
}
