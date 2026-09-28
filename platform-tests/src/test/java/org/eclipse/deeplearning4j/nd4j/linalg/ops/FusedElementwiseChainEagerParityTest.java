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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear;
import org.nd4j.linalg.api.ops.impl.scalar.Relu6;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SiLU;
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.same.Ceil;
import org.nd4j.linalg.api.ops.impl.transforms.same.Floor;
import org.nd4j.linalg.api.ops.impl.transforms.same.Negative;
import org.nd4j.linalg.api.ops.impl.transforms.same.Reciprocal;
import org.nd4j.linalg.api.ops.impl.transforms.same.Round;
import org.nd4j.linalg.api.ops.impl.transforms.same.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.same.Square;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Cos;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erf;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Erfc;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.HardTanh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Mish;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SELU;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sin;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftSign;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain.*;

/**
 * The fused element-wise chain's eager contract: every member computes exactly what its eager
 * op computes in the storage type, with the chain value as the eager op's first input. Each
 * member is compared bit for bit with the eager op it stands for, over signed zeros, infinities,
 * NaN, overflow and rounding boundaries, in HALF, BFLOAT16, FLOAT and DOUBLE. Chains are
 * compared with the eager ops run one by one, each writing a materialized intermediate.
 */
public class FusedElementwiseChainEagerParityTest extends BaseNd4jTestWithBackends {

    private static final DataType[] STORAGE = {DataType.HALF, DataType.BFLOAT16, DataType.FLOAT, DataType.DOUBLE};

    /** Written into every output first, so an element a kernel skips cannot pass as a result. */
    private static final double SENTINEL = 7777.0;

    private static final double[] SPECIALS = {-1000, -90, -30, -6.5, -2.5, -1.5, -1, -0.75, -0.5, -0.1, -0.0, 0.0,
            1e-7, 0.1, 0.5, 0.75, 1, 1.5, 2.5, 3.0000001, 6, 6.5, 30, 90, 1000,
            Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN};

    private static final double[] SECONDARY = {3, -2, 0.5, 7, -0.0, 0.0, -1.5, 2.5, 1e-3, -1000, 4, -4, 0.25, -0.75,
            1, -1, 2, 10, -6, 0.1, 1e6, -1e6, 1.5, Double.NaN, 2, 0, Double.POSITIVE_INFINITY, -3};

    private static final int[] UNARY_CODES = {OP_RELU, OP_SIGMOID, OP_TANH, OP_GELU, OP_EXP, OP_LOG, OP_ABS, OP_NEG,
            OP_SQUARE, OP_SQRT, OP_SWISH, OP_SILU, OP_MISH, OP_RSQRT, OP_RECIPROCAL, OP_SIGN, OP_ERF, OP_ERFC,
            OP_LOG1P, OP_CEIL, OP_FLOOR, OP_ROUND, OP_SIN, OP_COS, OP_ELU, OP_SELU, OP_SOFTPLUS, OP_SOFTSIGN,
            OP_HARD_SIGMOID, OP_HARDTANH, OP_RELU6};

    /** Binary members whose eager op is a declarable taking the chain value as input 0. */
    private static final int[] DECLARABLE_BINARY_CODES = {OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_REVERSE_SUB,
            OP_REVERSE_DIV, OP_SQUARED_SUB, OP_MIN, OP_MAX, OP_MOD, OP_ATAN2, OP_FLOORDIV, OP_POW};

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testUnaryMembersMatchEagerOps() {
        for (DataType dtype : STORAGE) {
            INDArray x = array(dtype, unaryValues());
            INDArray original = x.dup();
            for (int code : UNARY_CODES) {
                INDArray expected = eagerUnary(code, x);
                INDArray actual = fused(new INDArray[]{x}, code);
                assertBitwise(dtype + " unary code " + code, expected, actual, x);
            }
            assertBitwise(dtype + " input after the unary chains", original, x);
        }
    }

    @Test
    public void testBinaryMembersMatchEagerOps() {
        for (DataType dtype : STORAGE) {
            INDArray[] pair = crossProduct(dtype);
            INDArray x = pair[0];
            INDArray s = pair[1];
            INDArray originalX = x.dup();
            INDArray originalS = s.dup();
            for (int code : DECLARABLE_BINARY_CODES) {
                INDArray expected = eagerBinary(eagerBinaryName(code), x, s);
                assertBitwise(dtype + " " + eagerBinaryName(code), expected, fused(new INDArray[]{x, s}, code), x, s);
            }
            assertBitwise(dtype + " realdiv", eagerBinary("realdiv", x, s), fused(new INDArray[]{x, s}, OP_DIV), x, s);
            assertBitwise(dtype + " mul_no_nan", eagerMulNoNan(x, s), fused(new INDArray[]{x, s}, OP_MUL_NO_NAN), x, s);
            for (double alpha : new double[]{0.2, 0.01, -0.5, 0.0}) {
                INDArray alphaArray = Nd4j.scalar(dtype, alpha);
                assertBitwise(dtype + " leakyrelu alpha " + alpha, eagerLeakyRelu(x, alphaArray),
                        fused(new INDArray[]{x, alphaArray}, OP_LEAKY_RELU), x);
            }
            assertBitwise(dtype + " chain input after the binary chains", originalX, x);
            assertBitwise(dtype + " secondary after the binary chains", originalS, s);
        }
    }

    /**
     * FusionPass maps a member whose chain value is the eager op's second input to the swapped
     * code. Every swapped code must equal its eager op with the operands exchanged.
     */
    @Test
    public void testSwappedCodesMatchEagerOpsWithOperandsExchanged() {
        for (DataType dtype : STORAGE) {
            INDArray[] pair = crossProduct(dtype);
            INDArray x = pair[0];
            INDArray s = pair[1];
            assertBitwise(dtype + " reverse_sub(x, s) == subtract(s, x)", eagerBinary("subtract", s, x),
                    fused(new INDArray[]{x, s}, OP_REVERSE_SUB), x, s);
            assertBitwise(dtype + " reverse_div(x, s) == divide(s, x)", eagerBinary("divide", s, x),
                    fused(new INDArray[]{x, s}, OP_REVERSE_DIV), x, s);
            assertBitwise(dtype + " add(x, s) == add(s, x)", eagerBinary("add", s, x),
                    fused(new INDArray[]{x, s}, OP_ADD), x, s);
            assertBitwise(dtype + " mul(x, s) == multiply(s, x)", eagerBinary("multiply", s, x),
                    fused(new INDArray[]{x, s}, OP_MUL), x, s);
            assertBitwise(dtype + " squared_sub(x, s) == squaredsubtract(s, x)", eagerBinary("squaredsubtract", s, x),
                    fused(new INDArray[]{x, s}, OP_SQUARED_SUB), x, s);
        }
    }

    @Test
    public void testChainsRoundEveryIntermediateLikeEagerOps() {
        for (DataType dtype : STORAGE) {
            double[] values = unaryValues();
            INDArray x = array(dtype, values);
            INDArray a = array(dtype, ramp(values.length, -1.75, 0.03125));
            INDArray c = array(dtype, ramp(values.length, 2.5, -0.0234375));
            INDArray bias = Nd4j.scalar(dtype, 0.375);
            INDArray cap = Nd4j.scalar(dtype, 30.0);

            assertChain(dtype + " mul-add-sigmoid-mul", x, new int[]{OP_MUL, OP_ADD, OP_SIGMOID, OP_MUL},
                    new INDArray[]{a, bias, c}, null);
            assertChain(dtype + " softcap", x, new int[]{OP_DIV, OP_TANH, OP_MUL}, new INDArray[]{cap, cap}, null);
            assertChain(dtype + " eight members", x,
                    new int[]{OP_SUB, OP_ABS, OP_SQRT, OP_MUL, OP_SQUARED_SUB, OP_LOG1P, OP_MAX, OP_TANH},
                    new INDArray[]{a, Nd4j.scalar(dtype, 1.5), c, Nd4j.scalar(dtype, 0.25)}, null);
            assertChain(dtype + " eight unary members", x,
                    new int[]{OP_SILU, OP_SWISH, OP_MISH, OP_GELU, OP_ELU, OP_SELU, OP_SOFTPLUS, OP_HARD_SIGMOID},
                    new INDArray[0], null);
            assertChain(dtype + " reverse members", x, new int[]{OP_REVERSE_SUB, OP_EXP, OP_REVERSE_DIV, OP_MIN},
                    new INDArray[]{a, c, Nd4j.scalar(dtype, 0.5)}, null);

            // The builder path, with the chain's clip bounds pair.
            INDArray expected = eagerChain(x, new int[]{OP_MUL, OP_CLIP, OP_ADD}, new INDArray[]{a, bias},
                    new double[]{-0.3, 0.7});
            INDArray actual = sentinel(dtype, x.shape());
            Nd4j.exec(FusedElementwiseChain.builder().input(x).multiply(a).clip(-0.3, 0.7).add(bias)
                    .output(actual).build());
            assertBitwise(dtype + " builder mul-clip-add", expected, actual, x);
        }
    }

    @Test
    public void testViewsAndAliasedOutputs() {
        int[] codes = {OP_MUL, OP_TANH, OP_SUB};
        for (DataType dtype : STORAGE) {
            INDArray base = array(dtype, ramp(30, -3.5, 0.25)).reshape(3, 10);
            // Columns 1, 3, 5, 7: a stepped view with an offset.
            INDArray x = base.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 9));
            // A transposed view: F-order strides.
            INDArray s = array(dtype, ramp(12, 1.25, -0.1875)).reshape(4, 3).transpose();
            INDArray row = array(dtype, 0.5, -1.0, 2.0, -0.0);
            INDArray expected = eagerChain(x.dup('c'), codes, new INDArray[]{s.dup('c'), row}, null);

            assertBitwise(dtype + " stepped input, transposed secondary", expected,
                    fused(new INDArray[]{x, s, row}, codes), x, s);

            // Stepped output: the columns between the written ones keep the sentinel.
            INDArray wide = sentinel(dtype, 3, 8);
            INDArray outView = wide.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 8));
            Nd4j.exec(new FusedElementwiseChain(new INDArray[]{x, s, row}, outView, codes));
            assertBitwise(dtype + " stepped output", expected, outView, x, s);
            assertBitwise(dtype + " columns between the stepped output", sentinel(dtype, 3, 4),
                    wide.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 8)));

            // In place on the stepped view: only its own elements change.
            INDArray inPlaceBase = base.dup();
            INDArray inPlace = inPlaceBase.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 9));
            Nd4j.exec(new FusedElementwiseChain(new INDArray[]{inPlace, s, row}, inPlace, codes));
            assertBitwise(dtype + " in place", expected, inPlace);
            assertBitwise(dtype + " columns beside the in-place view",
                    base.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 10)),
                    inPlaceBase.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 10)));
            assertBitwise(dtype + " last column beside the in-place view",
                    base.get(NDArrayIndex.all(), NDArrayIndex.point(9)),
                    inPlaceBase.get(NDArrayIndex.all(), NDArrayIndex.point(9)));

            // The output is the same-shape secondary.
            INDArray aliased = s.dup('c');
            Nd4j.exec(new FusedElementwiseChain(new INDArray[]{x, aliased, row}, aliased, codes));
            assertBitwise(dtype + " output aliasing the secondary", expected, aliased);

            // Permuted input and secondary.
            INDArray p = array(dtype, ramp(60, -2.0, 0.0703125)).reshape(4, 5, 3).permute(2, 0, 1);
            INDArray q = array(dtype, ramp(60, 3.0, -0.09375)).reshape(5, 3, 4).permute(1, 2, 0);
            INDArray permutedExpected = eagerChain(p.dup('c'), codes, new INDArray[]{q.dup('c'), Nd4j.scalar(dtype, 0.75)},
                    null);
            assertBitwise(dtype + " permuted input and secondary", permutedExpected,
                    fused(new INDArray[]{p, q, Nd4j.scalar(dtype, 0.75)}, codes), p, q);
        }
    }

    @Test
    public void testBroadcastSecondaries() {
        long[][] shapes = {{5}, {4, 1}, {1, 5}, {3, 1, 1}, {1}, {}};
        int[] codes = {OP_SUB, OP_REVERSE_DIV, OP_TANH};
        for (DataType dtype : STORAGE) {
            INDArray x = array(dtype, ramp(60, -4.0, 0.140625)).reshape(3, 4, 5);
            for (int i = 0; i < shapes.length; i++) {
                INDArray s = shaped(dtype, shapes[i], 0.625, -0.3125);
                INDArray s2 = shaped(dtype, shapes[(i + 1) % shapes.length], -1.5, 0.4375);
                assertChain(dtype + " secondaries " + java.util.Arrays.toString(shapes[i]) + " and "
                        + java.util.Arrays.toString(shapes[(i + 1) % shapes.length]), x, codes,
                        new INDArray[]{s, s2}, null);
            }
        }
    }

    @Test
    public void testClipMatchesClipByValue() {
        double[][] bounds = {{-0.3, 0.7}, {-2.5, 6.0}, {1e-7, 0.5}, {-1e6, 1e6}};
        for (DataType dtype : STORAGE) {
            INDArray x = array(dtype, unaryValues());
            for (double[] b : bounds) {
                INDArray actual = sentinel(dtype, x.shape());
                Nd4j.exec(FusedElementwiseChain.builder().input(x).clip(b[0], b[1]).output(actual).build());
                assertBitwise(dtype + " clip [" + b[0] + ", " + b[1] + "]", eagerClip(x, b[0], b[1]), actual, x);
            }
        }
    }

    @Test
    public void testInvalidChainsFail() {
        INDArray x = array(DataType.FLOAT, 1.0, -2.0, 3.0, -4.0);
        INDArray out = sentinel(DataType.FLOAT, 4);

        int[] nine = new int[9];
        java.util.Arrays.fill(nine, OP_ABS);
        assertExecFails("nine members", new FusedElementwiseChain(new INDArray[]{x}, out, nine));
        assertExecFails("no members", new FusedElementwiseChain(new INDArray[]{x}, out));
        assertExecFails("unimplemented code 4", new FusedElementwiseChain(new INDArray[]{x}, out, 4));
        assertExecFails("unimplemented code 43", new FusedElementwiseChain(new INDArray[]{x}, out, 43));
        assertExecFails("binary member without a secondary", new FusedElementwiseChain(new INDArray[]{x}, out, OP_ADD));
        assertExecFails("secondary without a binary member",
                new FusedElementwiseChain(new INDArray[]{x, x.dup()}, out, OP_ABS));
        assertExecFails("DOUBLE secondary on a FLOAT chain", new FusedElementwiseChain(
                new INDArray[]{x, array(DataType.DOUBLE, 1, 2, 3, 4)}, out, OP_ADD));
        assertExecFails("secondary that does not broadcast", new FusedElementwiseChain(
                new INDArray[]{array(DataType.FLOAT, 1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 4), array(DataType.FLOAT, 1, 2, 3)},
                sentinel(DataType.FLOAT, 2, 4), OP_ADD));
        assertExecFails("secondary of higher rank", new FusedElementwiseChain(
                new INDArray[]{x, array(DataType.FLOAT, 1, 2, 3, 4).reshape(1, 4)}, out, OP_ADD));
        assertExecFails("empty secondary on a non-empty chain", new FusedElementwiseChain(
                new INDArray[]{array(DataType.FLOAT, 1, 2, 3, 4, 5, 6).reshape(2, 3), Nd4j.create(DataType.FLOAT, 0, 3)},
                sentinel(DataType.FLOAT, 2, 3), OP_ADD));
        assertExecFails("INT32 chain", new FusedElementwiseChain(new INDArray[]{Nd4j.createFromArray(1, -2, 3, -4)},
                Nd4j.create(DataType.INT32, 4), OP_ABS));
        assertExecFails("HALF output on a FLOAT chain",
                new FusedElementwiseChain(new INDArray[]{x}, Nd4j.create(DataType.HALF, 4), OP_ABS));
        assertExecFails("output of another shape",
                new FusedElementwiseChain(new INDArray[]{x}, sentinel(DataType.FLOAT, 2, 2), OP_ABS));
        assertExecFails("clip without bounds", new FusedElementwiseChain(new INDArray[]{x}, out, OP_CLIP));
        for (double[] bounds : new double[][]{{0.7, -0.3}, {0.5, 0.5}, {Double.NaN, 1.0}, {-1.0, Double.NaN}}) {
            FusedElementwiseChain op = new FusedElementwiseChain(new INDArray[]{x}, out, OP_CLIP);
            op.addTArgument(bounds);
            assertExecFails("clip bounds " + java.util.Arrays.toString(bounds) + " (clipbyvalue rejects them)", op);
        }
        assertBitwise("output after the failed chains", sentinel(DataType.FLOAT, 4), out);

        assertThrows(IllegalStateException.class,
                () -> FusedElementwiseChain.builder().input(x).clip(-1, 1).clip(-2, 2));
        assertThrows(IllegalArgumentException.class, () -> FusedElementwiseChain.builder().input(x).clip(1, -1));
        assertThrows(IllegalArgumentException.class, () -> FusedElementwiseChain.builder().input(x).clip(0.5, 0.5));
        // The same pair twice is one bounds pair.
        INDArray twice = sentinel(DataType.FLOAT, 4);
        Nd4j.exec(FusedElementwiseChain.builder().input(x).clip(-2.5, 2.5).abs().clip(-2.5, 2.5).output(twice).build());
        assertBitwise("clip, abs, clip", array(DataType.FLOAT, 1, 2, 2.5, 2.5), twice);
    }

    @Test
    public void testEmptyChain() {
        for (DataType dtype : STORAGE) {
            INDArray x = Nd4j.create(dtype, 0, 3);
            INDArray s = array(dtype, 1, 2, 3);
            INDArray out = Nd4j.create(dtype, 0, 3);
            Nd4j.exec(new FusedElementwiseChain(new INDArray[]{x, s}, out, OP_ADD, OP_TANH));
            assertTrue(out.isEmpty(), dtype + " pre-allocated empty output");
            assertArrayEquals(new long[]{0, 3}, out.shape());

            INDArray[] allocated = Nd4j.exec(new FusedElementwiseChain(new INDArray[]{x, s}, null, OP_ADD, OP_TANH));
            assertEquals(1, allocated.length);
            assertTrue(allocated[0].isEmpty(), dtype + " allocated empty output");
            assertEquals(dtype, allocated[0].dataType());
            assertArrayEquals(new long[]{0, 3}, allocated[0].shape());
        }
    }

    @Test
    public void testAllocatedOutputOfSteppedInput() {
        for (DataType dtype : STORAGE) {
            INDArray base = array(dtype, ramp(30, -3.5, 0.25)).reshape(3, 10);
            INDArray x = base.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 9));
            INDArray s = Nd4j.scalar(dtype, -0.625);
            INDArray[] allocated = Nd4j.exec(new FusedElementwiseChain(new INDArray[]{x, s}, null, OP_MUL, OP_SIGMOID));
            assertBitwise(dtype + " allocated output of a stepped input",
                    eagerChain(x.dup('c'), new int[]{OP_MUL, OP_SIGMOID}, new INDArray[]{s}, null), allocated[0], x);
        }
    }

    // ---- eager references ----

    private static void assertChain(String context, INDArray x, int[] codes, INDArray[] secondaries, double[] clip) {
        INDArray original = x.dup();
        List<INDArray> originalSecondaries = new ArrayList<>();
        for (INDArray s : secondaries) originalSecondaries.add(s.dup());
        INDArray expected = eagerChain(x, codes, secondaries, clip);
        INDArray[] inputs = new INDArray[1 + secondaries.length];
        inputs[0] = x;
        System.arraycopy(secondaries, 0, inputs, 1, secondaries.length);
        INDArray actual = sentinel(x.dataType(), x.shape());
        FusedElementwiseChain op = new FusedElementwiseChain(inputs, actual, codes);
        if (clip != null) op.addTArgument(clip);
        Nd4j.exec(op);
        assertBitwise(context, expected, actual, x);
        assertBitwise(context + ": chain input unchanged", original, x);
        for (int i = 0; i < secondaries.length; i++)
            assertBitwise(context + ": secondary " + i + " unchanged", originalSecondaries.get(i), secondaries[i]);
    }

    /** The eager ops, one by one, each into a freshly allocated intermediate. */
    private static INDArray eagerChain(INDArray x, int[] codes, INDArray[] secondaries, double[] clip) {
        INDArray v = x;
        int next = 0;
        for (int code : codes) {
            if (isBinary(code)) v = eagerBinaryMember(code, v, secondaries[next++]);
            else if (code == OP_CLIP) v = eagerClip(v, clip[0], clip[1]);
            else v = eagerUnary(code, v);
        }
        assertEquals(secondaries.length, next, "one secondary per binary member");
        return v;
    }

    private static boolean isBinary(int code) {
        return code <= OP_DIV || code == OP_LEAKY_RELU || (code >= OP_MIN && code <= OP_POW);
    }

    private static INDArray eagerBinaryMember(int code, INDArray v, INDArray s) {
        if (code == OP_MUL_NO_NAN) return eagerMulNoNan(v, s);
        if (code == OP_LEAKY_RELU) return eagerLeakyRelu(v, s);
        return eagerBinary(eagerBinaryName(code), v, s);
    }

    private static String eagerBinaryName(int code) {
        switch (code) {
            case OP_ADD: return "add";
            case OP_SUB: return "subtract";
            case OP_MUL: return "multiply";
            case OP_DIV: return "divide";
            case OP_REVERSE_SUB: return "reversesubtract";
            case OP_REVERSE_DIV: return "reversedivide";
            case OP_SQUARED_SUB: return "squaredsubtract";
            case OP_MIN: return "minimum";
            case OP_MAX: return "maximum";
            case OP_MOD: return "mod";
            case OP_ATAN2: return "tf_atan2";
            case OP_FLOORDIV: return "floordiv";
            case OP_POW: return "Pow";
            default: throw new IllegalArgumentException("no eager declarable for code " + code);
        }
    }

    private static INDArray eagerBinary(String opName, INDArray x, INDArray y) {
        INDArray z = sentinel(x.dataType(), x.shape());
        Nd4j.exec(DynamicCustomOp.builder(opName).addInputs(x, y).addOutputs(z).build());
        return z;
    }

    /** multiply, then +0 wherever the multiplier is zero. */
    private static INDArray eagerMulNoNan(INDArray x, INDArray s) {
        INDArray z = eagerBinary("multiply", x, s);
        double[] multiplier = s.dup('c').data().asDouble();
        for (int i = 0; i < multiplier.length; i++) {
            if (multiplier[i] == 0.0) z.putScalar(i, 0.0);
        }
        return z;
    }

    private static INDArray eagerLeakyRelu(INDArray x, INDArray alpha) {
        assertEquals(1, alpha.length(), "the eager leakyrelu has one alpha");
        INDArray z = sentinel(x.dataType(), x.shape());
        Nd4j.exec(new LeakyReLU(x, z, alpha.getDouble(0)));
        return z;
    }

    private static INDArray eagerClip(INDArray x, double min, double max) {
        INDArray z = sentinel(x.dataType(), x.shape());
        Nd4j.exec(DynamicCustomOp.builder("clipbyvalue").addInputs(x).addOutputs(z)
                .addFloatingPointArguments(min, max).build());
        return z;
    }

    private static INDArray eagerUnary(int code, INDArray x) {
        INDArray z = sentinel(x.dataType(), x.shape());
        switch (code) {
            case OP_RELU: Nd4j.exec(new RectifiedLinear(x, z, 0.0)); break;
            case OP_RELU6: Nd4j.exec(new Relu6(x, z, 0.0)); break;
            case OP_ELU: Nd4j.exec(new ELU(x, z)); break;
            case OP_SILU: Nd4j.exec(new SiLU(x, z)); break;
            case OP_SIGMOID: Nd4j.exec(new Sigmoid(x, z)); break;
            case OP_TANH: Nd4j.exec(new Tanh(x, z)); break;
            case OP_GELU: Nd4j.exec(new GELU(x, z)); break;
            case OP_EXP: Nd4j.exec(new Exp(x, z)); break;
            case OP_LOG: Nd4j.exec(new Log(x, z)); break;
            case OP_ABS: Nd4j.exec(new Abs(x, z)); break;
            case OP_NEG: Nd4j.exec(new Negative(x, z)); break;
            case OP_SQUARE: Nd4j.exec(new Square(x, z)); break;
            case OP_SQRT: Nd4j.exec(new Sqrt(x, z)); break;
            case OP_SWISH: Nd4j.exec(new Swish(x, z)); break;
            case OP_MISH: Nd4j.exec(new Mish(x, z)); break;
            case OP_RSQRT: Nd4j.exec(new RSqrt(x, z)); break;
            case OP_RECIPROCAL: Nd4j.exec(new Reciprocal(x, z)); break;
            case OP_SIGN: Nd4j.exec(new Sign(x, z)); break;
            case OP_ERF: Nd4j.exec(new Erf(x, z)); break;
            case OP_ERFC: Nd4j.exec(new Erfc(x, z)); break;
            case OP_LOG1P: Nd4j.exec(new Log1p(x, z)); break;
            case OP_CEIL: Nd4j.exec(new Ceil(x, z)); break;
            case OP_FLOOR: Nd4j.exec(new Floor(x, z)); break;
            case OP_ROUND: Nd4j.exec(new Round(x, z)); break;
            case OP_SIN: Nd4j.exec(new Sin(x, z)); break;
            case OP_COS: Nd4j.exec(new Cos(x, z)); break;
            case OP_SELU: Nd4j.exec(new SELU(x, z)); break;
            case OP_SOFTPLUS: Nd4j.exec(new SoftPlus(x, z)); break;
            case OP_SOFTSIGN: Nd4j.exec(new SoftSign(x, z)); break;
            case OP_HARD_SIGMOID: Nd4j.exec(new HardSigmoid(x, z)); break;
            case OP_HARDTANH: Nd4j.exec(new HardTanh(x, z)); break;
            default: throw new IllegalArgumentException("no eager op for code " + code);
        }
        return z;
    }

    // ---- inputs ----

    private static INDArray fused(INDArray[] inputs, int... codes) {
        INDArray out = sentinel(inputs[0].dataType(), inputs[0].shape());
        Nd4j.exec(new FusedElementwiseChain(inputs, out, codes));
        return out;
    }

    private static INDArray array(DataType dtype, double... values) {
        return Nd4j.createFromArray(values).castTo(dtype);
    }

    private static INDArray sentinel(DataType dtype, long... shape) {
        return Nd4j.valueArrayOf(shape, SENTINEL, dtype);
    }

    private static INDArray shaped(DataType dtype, long[] shape, double start, double step) {
        if (shape.length == 0) return Nd4j.scalar(dtype, start);
        long length = 1;
        for (long d : shape) length *= d;
        return array(dtype, ramp((int) length, start, step)).reshape(shape);
    }

    private static double[] ramp(int length, double start, double step) {
        double[] values = new double[length];
        for (int i = 0; i < length; i++) values[i] = start + i * step;
        return values;
    }

    /** The specials, then -12..12 in steps of 3/32. */
    private static double[] unaryValues() {
        double[] values = new double[SPECIALS.length + 257];
        System.arraycopy(SPECIALS, 0, values, 0, SPECIALS.length);
        for (int k = 0; k <= 256; k++) values[SPECIALS.length + k] = -12.0 + k * 3.0 / 32.0;
        return values;
    }

    /** Every special against every secondary value, element by element. */
    private static INDArray[] crossProduct(DataType dtype) {
        int n = SPECIALS.length * SECONDARY.length;
        double[] chain = new double[n];
        double[] secondary = new double[n];
        for (int i = 0; i < SPECIALS.length; i++) {
            for (int j = 0; j < SECONDARY.length; j++) {
                chain[i * SECONDARY.length + j] = SPECIALS[i];
                secondary[i * SECONDARY.length + j] = SECONDARY[j];
            }
        }
        return new INDArray[]{array(dtype, chain), array(dtype, secondary)};
    }

    // ---- assertions ----

    private static void assertExecFails(String context, FusedElementwiseChain op) {
        assertThrows(RuntimeException.class, () -> Nd4j.exec(op), context + " must fail");
    }

    /**
     * Bit-for-bit equality; every NaN matches every NaN. Operands the same length as the result
     * are printed beside each mismatch.
     */
    private static void assertBitwise(String context, INDArray expected, INDArray actual, INDArray... operands) {
        assertEquals(expected.dataType(), actual.dataType(), context + ": dtype");
        assertArrayEquals(expected.shape(), actual.shape(), context + ": shape");
        double[] e = expected.dup('c').data().asDouble();
        double[] a = actual.dup('c').data().asDouble();
        List<double[]> printed = new ArrayList<>();
        for (INDArray operand : operands) {
            if (operand.length() == expected.length()) printed.add(operand.dup('c').data().asDouble());
        }
        boolean isDouble = expected.dataType() == DataType.DOUBLE;
        StringBuilder mismatches = new StringBuilder();
        int count = 0;
        for (int i = 0; i < e.length; i++) {
            boolean same = isDouble
                    ? Double.doubleToLongBits(e[i]) == Double.doubleToLongBits(a[i])
                    : Float.floatToIntBits((float) e[i]) == Float.floatToIntBits((float) a[i]);
            if (same) continue;
            if (count++ < 8) {
                mismatches.append("\n  [").append(i).append("]");
                for (double[] operand : printed) mismatches.append(" operand=").append(operand[i]);
                mismatches.append(" expected=").append(e[i]).append(" actual=").append(a[i]);
            }
        }
        if (count > 0) fail(context + ": " + count + " of " + e.length + " elements differ" + mismatches);
    }
}
