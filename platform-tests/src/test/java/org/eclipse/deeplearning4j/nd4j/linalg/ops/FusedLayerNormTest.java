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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedLayerNorm;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * fused_layer_norm and its backprop against a double-precision reference, for rows of every length and layout. CUDA
 * launched 512 or 1024 threads for rows past 256 (beyond the kernel's launch bounds: the launch failed), rounded
 * shorter rows to 96, 160, 192 or 224 threads (the halving merge dropped partial statistics), read every operand as
 * dense rows and the gain and bias as the input's type; CPU took rows one second-to-last stride apart (wrong for any
 * layout whose leading dimensions are not one run) and accumulated doubles in float. The backprop had no CUDA kernel,
 * and with a bias it returned two gradients for three inputs.
 *
 * <p>The shape functions gave each output the cached shape of the input it is shaped like, strides included: the
 * output of a stepped view (every second column of a wider array) then addressed several times the elements its
 * buffer holds (the buffer holds one per element), so the helpers' copy back into it wrote past the end of the
 * buffer. CUDA read back garbage past the allocation; CPU read back what it had written over the heap. The outputs are
 * dense arrays now, for every layout of every operand, and caller-provided outputs of any layout, offset or type are
 * written in place.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class FusedLayerNormTest extends BaseNd4jTestWithBackends {

    private static final float EPSILON = 1e-5f;
    /** exactly representable in every type, so untouched elements compare exactly */
    private static final double SENTINEL = 7.0;

    private static INDArray uniform(DataType type, long seed, double lo, double hi, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.rand(type, shape).muli(hi - lo).addi(lo);
    }

    /** HALF or BFLOAT16 values: FLOAT values rounded to the type, so the double reference sees the same numbers. */
    private static INDArray lowPrecision(DataType type, long seed, double lo, double hi, long... shape) {
        return uniform(DataType.FLOAT, seed, lo, hi, shape).castTo(type);
    }

    /** [rows, cols], every second column of a [rows, 2 * cols] array: strides [2 * cols, 2]. */
    private static INDArray everySecondColumn(DataType type, long seed, double lo, double hi, long rows, long cols) {
        return uniform(type, seed, lo, hi, rows, 2 * cols)
                .get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 2 * cols));
    }

    /** [rows, cols], every second column from the second on, of a [rows, 2 * cols + 1] array: a base offset and a step. */
    private static INDArray offsetEverySecondColumn(DataType type, long seed, double lo, double hi, long rows,
                                                    long cols) {
        return uniform(type, seed, lo, hi, rows, 2 * cols + 1)
                .get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 2 * cols + 1));
    }

    /** [rows, cols], rows [first, first + rows) of a taller array: dense strides and a nonzero base offset. */
    private static INDArray offsetRows(DataType type, long seed, double lo, double hi, long first, long rows,
                                       long cols) {
        return uniform(type, seed, lo, hi, first + rows + 1, cols)
                .get(NDArrayIndex.interval(first, first + rows), NDArrayIndex.all());
    }

    /** [length], every second element of a [2 * length] vector. */
    private static INDArray everySecond(DataType type, long seed, double lo, double hi, long length) {
        return uniform(type, seed, lo, hi, 2 * length).get(NDArrayIndex.interval(0, 2, 2 * length));
    }

    /** The reference: [0] the output, [1] dx, [2] dgain, [3] dbias (the last three only with dy). */
    private static double[][] reference(INDArray x, INDArray gain, INDArray bias, INDArray dy) {
        int rowLen = (int) x.size(x.rank() - 1);
        int rows = (int) (x.length() / rowLen);
        double[] xs = x.dup('c').data().asDouble();
        double[] gs = gain.dup('c').data().asDouble();
        double[] bs = bias == null ? new double[rowLen] : bias.dup('c').data().asDouble();
        double[] dys = dy == null ? null : dy.dup('c').data().asDouble();
        double[] out = new double[xs.length];
        double[] dx = new double[xs.length];
        double[] dgain = new double[rowLen];
        double[] dbias = new double[rowLen];
        for (int r = 0; r < rows; r++) {
            double mean = 0;
            for (int i = 0; i < rowLen; i++) mean += xs[r * rowLen + i];
            mean /= rowLen;
            double variance = 0;
            for (int i = 0; i < rowLen; i++) {
                double centered = xs[r * rowLen + i] - mean;
                variance += centered * centered;
            }
            variance /= rowLen;
            double inverse = 1.0 / Math.sqrt(variance + EPSILON);
            double sumDnorm = 0;
            double sumDnormXhat = 0;
            for (int i = 0; i < rowLen; i++) {
                double xhat = (xs[r * rowLen + i] - mean) * inverse;
                out[r * rowLen + i] = xhat * gs[i] + bs[i];
                if (dys != null) {
                    double dnorm = dys[r * rowLen + i] * gs[i];
                    sumDnorm += dnorm;
                    sumDnormXhat += dnorm * xhat;
                    dgain[i] += dys[r * rowLen + i] * xhat;
                    dbias[i] += dys[r * rowLen + i];
                }
            }
            if (dys != null) {
                for (int i = 0; i < rowLen; i++) {
                    double xhat = (xs[r * rowLen + i] - mean) * inverse;
                    double dnorm = dys[r * rowLen + i] * gs[i];
                    dx[r * rowLen + i] = inverse * (dnorm - sumDnorm / rowLen - xhat * sumDnormXhat / rowLen);
                }
            }
        }
        return new double[][]{out, dx, dgain, dbias};
    }

    private static void assertClose(String label, double[] expected, INDArray actual, double tolerance) {
        double[] values = actual.dup('c').data().asDouble();
        assertEquals(expected.length, values.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], tolerance * Math.max(1.0, Math.abs(expected[i])), label + " at " + i);
        }
    }

    private static double tolerance(DataType type) {
        return type == DataType.DOUBLE ? 1e-10 : 1e-4;
    }

    /** One rounding of the output to the type is the whole error: half a unit in the last place of 11 or 8 bits. */
    private static double lowPrecisionTolerance(DataType type) {
        return type == DataType.FLOAT16 ? 2e-3 : 1e-2;
    }

    private static void assertForward(String label, INDArray x, INDArray gain, INDArray bias, double tolerance) {
        INDArray out = Nd4j.exec(new FusedLayerNorm(x, gain, bias, EPSILON))[0];
        assertEquals(Arrays.toString(x.shape()), Arrays.toString(out.shape()), label + " shape");
        assertEquals(x.dataType(), out.dataType(), label + " type");
        assertClose(label, reference(x, gain, bias, null)[0], out, tolerance);
    }

    private static DynamicCustomOp backwardOp(INDArray x, INDArray gain, INDArray bias, INDArray dy) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("fused_layer_norm_bp")
                .addFloatingPointArguments((double) EPSILON);
        if (bias == null) builder.addInputs(x, gain, dy);
        else builder.addInputs(x, gain, dy, bias);
        return builder.build();
    }

    private static void assertBackward(String label, INDArray x, INDArray gain, INDArray bias, INDArray dy,
                                       double tolerance) {
        INDArray[] grads = Nd4j.exec(backwardOp(x, gain, bias, dy));
        assertEquals(bias == null ? 2 : 3, grads.length, label + " gradients");
        double[][] expected = reference(x, gain, bias, dy);
        assertClose(label + " dx", expected[1], grads[0], tolerance);
        assertClose(label + " dgain", expected[2], grads[1], tolerance);
        if (bias != null) assertClose(label + " dbias", expected[3], grads[2], tolerance);
    }

    /** The output shape the native shape function reports addresses exactly the elements it holds. */
    private static void assertDenseShape(String label, DataBuffer shapeInfo) {
        long[] info = shapeInfo.asLong();
        long[] shape = Shape.shape(info);
        long[] strides = Shape.stride(info);
        long length = 1;
        long farthest = 0;
        for (int i = 0; i < shape.length; i++) {
            length *= shape[i];
            farthest += (shape[i] - 1) * strides[i];
        }
        assertEquals(length - 1, farthest, label + ": shape " + Arrays.toString(shape) + " strides "
                + Arrays.toString(strides) + " must address exactly " + length + " elements");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardForEveryRowLengthAndLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (int rowLen : new int[]{1, 31, 70, 96, 200, 256, 257, 1024, 1500}) {
                assertForward(type + " row length " + rowLen, uniform(type, rowLen, -2, 3, 5, rowLen),
                        uniform(type, rowLen + 1, 0.5, 1.5, rowLen), uniform(type, rowLen + 2, -0.5, 0.5, rowLen),
                        tolerance(type));
            }
            // rows that are not one stride apart: F order, a permuted view, every second column
            INDArray gain = uniform(type, 3, 0.5, 1.5, 70);
            INDArray bias = uniform(type, 4, -0.5, 0.5, 70);
            assertForward(type + " F order", uniform(type, 5, -2, 3, 2, 3, 70).dup('f'), gain, bias, tolerance(type));
            assertForward(type + " permuted view", uniform(type, 6, -2, 3, 70, 3, 2).permute(2, 1, 0), gain, bias,
                    tolerance(type));
            assertForward(type + " every second column",
                    uniform(type, 7, -2, 3, 4, 140).get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 140)), gain,
                    bias, tolerance(type));
            assertForward(type + " no bias", uniform(type, 8, -2, 3, 3, 300), uniform(type, 9, 0.5, 1.5, 300), null,
                    tolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardWithAGainAndBiasOfAnotherType(Nd4jBackend backend) {
        // half weights with float activations (the op takes any float type for each input)
        assertForward("FLOAT input, HALF gain and bias", uniform(DataType.FLOAT, 11, -2, 3, 4, 300),
                uniform(DataType.FLOAT16, 12, 0.5, 1.5, 300), uniform(DataType.FLOAT16, 13, -0.5, 0.5, 300), 1e-4);
        assertForward("DOUBLE input, FLOAT gain and bias", uniform(DataType.DOUBLE, 14, -2, 3, 4, 300),
                uniform(DataType.FLOAT, 15, 0.5, 1.5, 300), uniform(DataType.FLOAT, 16, -0.5, 0.5, 300), 1e-10);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardForEveryRowLengthAndLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (int rowLen : new int[]{1, 31, 70, 257, 1024}) {
                INDArray x = uniform(type, rowLen, -2, 3, 5, rowLen);
                INDArray gain = uniform(type, rowLen + 1, 0.5, 1.5, rowLen);
                INDArray bias = uniform(type, rowLen + 2, -0.5, 0.5, rowLen);
                INDArray dy = uniform(type, rowLen + 3, -1, 1, 5, rowLen);
                assertBackward(type + " row length " + rowLen, x, gain, bias, dy, tolerance(type));
                assertBackward(type + " row length " + rowLen + " without bias", x, gain, null, dy, tolerance(type));
            }
            INDArray gain = uniform(type, 21, 0.5, 1.5, 70);
            INDArray bias = uniform(type, 22, -0.5, 0.5, 70);
            assertBackward(type + " F order", uniform(type, 23, -2, 3, 2, 3, 70).dup('f'), gain, bias,
                    uniform(type, 24, -1, 1, 2, 3, 70).dup('f'), tolerance(type));
            assertBackward(type + " permuted views", uniform(type, 25, -2, 3, 70, 3, 2).permute(2, 1, 0), gain, bias,
                    uniform(type, 26, -1, 1, 70, 3, 2).permute(2, 1, 0), tolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffGradientsReachTheBias(Nd4jBackend backend) {
        INDArray xArr = uniform(DataType.DOUBLE, 31, -2, 3, 4, 70);
        INDArray gainArr = uniform(DataType.DOUBLE, 32, 0.5, 1.5, 70);
        INDArray biasArr = uniform(DataType.DOUBLE, 33, -0.5, 0.5, 70);
        INDArray weights = uniform(DataType.DOUBLE, 34, -1, 1, 4, 70);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);
        SDVariable gain = sd.var("gain", gainArr);
        SDVariable bias = sd.var("bias", biasArr);
        SDVariable out = new FusedLayerNorm(sd, x, gain, bias, EPSILON).outputVariable();
        // the loss sum(out * w) has dLoss/dout = w
        out.mul(sd.constant("w", weights)).sum("loss").markAsLoss();
        Map<String, INDArray> grads = sd.calculateGradients(null, "x", "gain", "bias");

        double[][] expected = reference(xArr, gainArr, biasArr, weights);
        assertClose("SameDiff dx", expected[1], grads.get("x"), 1e-10);
        assertClose("SameDiff dgain", expected[2], grads.get("gain"), 1e-10);
        assertClose("SameDiff dbias", expected[3], grads.get("bias"), 1e-10);
    }

    /**
     * The root cause of every wrong value a stepped view gave: the output array is allocated with one element per
     * element of the shape the shape function returns, so that shape must not keep the input's strides.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void outputShapesAreDenseForEveryOperandLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray gain = uniform(type, 201, 0.5, 1.5, 70);
            INDArray bias = uniform(type, 202, -0.5, 0.5, 70);
            INDArray[] inputs = {
                    uniform(type, 203, -2, 3, 5, 70),
                    uniform(type, 204, -2, 3, 2, 3, 70).dup('f'),
                    uniform(type, 205, -2, 3, 70, 3, 2).permute(2, 1, 0),
                    everySecondColumn(type, 206, -2, 3, 4, 70),
                    offsetRows(type, 207, -2, 3, 2, 3, 70),
                    offsetEverySecondColumn(type, 208, -2, 3, 4, 70),
                    everySecond(type, 209, -2, 3, 70),
                    uniform(type, 210, -2, 3, 2, 3, 4, 140).get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 140)),
            };
            for (INDArray x : inputs) {
                String layout = type + " input " + Arrays.toString(x.shape()) + " strides "
                        + Arrays.toString(x.stride());
                List<DataBuffer> forward = Nd4j.getExecutioner()
                        .calculateOutputShape(new FusedLayerNorm(x, gain, bias, EPSILON));
                assertEquals(1, forward.size(), layout + " forward outputs");
                assertDenseShape(layout + " forward output", forward.get(0));

                INDArray dy = uniform(type, 211, -1, 1, x.shape());
                List<DataBuffer> backward = Nd4j.getExecutioner().calculateOutputShape(backwardOp(x, gain, bias, dy));
                assertEquals(3, backward.size(), layout + " backward outputs");
                for (int i = 0; i < backward.size(); i++) {
                    assertDenseShape(layout + " backward output " + i, backward.get(i));
                }
            }
            // a gain and bias that are stepped views: their gradients are dense too
            INDArray steppedGain = everySecond(type, 212, 0.5, 1.5, 70);
            INDArray steppedBias = everySecond(type, 213, -0.5, 0.5, 70);
            INDArray x = uniform(type, 214, -2, 3, 4, 70);
            List<DataBuffer> backward = Nd4j.getExecutioner()
                    .calculateOutputShape(backwardOp(x, steppedGain, steppedBias, uniform(type, 215, -1, 1, 4, 70)));
            assertEquals(3, backward.size(), type + " stepped gain backward outputs");
            for (int i = 0; i < backward.size(); i++) {
                assertDenseShape(type + " stepped gain and bias, backward output " + i, backward.get(i));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void forwardOnOffsetAndSteppedViews(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = tolerance(type);
            INDArray gain = uniform(type, 221, 0.5, 1.5, 70);
            INDArray bias = uniform(type, 222, -0.5, 0.5, 70);
            assertForward(type + " offset rows", offsetRows(type, 223, -2, 3, 2, 3, 70), gain, bias, tolerance);
            assertForward(type + " offset every second column", offsetEverySecondColumn(type, 224, -2, 3, 4, 70), gain,
                    bias, tolerance);
            assertForward(type + " rank 1 every second element", everySecond(type, 225, -2, 3, 70), gain, bias,
                    tolerance);
            assertForward(type + " rank 4 every second column",
                    uniform(type, 226, -2, 3, 2, 3, 4, 140).get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 140)), gain, bias, tolerance);
            assertForward(type + " every second element of the gain and bias", uniform(type, 227, -2, 3, 5, 70),
                    everySecond(type, 228, 0.5, 1.5, 70), everySecond(type, 229, -0.5, 0.5, 70), tolerance);
            assertForward(type + " every second column, no bias", everySecondColumn(type, 230, -2, 3, 4, 300),
                    uniform(type, 231, 0.5, 1.5, 300), null, tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardOnOffsetAndSteppedViews(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = tolerance(type);
            INDArray gain = uniform(type, 241, 0.5, 1.5, 70);
            INDArray bias = uniform(type, 242, -0.5, 0.5, 70);
            assertBackward(type + " every second column", everySecondColumn(type, 243, -2, 3, 4, 70), gain, bias,
                    everySecondColumn(type, 244, -1, 1, 4, 70), tolerance);
            assertBackward(type + " every second column without bias", everySecondColumn(type, 245, -2, 3, 4, 70), gain,
                    null, everySecondColumn(type, 246, -1, 1, 4, 70), tolerance);
            assertBackward(type + " every second column of x only", everySecondColumn(type, 247, -2, 3, 4, 70), gain,
                    bias, uniform(type, 248, -1, 1, 4, 70), tolerance);
            assertBackward(type + " offset rows", offsetRows(type, 249, -2, 3, 2, 3, 70), gain, bias,
                    offsetRows(type, 250, -1, 1, 1, 3, 70), tolerance);
            assertBackward(type + " offset every second column", offsetEverySecondColumn(type, 251, -2, 3, 4, 70), gain,
                    bias, offsetEverySecondColumn(type, 252, -1, 1, 4, 70), tolerance);
            assertBackward(type + " rank 1 every second element", everySecond(type, 253, -2, 3, 70), gain, bias,
                    everySecond(type, 254, -1, 1, 70), tolerance);
            assertBackward(type + " every second element of the gain and bias", uniform(type, 255, -2, 3, 5, 70),
                    everySecond(type, 256, 0.5, 1.5, 70), everySecond(type, 257, -0.5, 0.5, 70),
                    uniform(type, 258, -1, 1, 5, 70), tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void callerProvidedOutputsAreFilledInPlace(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = tolerance(type);
            INDArray x = uniform(type, 261, -2, 3, 4, 70);
            INDArray gain = uniform(type, 262, 0.5, 1.5, 70);
            INDArray bias = uniform(type, 263, -0.5, 0.5, 70);
            INDArray dy = uniform(type, 264, -1, 1, 4, 70);
            double[] expected = reference(x, gain, bias, null)[0];
            double[][] expectedBackward = reference(x, gain, bias, dy);

            // forward into every second column of a sentinel-filled array: the other columns keep the sentinel
            INDArray stepped = Nd4j.valueArrayOf(new long[]{4, 140}, SENTINEL, type);
            Nd4j.exec(new FusedLayerNorm(x, gain, bias,
                    stepped.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 140)), EPSILON));
            double[] flat = stepped.dup('c').data().asDouble();
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 70; c++) {
                    assertEquals(expected[r * 70 + c], flat[r * 140 + 2 * c],
                            tolerance * Math.max(1.0, Math.abs(expected[r * 70 + c])),
                            type + " stepped output at " + r + ", " + c);
                    assertEquals(SENTINEL, flat[r * 140 + 2 * c + 1], 0.0,
                            type + " stepped output's neighbor at " + r + ", " + c);
                }
            }

            // rows 1 to 3 of x into rows 2 to 4 of a sentinel-filled array (offset views both): the other rows of the
            // array keep the sentinel
            INDArray offset = Nd4j.valueArrayOf(new long[]{6, 70}, SENTINEL, type);
            Nd4j.exec(new FusedLayerNorm(x.get(NDArrayIndex.interval(1, 4), NDArrayIndex.all()), gain, bias,
                    offset.get(NDArrayIndex.interval(2, 5), NDArrayIndex.all()), EPSILON));
            flat = offset.dup('c').data().asDouble();
            for (int i = 0; i < 6 * 70; i++) {
                if (i >= 2 * 70 && i < 5 * 70) {
                    double expectedValue = expected[70 + i - 2 * 70];
                    assertEquals(expectedValue, flat[i], tolerance * Math.max(1.0, Math.abs(expectedValue)),
                            type + " offset output at " + i);
                } else {
                    assertEquals(SENTINEL, flat[i], 0.0, type + " offset output's surroundings at " + i);
                }
            }

            // forward into an array of another float type: the op takes any float type for its output
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 4, 70);
            Nd4j.exec(new FusedLayerNorm(x, gain, bias, otherType, EPSILON));
            assertEquals(other, otherType.dataType(), type + " output type");
            assertClose(type + " output of type " + other, expected, otherType, 1e-4);

            // backward into every second column, every second element and every second element from the second on
            INDArray dxParent = Nd4j.valueArrayOf(new long[]{4, 140}, SENTINEL, type);
            INDArray dgainParent = Nd4j.valueArrayOf(new long[]{140}, SENTINEL, type);
            INDArray dbiasParent = Nd4j.valueArrayOf(new long[]{141}, SENTINEL, type);
            Nd4j.exec(DynamicCustomOp.builder("fused_layer_norm_bp").addFloatingPointArguments((double) EPSILON)
                    .addInputs(x, gain, dy, bias)
                    .addOutputs(dxParent.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 140)),
                            dgainParent.get(NDArrayIndex.interval(0, 2, 140)),
                            dbiasParent.get(NDArrayIndex.interval(1, 2, 141)))
                    .build());
            double[] dx = dxParent.dup('c').data().asDouble();
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 70; c++) {
                    assertEquals(expectedBackward[1][r * 70 + c], dx[r * 140 + 2 * c],
                            tolerance * Math.max(1.0, Math.abs(expectedBackward[1][r * 70 + c])),
                            type + " dx at " + r + ", " + c);
                    assertEquals(SENTINEL, dx[r * 140 + 2 * c + 1], 0.0,
                            type + " dx's neighbor at " + r + ", " + c);
                }
            }
            double[] dgain = dgainParent.dup('c').data().asDouble();
            double[] dbias = dbiasParent.dup('c').data().asDouble();
            for (int c = 0; c < 70; c++) {
                assertEquals(expectedBackward[2][c], dgain[2 * c], tolerance * Math.max(1.0, Math.abs(expectedBackward[2][c])),
                        type + " dgain at " + c);
                assertEquals(SENTINEL, dgain[2 * c + 1], 0.0, type + " dgain's neighbor at " + c);
                assertEquals(expectedBackward[3][c], dbias[2 * c + 1],
                        tolerance * Math.max(1.0, Math.abs(expectedBackward[3][c])), type + " dbias at " + c);
                assertEquals(SENTINEL, dbias[2 * c], 0.0, type + " dbias's neighbor at " + c);
            }
            assertEquals(SENTINEL, dbias[140], 0.0, type + " dbias's last element");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void halfAndBfloat16ActivationsAndWeights(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT16, DataType.BFLOAT16}) {
            double tolerance = lowPrecisionTolerance(type);
            for (int rowLen : new int[]{1, 31, 70, 300}) {
                INDArray x = lowPrecision(type, 271 + rowLen, -2, 3, 5, rowLen);
                INDArray gain = lowPrecision(type, 272 + rowLen, 0.5, 1.5, rowLen);
                INDArray bias = lowPrecision(type, 273 + rowLen, -0.5, 0.5, rowLen);
                INDArray dy = lowPrecision(type, 274 + rowLen, -1, 1, 5, rowLen);
                assertForward(type + " row length " + rowLen, x, gain, bias, tolerance);
                assertForward(type + " row length " + rowLen + " without bias", x, gain, null, tolerance);
                assertBackward(type + " row length " + rowLen, x, gain, bias, dy, tolerance);
                assertBackward(type + " row length " + rowLen + " without bias", x, gain, null, dy, tolerance);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void backwardWithAGainAndBiasOfAnotherType(Nd4jBackend backend) {
        // half weights with float activations: dx is float, the gradients of the gain and bias are half
        INDArray x = uniform(DataType.FLOAT, 281, -2, 3, 4, 300);
        INDArray dy = uniform(DataType.FLOAT, 282, -1, 1, 4, 300);
        INDArray gain = lowPrecision(DataType.FLOAT16, 283, 0.5, 1.5, 300);
        INDArray bias = lowPrecision(DataType.FLOAT16, 284, -0.5, 0.5, 300);
        INDArray[] grads = Nd4j.exec(backwardOp(x, gain, bias, dy));
        assertEquals(3, grads.length, "gradients");
        assertEquals(DataType.FLOAT, grads[0].dataType(), "dx type");
        assertEquals(DataType.FLOAT16, grads[1].dataType(), "dgain type");
        assertEquals(DataType.FLOAT16, grads[2].dataType(), "dbias type");
        double[][] expected = reference(x, gain, bias, dy);
        assertClose("dx", expected[1], grads[0], 1e-4);
        assertClose("dgain", expected[2], grads[1], lowPrecisionTolerance(DataType.FLOAT16));
        assertClose("dbias", expected[3], grads[2], lowPrecisionTolerance(DataType.FLOAT16));
    }
}
