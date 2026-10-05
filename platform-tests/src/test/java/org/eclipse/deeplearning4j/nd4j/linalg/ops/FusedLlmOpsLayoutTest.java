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
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedBiasDropoutResidual;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGELU;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGELUBp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedMRoPE;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * fused_rope (dynamic and cached, and its backprop), fused_bias_dropout_residual, fused_mrope, fused_gelu (and its
 * backprop), fused_attention_projection and vision_embedding_merge against a double-precision reference, for every
 * operand layout and float type. The kernels
 * indexed their operands as dense row-major arrays (CUDA), or read the output through the strides of the input's shape
 * (CPU), while the shape functions gave each output the cached shape of its first input, strides included: any stepped
 * view, F-ordered or permuted array, or output of another type than the input's gave wrong values, and the output of a
 * stepped view addressed several times the elements its buffer holds. The outputs are dense arrays now, every operand is
 * read by its logical coordinates, and caller-provided outputs of any layout, offset or type are written in place.
 *
 * <p>The cached rope also read the sin table through the strides of the cos table, the CUDA kernel for it had no
 * BFLOAT16 pairs, and the sizes of the tables were never checked against the input's. The GELU paired the elements of
 * its input and output by their offsets from the start of their buffers (the CPU) or read the input flat (CUDA), and
 * computed doubles in float on the CPU; the attention projection wrote its product into a copy when the output was not
 * a dense array, and added a bias of another type to the product as if it were of the output's.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class FusedLlmOpsLayoutTest extends BaseNd4jTestWithBackends {

    /** exactly representable in every type, so untouched elements compare exactly */
    private static final double SENTINEL = 7.0;
    private static final DataType[] FLOAT_TYPES = {DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16, DataType.BFLOAT16};

    /** Values of the given type, uniform in [lo, hi): generated in FLOAT (DOUBLE for DOUBLE) and rounded to the type. */
    private static INDArray random(DataType type, long seed, double lo, double hi, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        DataType generated = type == DataType.DOUBLE ? DataType.DOUBLE : DataType.FLOAT;
        INDArray values = Nd4j.rand(generated, shape).muli(hi - lo).addi(lo);
        return values.dataType() == type ? values : values.castTo(type);
    }

    /** Whole-number values in [0, max) of an integer or float type. */
    private static INDArray positions(DataType type, long seed, double max, long... shape) {
        return random(DataType.FLOAT, seed, 0, max, shape).castTo(DataType.INT64).castTo(type);
    }

    /** One rounding of the result to the type (half a unit in the last place of 11 or 8 bits, four times). */
    private static double roundingTolerance(DataType type) {
        switch (type) {
            case DOUBLE:
                return 1e-12;
            case FLOAT:
                return 1e-5;
            case HALF:
                return 2e-3;
            default:
                return 1.6e-2;
        }
    }

    /** The rotations (the CPU computes them in float, whatever the type) and one rounding of the result. */
    private static double rotationTolerance(DataType type) {
        switch (type) {
            case DOUBLE:
            case FLOAT:
                return 1e-4;
            case HALF:
                return 4e-3;
            default:
                return 3e-2;
        }
    }

    /** CUDA computes the multimodal rotations with the fast sine, cosine and power of the device. */
    private static double mropeTolerance(DataType type) {
        switch (type) {
            case DOUBLE:
            case FLOAT:
                return 1e-3;
            case HALF:
                return 8e-3;
            default:
                return 4e-2;
        }
    }

    private static void assertClose(String label, double[] expected, INDArray actual, double tolerance) {
        double[] values = actual.dup('c').data().asDouble();
        assertEquals(expected.length, values.length, label + " length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], values[i], tolerance * Math.max(1.0, Math.abs(expected[i])), label + " at " + i);
        }
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

    private static void assertDenseOutputs(String label, DynamicCustomOp op, int outputs) {
        List<DataBuffer> shapes = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(outputs, shapes.size(), label + " outputs");
        for (int i = 0; i < shapes.size(); i++) assertDenseShape(label + " output " + i, shapes.get(i));
    }

    /** Layouts of one [batch, seq, heads, dim] array of values uniform in [-2, 3), numbered 0 to 5. */
    private static INDArray rank4Layout(int layout, DataType type, long seed, long batch, long seq, long heads, long dim) {
        switch (layout) {
            case 0: // dense C order
                return random(type, seed, -2, 3, batch, seq, heads, dim);
            case 1: // F order
                return random(type, seed, -2, 3, batch, seq, heads, dim).dup('f');
            case 2: // [batch, heads, seq, dim] permuted to [batch, seq, heads, dim]: the layout of an attention output
                return random(type, seed, -2, 3, batch, heads, seq, dim).permute(0, 2, 1, 3);
            case 3: // every second element of the last dimension
                return random(type, seed, -2, 3, batch, seq, heads, 2 * dim).get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 2 * dim));
            case 4: // a nonzero base offset, dense strides
                return random(type, seed, -2, 3, batch + 2, seq, heads, dim).get(NDArrayIndex.interval(1, 1 + batch),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            default: // fully reversed axes
                return random(type, seed, -2, 3, dim, heads, seq, batch).permute(3, 2, 1, 0);
        }
    }

    private static final int RANK4_LAYOUTS = 6;

    // ------------------------------------------------------------------------------------------------------------
    // fused_rope
    // ------------------------------------------------------------------------------------------------------------

    /**
     * The rotation of every pair of every head by the angle of its position, flat in C order: pairs (i, i + half) for
     * rope type 0 and (2i, 2i + 1) for 1, angle (position + s) * scale / base^(2i / rotateDims), the dimensions past the
     * rotated ones unchanged. {@code inverse} is the gradient: the rotation by the negated angle.
     */
    private static double[] rope(INDArray x, int ropeType, long position, double base, double scale, int rotaryDims,
                                 boolean inverse) {
        int rank = x.rank();
        int dim = (int) x.size(rank - 1);
        int heads = rank == 4 ? (int) x.size(2) : 1;
        int seq = (int) x.size(1);
        int batch = (int) x.size(0);
        int rotate = rotaryDims > 0 && rotaryDims < dim ? rotaryDims : dim;
        int half = rotate / 2;
        double[] in = x.dup('c').data().asDouble();
        double[] out = in.clone();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int h = 0; h < heads; h++) {
                    int offset = ((b * seq + s) * heads + h) * dim;
                    for (int i = 0; i < half; i++) {
                        double theta = (position + s) * scale / Math.pow(base, 2.0 * i / rotate);
                        double cos = Math.cos(theta);
                        double sin = inverse ? -Math.sin(theta) : Math.sin(theta);
                        int first = ropeType == 1 ? 2 * i : i;
                        int second = ropeType == 1 ? 2 * i + 1 : i + half;
                        out[offset + first] = in[offset + first] * cos - in[offset + second] * sin;
                        out[offset + second] = in[offset + first] * sin + in[offset + second] * cos;
                    }
                }
            }
        }
        return out;
    }

    private static DynamicCustomOp ropeOp(INDArray x, int ropeType, long position, double base, double scale,
                                          int rotaryDims, INDArray... outputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("fused_rope")
                .addInputs(x)
                .addIntegerArguments((long) ropeType, position, (long) rotaryDims)
                .addFloatingPointArguments(base, scale);
        if (outputs.length > 0) builder.addOutputs(outputs);
        return builder.build();
    }

    private static DynamicCustomOp ropeBackwardOp(INDArray x, INDArray gradOut, int ropeType, long position, double base,
                                                  double scale, int rotaryDims, INDArray... outputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("fused_rope_bp")
                .addInputs(x, gradOut)
                .addIntegerArguments((long) ropeType, position, (long) rotaryDims)
                .addFloatingPointArguments(base, scale);
        if (outputs.length > 0) builder.addOutputs(outputs);
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void ropeOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                INDArray x = rank4Layout(layout, type, 100 + layout, 2, 3, 2, 16);
                for (int ropeType = 0; ropeType <= 1; ropeType++) {
                    for (int rotaryDims : new int[]{0, 8}) {
                        String label = type + " layout " + layout + " rope type " + ropeType + " rotary dims "
                                + rotaryDims;
                        INDArray out = Nd4j.exec(ropeOp(x, ropeType, 5, 10000.0, 1.0, rotaryDims))[0];
                        assertEquals(type, out.dataType(), label + " type");
                        assertEquals(Arrays.toString(x.shape()), Arrays.toString(out.shape()), label + " shape");
                        assertClose(label, rope(x, ropeType, 5, 10000.0, 1.0, rotaryDims, false), out,
                                rotationTolerance(type));
                    }
                }
            }
            // rank 3, [batch, seq, head_dim], and a frequency scale
            INDArray rank3 = random(type, 120, -2, 3, 2, 3, 32).dup('f');
            INDArray out = Nd4j.exec(ropeOp(rank3, 0, 2, 500.0, 0.5, 0))[0];
            assertClose(type + " rank 3 F order", rope(rank3, 0, 2, 500.0, 0.5, 0, false), out, rotationTolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void ropeWithAPositionTensor(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.FLOAT16}) {
            for (int layout : new int[]{0, 2, 3}) {
                INDArray x = rank4Layout(layout, type, 130 + layout, 1, 4, 2, 16);
                INDArray position = Nd4j.scalar(DataType.INT64, 7L);
                INDArray out = Nd4j.exec(DynamicCustomOp.builder("fused_rope").addInputs(x, position)
                        .addIntegerArguments(0L).addFloatingPointArguments(10000.0, 1.0).build())[0];
                assertClose(type + " layout " + layout + " position tensor", rope(x, 0, 7, 10000.0, 1.0, 0, false), out,
                        rotationTolerance(type));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void ropeBackwardOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            INDArray x = rank4Layout(0, type, 140, 2, 3, 2, 16);
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                INDArray gradOut = rank4Layout(layout, type, 150 + layout, 2, 3, 2, 16);
                for (int ropeType = 0; ropeType <= 1; ropeType++) {
                    for (int rotaryDims : new int[]{0, 8}) {
                        String label = type + " layout " + layout + " rope type " + ropeType + " rotary dims "
                                + rotaryDims;
                        INDArray gradIn = Nd4j.exec(ropeBackwardOp(x, gradOut, ropeType, 5, 10000.0, 1.0, rotaryDims))[0];
                        assertEquals(type, gradIn.dataType(), label + " type");
                        assertClose(label, rope(gradOut, ropeType, 5, 10000.0, 1.0, rotaryDims, true), gradIn,
                                rotationTolerance(type));
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void ropeIntoCallerProvidedOutputs(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = rotationTolerance(type);
            INDArray x = rank4Layout(0, type, 160, 2, 3, 2, 16);
            INDArray gradOut = rank4Layout(0, type, 161, 2, 3, 2, 16);
            double[] expected = rope(x, 0, 3, 10000.0, 1.0, 0, false);
            double[] expectedBackward = rope(gradOut, 0, 3, 10000.0, 1.0, 0, true);

            // into every second element of the last dimension of a sentinel-filled array: the others keep it
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            Nd4j.exec(ropeOp(x, 0, 3, 10000.0, 1.0, 0, stepped.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 32))));
            assertSteppedOutput(type + " rope", expected, stepped, 2 * 3 * 2, 16, tolerance);

            // the same for the backprop, into every second one from the second on
            INDArray steppedBackward = Nd4j.valueArrayOf(new long[]{2, 3, 2, 33}, SENTINEL, type);
            Nd4j.exec(ropeBackwardOp(x, gradOut, 0, 3, 10000.0, 1.0, 0, steppedBackward.get(NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 33))));
            double[] flat = steppedBackward.dup('c').data().asDouble();
            for (int row = 0; row < 12; row++) {
                for (int c = 0; c < 16; c++) {
                    assertEquals(expectedBackward[row * 16 + c], flat[row * 33 + 2 * c + 1],
                            tolerance * Math.max(1.0, Math.abs(expectedBackward[row * 16 + c])),
                            type + " rope backward at " + row + ", " + c);
                    assertEquals(SENTINEL, flat[row * 33 + 2 * c], 0.0,
                            type + " rope backward's neighbor at " + row + ", " + c);
                }
                assertEquals(SENTINEL, flat[row * 33 + 32], 0.0, type + " rope backward's last element of row " + row);
            }

            // into arrays of another float type: the op takes any float type for its output
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 2, 3, 2, 16);
            Nd4j.exec(ropeOp(x, 0, 3, 10000.0, 1.0, 0, otherType));
            assertEquals(other, otherType.dataType(), type + " output type");
            assertClose(type + " rope into " + other, expected, otherType, 1e-4);
            INDArray otherBackward = Nd4j.create(other, 2, 3, 2, 16);
            Nd4j.exec(ropeBackwardOp(x, gradOut, 0, 3, 10000.0, 1.0, 0, otherBackward));
            assertClose(type + " rope backward into " + other, expectedBackward, otherBackward, 1e-4);
        }
    }

    /** `rows` rows of `cols` values at every second element of rows of 2 * cols (+ nothing), the other elements untouched. */
    private static void assertSteppedOutput(String label, double[] expected, INDArray parent, int rows, int cols,
                                            double tolerance) {
        double[] flat = parent.dup('c').data().asDouble();
        for (int row = 0; row < rows; row++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(expected[row * cols + c], flat[row * 2 * cols + 2 * c],
                        tolerance * Math.max(1.0, Math.abs(expected[row * cols + c])), label + " at " + row + ", " + c);
                assertEquals(SENTINEL, flat[row * 2 * cols + 2 * c + 1], 0.0, label + "'s neighbor at " + row + ", " + c);
            }
        }
    }

    /** The cached rope's table of logical shape [S, W], [B, S, W] or [B, S, 1, W], read at (b, s, i). */
    private static double tableAt(double[] flat, long[] shape, int b, int s, int i) {
        switch (shape.length) {
            case 2:
                return flat[(int) (s * shape[1] + i)];
            case 3:
                return flat[(int) ((b * shape[1] + s) * shape[2] + i)];
            default:
                return flat[(int) ((b * shape[1] + s) * shape[3] + i)];
        }
    }

    /** The reference of the cached rope: the rotation by the angles whose cosines and sines are the tables'. */
    private static double[] cachedRope(INDArray x, INDArray cos, INDArray sin, int ropeType) {
        int rank = x.rank();
        int dim = (int) x.size(rank - 1);
        int heads = rank == 4 ? (int) x.size(2) : 1;
        int seq = (int) x.size(1);
        int batch = (int) x.size(0);
        int half = dim / 2;
        double[] in = x.dup('c').data().asDouble();
        double[] cosines = cos.dup('c').data().asDouble();
        double[] sines = sin.dup('c').data().asDouble();
        double[] out = in.clone();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int h = 0; h < heads; h++) {
                    int offset = ((b * seq + s) * heads + h) * dim;
                    for (int i = 0; i < half; i++) {
                        double c = tableAt(cosines, cos.shape(), b, s, i);
                        double sn = tableAt(sines, sin.shape(), b, s, i);
                        int first = ropeType == 1 ? 2 * i : i;
                        int second = ropeType == 1 ? 2 * i + 1 : i + half;
                        out[offset + first] = in[offset + first] * c - in[offset + second] * sn;
                        out[offset + second] = in[offset + first] * sn + in[offset + second] * c;
                    }
                }
            }
        }
        return out;
    }

    private static DynamicCustomOp cachedRopeOp(INDArray x, INDArray cos, INDArray sin, int ropeType, INDArray... outputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("fused_rope")
                .addInputs(x, cos, sin).addIntegerArguments((long) ropeType);
        if (outputs.length > 0) builder.addOutputs(outputs);
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cachedRopeWithEveryTableLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = rotationTolerance(type);
            INDArray x = rank4Layout(0, type, 170, 2, 3, 2, 16);
            // [S, half] with more rows than the input has positions, [B, S, half] and [B, S, 1, half]
            INDArray[] cosines = {
                    random(type, 171, -1, 1, 5, 8),
                    random(type, 172, -1, 1, 2, 3, 8),
                    random(type, 173, -1, 1, 2, 3, 1, 8),
            };
            INDArray[] sines = {
                    random(type, 174, -1, 1, 5, 8),
                    random(type, 175, -1, 1, 2, 3, 8),
                    random(type, 176, -1, 1, 2, 3, 1, 8),
            };
            for (int t = 0; t < cosines.length; t++) {
                for (int ropeType = 0; ropeType <= 1; ropeType++) {
                    INDArray out = Nd4j.exec(cachedRopeOp(x, cosines[t], sines[t], ropeType))[0];
                    assertClose(type + " table rank " + cosines[t].rank() + " rope type " + ropeType,
                            cachedRope(x, cosines[t], sines[t], ropeType), out, tolerance);
                }
            }

            // tables that are slices of larger arrays: rows [1, 4) of an offset array, the first half of the columns of
            // tables as wide as the head (the layout of tables that repeat each half), an F-ordered sin table
            INDArray offsetRows = random(type, 177, -1, 1, 6, 8).get(NDArrayIndex.interval(1, 4), NDArrayIndex.all());
            INDArray offsetSin = random(type, 178, -1, 1, 6, 8).get(NDArrayIndex.interval(1, 4), NDArrayIndex.all());
            INDArray wideCos = random(type, 179, -1, 1, 3, 16).get(NDArrayIndex.all(), NDArrayIndex.interval(0, 8));
            INDArray wideSin = random(type, 180, -1, 1, 3, 16).get(NDArrayIndex.all(), NDArrayIndex.interval(0, 8));
            INDArray fSin = random(type, 181, -1, 1, 3, 8).dup('f');
            INDArray cDense = random(type, 182, -1, 1, 3, 8);
            INDArray permutedSin = random(type, 183, -1, 1, 8, 3).transpose();
            INDArray[][] pairs = {{offsetRows, offsetSin}, {wideCos, wideSin}, {cDense, fSin}, {fSin, cDense},
                    {cDense, permutedSin}, {wideCos, fSin}};
            String[] names = {"offset rows", "wide tables", "F order sin", "F order cos", "permuted sin",
                    "wide cos and F order sin"};
            for (int p = 0; p < pairs.length; p++) {
                INDArray out = Nd4j.exec(cachedRopeOp(x, pairs[p][0], pairs[p][1], 0))[0];
                assertClose(type + " " + names[p], cachedRope(x, pairs[p][0], pairs[p][1], 0), out, tolerance);
            }

            // an input of any layout
            for (int layout = 1; layout < RANK4_LAYOUTS; layout++) {
                INDArray view = rank4Layout(layout, type, 190 + layout, 2, 3, 2, 16);
                INDArray out = Nd4j.exec(cachedRopeOp(view, cosines[1], sines[1], 0))[0];
                assertClose(type + " input layout " + layout, cachedRope(view, cosines[1], sines[1], 0), out, tolerance);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cachedRopeOfEveryInputAndTableType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            for (DataType tableType : FLOAT_TYPES) {
                INDArray x = rank4Layout(0, type, 200, 2, 3, 2, 16);
                INDArray cos = random(tableType, 201, -1, 1, 3, 8);
                INDArray sin = random(tableType, 202, -1, 1, 3, 8);
                INDArray out = Nd4j.exec(cachedRopeOp(x, cos, sin, 0))[0];
                assertEquals(type, out.dataType(), type + " with " + tableType + " tables: type");
                assertClose(type + " with " + tableType + " tables", cachedRope(x, cos, sin, 0), out,
                        rotationTolerance(type));
            }
            // a cos table and a sin table of different types
            INDArray x = rank4Layout(0, type, 203, 2, 3, 2, 16);
            INDArray cos = random(DataType.FLOAT, 204, -1, 1, 3, 8);
            INDArray sin = random(DataType.FLOAT16, 205, -1, 1, 3, 8);
            INDArray out = Nd4j.exec(cachedRopeOp(x, cos, sin, 0))[0];
            assertClose(type + " with FLOAT cos and HALF sin", cachedRope(x, cos, sin, 0), out, rotationTolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cachedRopeIntoCallerProvidedOutputs(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray x = rank4Layout(0, type, 210, 2, 3, 2, 16);
            INDArray cos = random(type, 211, -1, 1, 3, 8);
            INDArray sin = random(type, 212, -1, 1, 3, 8);
            double[] expected = cachedRope(x, cos, sin, 0);
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            Nd4j.exec(cachedRopeOp(x, cos, sin, 0, stepped.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 32))));
            assertSteppedOutput(type + " cached rope", expected, stepped, 12, 16, rotationTolerance(type));
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 2, 3, 2, 16);
            Nd4j.exec(cachedRopeOp(x, cos, sin, 0, otherType));
            assertClose(type + " cached rope into " + other, expected, otherType, 1e-4);
        }
    }

    /** Tables that are shorter than the rotation reads, or a sin table of another shape, are rejected, not read past. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cachedRopeRejectsTablesOfTheWrongSize(Nd4jBackend backend) {
        INDArray x = rank4Layout(0, DataType.FLOAT, 220, 2, 3, 2, 16);
        INDArray good = random(DataType.FLOAT, 221, -1, 1, 3, 8);
        // fewer rows than the 3 positions of the input
        INDArray shortTable = random(DataType.FLOAT, 222, -1, 1, 2, 8);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(cachedRopeOp(x, shortTable, shortTable.dup(), 0)));
        // rows narrower than half the head dimension
        INDArray narrow = random(DataType.FLOAT, 223, -1, 1, 3, 4);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(cachedRopeOp(x, narrow, narrow.dup(), 0)));
        // a sin table of another shape than the cos table
        INDArray longer = random(DataType.FLOAT, 224, -1, 1, 4, 8);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(cachedRopeOp(x, good, longer, 0)));
        // a table for one batch entry of an input with two
        INDArray oneBatch = random(DataType.FLOAT, 225, -1, 1, 1, 3, 8);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(cachedRopeOp(x, oneBatch, oneBatch.dup(), 0)));
        // rank 5 tables
        INDArray rank5 = random(DataType.FLOAT, 226, -1, 1, 2, 3, 1, 1, 8);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(cachedRopeOp(x, rank5, rank5.dup(), 0)));
        // a rank 2 input
        INDArray rank2 = random(DataType.FLOAT, 227, -1, 1, 3, 16);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(cachedRopeOp(rank2, good, good.dup(), 0)));
    }

    // ------------------------------------------------------------------------------------------------------------
    // fused_bias_dropout_residual
    // ------------------------------------------------------------------------------------------------------------

    /** Inference: input + bias (repeated along the dense order) + residual, flat in C order. */
    private static double[] biasResidual(INDArray x, INDArray bias, INDArray residual) {
        double[] xs = x.dup('c').data().asDouble();
        double[] bs = bias.dup('c').data().asDouble();
        double[] rs = residual.dup('c').data().asDouble();
        double[] out = new double[xs.length];
        for (int i = 0; i < xs.length; i++) out[i] = xs[i] + bs[i % bs.length] + rs[i];
        return out;
    }

    private static INDArray biasDropoutResidual(INDArray x, INDArray bias, INDArray residual, INDArray output,
                                                double probability, long seed, boolean training) {
        if (output == null) {
            return Nd4j.exec(new FusedBiasDropoutResidual(x, bias, residual, null, probability, seed, training))[0];
        }
        Nd4j.exec(new FusedBiasDropoutResidual(x, bias, residual, output, probability, seed, training));
        return output;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void biasDropoutResidualOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            double tolerance = roundingTolerance(type);
            INDArray bias = random(type, 300, -0.5, 0.5, 16);
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                INDArray x = rank4Layout(layout, type, 301 + layout, 2, 3, 2, 16);
                INDArray residual = rank4Layout((layout + 1) % RANK4_LAYOUTS, type, 311 + layout, 2, 3, 2, 16);
                INDArray out = biasDropoutResidual(x, bias, residual, null, 0.0, 0, false);
                assertEquals(type, out.dataType(), type + " layout " + layout + " type");
                assertEquals(Arrays.toString(x.shape()), Arrays.toString(out.shape()), type + " layout " + layout + " shape");
                assertClose(type + " layout " + layout, biasResidual(x, bias, residual), out, tolerance);
            }
            // a bias that is every second element of a longer vector
            INDArray x = rank4Layout(0, type, 320, 2, 3, 2, 16);
            INDArray residual = rank4Layout(0, type, 321, 2, 3, 2, 16);
            INDArray stepped = random(type, 322, -0.5, 0.5, 32).get(NDArrayIndex.interval(0, 2, 32));
            assertClose(type + " stepped bias", biasResidual(x, stepped, residual),
                    biasDropoutResidual(x, stepped, residual, null, 0.0, 0, false), tolerance);
            // a probability without training changes nothing
            assertClose(type + " dropout probability without training", biasResidual(x, bias, residual),
                    biasDropoutResidual(x, bias, residual, null, 0.5, 5, false), tolerance);
            // a probability of zero in training changes nothing either
            assertClose(type + " training without a probability", biasResidual(x, bias, residual),
                    biasDropoutResidual(x, bias, residual, null, 0.0, 5, true), tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void biasDropoutResidualIntoCallerProvidedOutputs(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = roundingTolerance(type);
            INDArray x = rank4Layout(0, type, 330, 2, 3, 2, 16);
            INDArray residual = rank4Layout(0, type, 331, 2, 3, 2, 16);
            INDArray bias = random(type, 332, -0.5, 0.5, 16);
            double[] expected = biasResidual(x, bias, residual);
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            biasDropoutResidual(x, bias, residual, stepped.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 32)), 0.0, 0, false);
            assertSteppedOutput(type + " bias dropout residual", expected, stepped, 12, 16, tolerance);
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 2, 3, 2, 16);
            biasDropoutResidual(x, bias, residual, otherType, 0.0, 0, false);
            assertClose(type + " bias dropout residual into " + other, expected, otherType, 1e-5);
        }
    }

    /**
     * In training every element is kept, scaled by 1 / (1 - p), or dropped to zero (leaving the residual), whatever the
     * layout of the operands: the elements of the unused neighbors of a stepped view (the sentinel) never reach the output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void biasDropoutResidualDropsAndScalesInTraining(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            // x is 1 on every second column of a sentinel-filled array, the bias and the residual are 0 and 4: an element
            // is 4 (dropped) or 6 (kept: 1 / 0.5 + 4)
            INDArray parent = Nd4j.valueArrayOf(new long[]{100, 400}, SENTINEL, type);
            INDArray x = parent.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 400));
            x.assign(1.0);
            INDArray bias = Nd4j.zeros(type, 200);
            INDArray residual = Nd4j.valueArrayOf(new long[]{100, 200}, 4.0, type);
            INDArray out = biasDropoutResidual(x, bias, residual, null, 0.5, 1234, true);
            double[] values = out.dup('c').data().asDouble();
            int dropped = 0;
            for (double value : values) {
                assertTrue(value == 4.0 || value == 6.0, type + " output element " + value + " is neither dropped nor kept");
                if (value == 4.0) dropped++;
            }
            double fraction = dropped / (double) values.length;
            assertTrue(Math.abs(fraction - 0.5) < 0.03, type + " dropped fraction " + fraction + " of 20000 at p = 0.5");
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // fused_mrope
    // ------------------------------------------------------------------------------------------------------------

    /**
     * The multimodal rotation of the kernels, flat in C order: pairs (d, d + head_dim / 2); contiguous sections take the
     * frequency of their own section, 1 / base^(2 local / size), and the position of theirs; interleaved mode takes the
     * position of d % 3 and the frequency 1 / 10000^(2 (d / 3) / ((head_dim + 2) / 3)).
     */
    private static double[] mrope(INDArray x, INDArray posT, INDArray posH, INDArray posW, int sectionT, int sectionH,
                                  int sectionW, boolean interleaved, double base) {
        int batch = (int) x.size(0);
        int seq = (int) x.size(1);
        int heads = (int) x.size(2);
        int dim = (int) x.size(3);
        int half = dim / 2;
        double[] in = x.dup('c').data().asDouble();
        double[] pt = posT.dup('c').data().asDouble();
        double[] ph = posH.dup('c').data().asDouble();
        double[] pw = posW.dup('c').data().asDouble();
        double[] out = in.clone();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int h = 0; h < heads; h++) {
                    int offset = ((b * seq + s) * heads + h) * dim;
                    for (int d = 0; d < half; d++) {
                        double position;
                        double frequency;
                        if (interleaved) {
                            position = d % 3 == 0 ? pt[b * seq + s] : d % 3 == 1 ? ph[b * seq + s] : pw[b * seq + s];
                            frequency = 1.0 / Math.pow(10000.0, 2.0 * (d / 3) / ((dim + 2) / 3));
                        } else if (d < sectionT / 2) {
                            position = pt[b * seq + s];
                            frequency = 1.0 / Math.pow(base, 2.0 * d / sectionT);
                        } else if (d < sectionT / 2 + sectionH / 2) {
                            position = ph[b * seq + s];
                            frequency = 1.0 / Math.pow(base, 2.0 * (d - sectionT / 2) / sectionH);
                        } else {
                            position = pw[b * seq + s];
                            frequency = 1.0 / Math.pow(base, 2.0 * (d - sectionT / 2 - sectionH / 2) / sectionW);
                        }
                        double angle = position * frequency;
                        double cos = Math.cos(angle);
                        double sin = Math.sin(angle);
                        out[offset + d] = in[offset + d] * cos - in[offset + d + half] * sin;
                        out[offset + d + half] = in[offset + d] * sin + in[offset + d + half] * cos;
                    }
                }
            }
        }
        return out;
    }

    private static DynamicCustomOp mropeOp(INDArray x, INDArray posT, INDArray posH, INDArray posW, int sectionT,
                                           int sectionH, int sectionW, boolean interleaved, double base,
                                           INDArray... outputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("fused_mrope")
                .addInputs(x, posT, posH, posW)
                .addIntegerArguments((long) sectionT, (long) sectionH, (long) sectionW, interleaved ? 1L : 0L)
                .addFloatingPointArguments(base);
        if (outputs.length > 0) builder.addOutputs(outputs);
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mropeOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16, DataType.BFLOAT16}) {
            for (boolean interleaved : new boolean[]{false, true}) {
                for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                    INDArray x = rank4Layout(layout, type, 400 + layout, 2, 3, 2, 16);
                    INDArray posT = positions(DataType.INT64, 410, 6, 2, 3);
                    INDArray posH = positions(DataType.INT64, 411, 9, 2, 3);
                    INDArray posW = positions(DataType.INT64, 412, 12, 2, 3);
                    String label = type + " layout " + layout + (interleaved ? " interleaved" : " contiguous");
                    INDArray out = Nd4j.exec(mropeOp(x, posT, posH, posW, 8, 4, 4, interleaved, 10000.0))[0];
                    assertEquals(type, out.dataType(), label + " type");
                    assertEquals(Arrays.toString(x.shape()), Arrays.toString(out.shape()), label + " shape");
                    assertClose(label, mrope(x, posT, posH, posW, 8, 4, 4, interleaved, 10000.0), out,
                            mropeTolerance(type));
                }
            }
            // the Qwen3-VL sections of a head of 64
            INDArray x = random(type, 420, -2, 3, 1, 4, 2, 64).dup('f');
            INDArray posT = positions(DataType.INT64, 421, 6, 1, 4);
            INDArray posH = positions(DataType.INT64, 422, 9, 1, 4);
            INDArray posW = positions(DataType.INT64, 423, 12, 1, 4);
            INDArray out = Nd4j.exec(new FusedMRoPE(x, posT, posH, posW, 24, 20, 20, false, 10000.0))[0];
            assertClose(type + " Qwen sections, F order", mrope(x, posT, posH, posW, 24, 20, 20, false, 10000.0), out,
                    mropeTolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mropeWithPositionsOfEveryTypeAndLayout(Nd4jBackend backend) {
        INDArray x = rank4Layout(0, DataType.FLOAT, 430, 2, 3, 2, 16);
        for (DataType positionType : new DataType[]{DataType.INT64, DataType.INT32, DataType.FLOAT}) {
            INDArray posT = positions(positionType, 431, 6, 2, 3);
            INDArray posH = positions(positionType, 432, 9, 2, 3);
            INDArray posW = positions(positionType, 433, 12, 2, 3);
            INDArray out = Nd4j.exec(mropeOp(x, posT, posH, posW, 8, 4, 4, false, 10000.0))[0];
            assertClose("positions of type " + positionType, mrope(x, posT, posH, posW, 8, 4, 4, false, 10000.0), out,
                    mropeTolerance(DataType.FLOAT));
        }
        // positions that are every second column, F ordered, or at an offset
        INDArray steppedT = positions(DataType.INT64, 434, 6, 2, 6).get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 6));
        INDArray fH = positions(DataType.INT64, 435, 9, 2, 3).dup('f');
        INDArray offsetW = positions(DataType.INT64, 436, 12, 4, 3).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all());
        INDArray out = Nd4j.exec(mropeOp(x, steppedT, fH, offsetW, 8, 4, 4, false, 10000.0))[0];
        assertClose("stepped, F ordered and offset positions", mrope(x, steppedT, fH, offsetW, 8, 4, 4, false, 10000.0),
                out, mropeTolerance(DataType.FLOAT));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mropeIntoCallerProvidedOutputs(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray x = rank4Layout(0, type, 440, 2, 3, 2, 16);
            INDArray posT = positions(DataType.INT64, 441, 6, 2, 3);
            INDArray posH = positions(DataType.INT64, 442, 9, 2, 3);
            INDArray posW = positions(DataType.INT64, 443, 12, 2, 3);
            double[] expected = mrope(x, posT, posH, posW, 8, 4, 4, false, 10000.0);
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            Nd4j.exec(mropeOp(x, posT, posH, posW, 8, 4, 4, false, 10000.0, stepped.get(NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 32))));
            assertSteppedOutput(type + " mrope", expected, stepped, 12, 16, mropeTolerance(type));
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 2, 3, 2, 16);
            Nd4j.exec(mropeOp(x, posT, posH, posW, 8, 4, 4, false, 10000.0, otherType));
            assertClose(type + " mrope into " + other, expected, otherType, mropeTolerance(type));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mropeRejectsArgumentsTheKernelsWouldReadOrLeaveUnwritten(Nd4jBackend backend) {
        INDArray x = rank4Layout(0, DataType.FLOAT, 450, 2, 3, 2, 16);
        INDArray posT = positions(DataType.INT64, 451, 6, 2, 3);
        INDArray posH = positions(DataType.INT64, 452, 9, 2, 3);
        INDArray posW = positions(DataType.INT64, 453, 12, 2, 3);
        // positions for fewer batch entries, of another shape than the temporal positions, and an odd head dimension
        INDArray shortPositions = positions(DataType.INT64, 454, 6, 1, 3);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(mropeOp(x, shortPositions, posH, posW, 8, 4, 4, false, 10000.0)));
        assertThrows(RuntimeException.class, () -> Nd4j.exec(mropeOp(x, posT, shortPositions, posW, 8, 4, 4, false, 10000.0)));
        INDArray odd = rank4Layout(0, DataType.FLOAT, 455, 2, 3, 2, 15);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(mropeOp(odd, posT, posH, posW, 7, 4, 4, false, 10000.0)));
    }

    // ------------------------------------------------------------------------------------------------------------
    // vision_embedding_merge
    // ------------------------------------------------------------------------------------------------------------

    /**
     * Each position holding the target token takes the next vision row of its batch entry (while there are any), the
     * others keep their text row, flat in C order.
     */
    private static double[] visionMerge(INDArray text, INDArray vision, INDArray tokens, long target) {
        int batch = (int) text.size(0);
        int seq = (int) text.size(1);
        int hidden = (int) text.size(2);
        int visionTokens = (int) vision.size(1);
        double[] t = text.dup('c').data().asDouble();
        double[] v = vision.dup('c').data().asDouble();
        double[] ids = tokens.dup('c').data().asDouble();
        double[] out = new double[t.length];
        for (int b = 0; b < batch; b++) {
            int next = 0;
            for (int s = 0; s < seq; s++) {
                boolean fromVision = (long) ids[b * seq + s] == target && next < visionTokens;
                for (int h = 0; h < hidden; h++) {
                    out[(b * seq + s) * hidden + h] = fromVision ? v[(b * visionTokens + next) * hidden + h]
                            : t[(b * seq + s) * hidden + h];
                }
                if (fromVision) next++;
            }
        }
        return out;
    }

    private static DynamicCustomOp visionMergeOp(INDArray text, INDArray vision, INDArray tokens, long target,
                                                 INDArray... outputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("vision_embedding_merge")
                .addInputs(text, vision, tokens).addIntegerArguments(target);
        if (outputs.length > 0) builder.addOutputs(outputs);
        return builder.build();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void visionEmbeddingMergeOfSteppedViews(Nd4jBackend backend) {
        long target = 9;
        // the second batch entry has five target tokens for its four vision rows: the fifth keeps its text row
        INDArray tokens = Nd4j.createFromArray(new long[][]{{1, 9, 9, 2, 9, 3}, {9, 4, 9, 9, 9, 9}});
        INDArray steppedTokens = Nd4j.create(DataType.INT64, 2, 12);
        steppedTokens.assign(5);
        steppedTokens.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 12)).assign(tokens);
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16}) {
            double tolerance = roundingTolerance(type);
            INDArray text = random(type, 500, -2, 3, 2, 6, 8);
            INDArray vision = random(type, 501, -2, 3, 2, 4, 8);
            INDArray out = Nd4j.exec(visionMergeOp(text, vision, tokens, target))[0];
            assertClose(type + " dense", visionMerge(text, vision, tokens, target), out, tolerance);

            // every operand stepped, F ordered or offset
            INDArray steppedText = random(type, 502, -2, 3, 2, 6, 16).get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 16));
            INDArray fVision = random(type, 503, -2, 3, 2, 4, 8).dup('f');
            INDArray merged = Nd4j.exec(visionMergeOp(steppedText, fVision, steppedTokens.get(NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 12)), target))[0];
            assertEquals(type, merged.dataType(), type + " stepped type");
            assertClose(type + " stepped text, F ordered vision, stepped tokens",
                    visionMerge(steppedText, fVision, tokens, target), merged, tolerance);
            INDArray offsetText = random(type, 504, -2, 3, 4, 6, 8).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                    NDArrayIndex.all());
            INDArray offsetVision = random(type, 505, -2, 3, 4, 4, 8).get(NDArrayIndex.interval(2, 4), NDArrayIndex.all(),
                    NDArrayIndex.all());
            assertClose(type + " offset text and vision", visionMerge(offsetText, offsetVision, tokens, target),
                    Nd4j.exec(visionMergeOp(offsetText, offsetVision, tokens, target))[0], tolerance);

            // into every second element of the last dimension of a sentinel-filled array
            INDArray parent = Nd4j.valueArrayOf(new long[]{2, 6, 16}, SENTINEL, type);
            Nd4j.exec(visionMergeOp(steppedText, fVision, tokens, target, parent.get(NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 16))));
            assertSteppedOutput(type + " vision merge", visionMerge(steppedText, fVision, tokens, target), parent, 12, 8,
                    tolerance);
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // fused_gelu
    // ------------------------------------------------------------------------------------------------------------

    /** x * sigmoid(1.702 x), or with dy its derivative times dy, flat in C order. */
    private static double[] gelu(INDArray x, INDArray dy) {
        double[] xs = x.dup('c').data().asDouble();
        double[] dys = dy == null ? null : dy.dup('c').data().asDouble();
        double[] out = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-1.702 * xs[i]));
            out[i] = dys == null ? xs[i] * sigmoid
                    : dys[i] * (sigmoid + xs[i] * 1.702 * sigmoid * (1.0 - sigmoid));
        }
        return out;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void geluOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                INDArray x = rank4Layout(layout, type, 700 + layout, 2, 3, 2, 16);
                INDArray out = Nd4j.exec(new FusedGELU(x))[0];
                String label = type + " layout " + layout;
                assertEquals(type, out.dataType(), label + " type");
                assertEquals(Arrays.toString(x.shape()), Arrays.toString(out.shape()), label + " shape");
                assertClose(label, gelu(x, null), out, roundingTolerance(type));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void geluBackwardOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                INDArray x = rank4Layout(layout, type, 710 + layout, 2, 3, 2, 16);
                // every operand in a different layout
                INDArray dy = rank4Layout((layout + 2) % RANK4_LAYOUTS, type, 720 + layout, 2, 3, 2, 16);
                INDArray gradIn = Nd4j.exec(new FusedGELUBp(x, dy, null))[0];
                String label = type + " layout " + layout;
                assertEquals(type, gradIn.dataType(), label + " type");
                assertClose(label, gelu(x, dy), gradIn, roundingTolerance(type));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void geluIntoCallerProvidedOutputsAndInPlace(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = roundingTolerance(type);
            INDArray x = rank4Layout(0, type, 730, 2, 3, 2, 16);
            INDArray dy = rank4Layout(0, type, 731, 2, 3, 2, 16);
            double[] expected = gelu(x, null);
            double[] expectedBackward = gelu(x, dy);

            // into every second element of the last dimension of a sentinel-filled array: the others keep it
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            Nd4j.exec(new FusedGELU(x, stepped.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 32))));
            assertSteppedOutput(type + " gelu", expected, stepped, 12, 16, tolerance);
            INDArray steppedGradient = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            Nd4j.exec(new FusedGELUBp(x, dy, steppedGradient.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 32))));
            assertSteppedOutput(type + " gelu backward", expectedBackward, steppedGradient, 12, 16, tolerance);

            // into arrays of another float type
            DataType other = type == DataType.FLOAT ? DataType.DOUBLE : DataType.FLOAT;
            INDArray otherType = Nd4j.create(other, 2, 3, 2, 16);
            Nd4j.exec(new FusedGELU(x, otherType));
            assertClose(type + " gelu into " + other, expected, otherType, 1e-5);

            // in place, on a dense array and on a stepped view of a sentinel-filled array
            INDArray inPlace = x.dup();
            Nd4j.exec(new FusedGELU(inPlace, inPlace));
            assertClose(type + " gelu in place", expected, inPlace, tolerance);
            INDArray parent = Nd4j.valueArrayOf(new long[]{2, 3, 2, 32}, SENTINEL, type);
            INDArray view = parent.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 32));
            view.assign(x);
            Nd4j.exec(new FusedGELU(view, view));
            assertSteppedOutput(type + " gelu in place on a stepped view", expected, parent, 12, 16, tolerance);
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // fused_attention_projection
    // ------------------------------------------------------------------------------------------------------------

    /** reshape(attention, [batch * seq, hidden]) @ wo + bias, flat in C order. */
    private static double[] projection(INDArray attention, INDArray wo, INDArray bias) {
        int hidden = (int) wo.size(0);
        int out = (int) wo.size(1);
        int rows = (int) (attention.length() / hidden);
        double[] a = attention.dup('c').data().asDouble();
        double[] w = wo.dup('c').data().asDouble();
        double[] b = bias == null ? new double[out] : bias.dup('c').data().asDouble();
        double[] result = new double[rows * out];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < out; j++) {
                double sum = b[j];
                for (int k = 0; k < hidden; k++) sum += a[r * hidden + k] * w[k * out + j];
                result[r * out + j] = sum;
            }
        }
        return result;
    }

    private static DynamicCustomOp projectionOp(INDArray attention, INDArray wo, INDArray bias, INDArray... outputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("fused_attention_projection");
        if (bias == null) builder.addInputs(attention, wo);
        else builder.addInputs(attention, wo, bias);
        if (outputs.length > 0) builder.addOutputs(outputs);
        return builder.build();
    }

    /** A projection weight [16, 5]: the products of attention rows of unit scale with it are of a scale of 2. */
    private static INDArray projectionWeights(DataType type, long seed) {
        return random(type, seed, -0.75, 0.75, 16, 5);
    }

    private static double projectionTolerance(DataType type) {
        switch (type) {
            case DOUBLE:
                return 1e-10;
            case FLOAT:
                return 1e-4;
            case HALF:
                return 8e-3;
            default:
                return 6e-2;
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void attentionProjectionOfEveryLayoutAndType(Nd4jBackend backend) {
        for (DataType type : FLOAT_TYPES) {
            double tolerance = projectionTolerance(type);
            INDArray wo = projectionWeights(type, 800);
            INDArray bias = random(type, 801, -0.5, 0.5, 5);
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                // [batch, seq, heads, head_dim] in every layout, among them the permuted one of an attention output
                INDArray attention = rank4Layout(layout, type, 810 + layout, 2, 3, 2, 8);
                String label = type + " layout " + layout;
                INDArray with = Nd4j.exec(projectionOp(attention, wo, bias))[0];
                assertEquals(type, with.dataType(), label + " type");
                assertEquals(Arrays.toString(new long[]{2, 3, 5}), Arrays.toString(with.shape()), label + " shape");
                assertClose(label + " with a bias", projection(attention, wo, bias), with, tolerance);
                assertClose(label + " without a bias", projection(attention, wo, null),
                        Nd4j.exec(projectionOp(attention, wo, null))[0], tolerance);
            }
            // rank 3 attention output [batch, seq, hidden], F ordered
            INDArray rank3 = random(type, 820, -2, 3, 2, 3, 16).dup('f');
            assertClose(type + " rank 3 F order", projection(rank3, wo, bias),
                    Nd4j.exec(projectionOp(rank3, wo, bias))[0], tolerance);
            // the weights and the bias in other layouts
            INDArray attention = rank4Layout(0, type, 821, 2, 3, 2, 8);
            INDArray fWeights = wo.dup('f');
            assertClose(type + " F order weights", projection(attention, fWeights, bias),
                    Nd4j.exec(projectionOp(attention, fWeights, bias))[0], tolerance);
            INDArray steppedBias = random(type, 822, -0.5, 0.5, 10).get(NDArrayIndex.interval(0, 2, 10));
            assertClose(type + " stepped bias", projection(attention, wo, steppedBias),
                    Nd4j.exec(projectionOp(attention, wo, steppedBias))[0], tolerance);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void attentionProjectionWithABiasOfAnotherType(Nd4jBackend backend) {
        // the bias is added in the type of the output, whatever the type it is stored in
        INDArray attention = rank4Layout(0, DataType.FLOAT, 830, 2, 3, 2, 8);
        INDArray wo = projectionWeights(DataType.FLOAT, 831);
        INDArray halfBias = random(DataType.FLOAT16, 832, -0.5, 0.5, 5);
        assertClose("HALF bias", projection(attention, wo, halfBias), Nd4j.exec(projectionOp(attention, wo, halfBias))[0],
                1e-4);
        INDArray doubleBias = random(DataType.DOUBLE, 833, -0.5, 0.5, 5);
        assertClose("DOUBLE bias", projection(attention, wo, doubleBias),
                Nd4j.exec(projectionOp(attention, wo, doubleBias))[0], 1e-4);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void attentionProjectionIntoCallerProvidedOutputs(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            double tolerance = projectionTolerance(type);
            INDArray attention = rank4Layout(0, type, 840, 2, 3, 2, 8);
            INDArray wo = projectionWeights(type, 841);
            INDArray bias = random(type, 842, -0.5, 0.5, 5);
            double[] expected = projection(attention, wo, bias);
            // into every second element of the last dimension of a sentinel-filled array (the reshape of such an
            // output was a copy: the product was lost in it), with and without a bias
            INDArray stepped = Nd4j.valueArrayOf(new long[]{2, 3, 10}, SENTINEL, type);
            Nd4j.exec(projectionOp(attention, wo, bias, stepped.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 10))));
            assertSteppedOutput(type + " attention projection", expected, stepped, 6, 5, tolerance);
            INDArray steppedNoBias = Nd4j.valueArrayOf(new long[]{2, 3, 10}, SENTINEL, type);
            Nd4j.exec(projectionOp(attention, wo, null, steppedNoBias.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, 2, 10))));
            assertSteppedOutput(type + " attention projection without a bias", projection(attention, wo, null),
                    steppedNoBias, 6, 5, tolerance);
            // into the middle batch entries of a taller array (an offset view)
            INDArray offset = Nd4j.valueArrayOf(new long[]{4, 3, 5}, SENTINEL, type);
            Nd4j.exec(projectionOp(attention, wo, bias, offset.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                    NDArrayIndex.all())));
            double[] flat = offset.dup('c').data().asDouble();
            for (int i = 0; i < 4 * 3 * 5; i++) {
                if (i >= 15 && i < 45) {
                    assertEquals(expected[i - 15], flat[i], tolerance * Math.max(1.0, Math.abs(expected[i - 15])),
                            type + " offset output at " + i);
                } else {
                    assertEquals(SENTINEL, flat[i], 0.0, type + " offset output's surroundings at " + i);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // the shape functions of every op above
    // ------------------------------------------------------------------------------------------------------------

    /**
     * The root cause of the wrong values and the writes past the end of a buffer: the output arrays are allocated with one
     * element per element of the shapes the shape functions return, so those shapes must not keep the strides of the
     * arrays they are shaped like.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void outputShapesAreDenseForEveryOperandLayout(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray bias = random(type, 600, -0.5, 0.5, 16);
            INDArray cos = random(type, 601, -1, 1, 3, 8);
            INDArray sin = random(type, 602, -1, 1, 3, 8);
            INDArray posT = positions(DataType.INT64, 603, 6, 2, 3);
            INDArray posH = positions(DataType.INT64, 604, 9, 2, 3);
            INDArray posW = positions(DataType.INT64, 605, 12, 2, 3);
            for (int layout = 0; layout < RANK4_LAYOUTS; layout++) {
                INDArray x = rank4Layout(layout, type, 610 + layout, 2, 3, 2, 16);
                INDArray other = rank4Layout((layout + 3) % RANK4_LAYOUTS, type, 620 + layout, 2, 3, 2, 16);
                String label = type + " layout " + layout + " strides " + Arrays.toString(x.stride());
                assertDenseOutputs(label + " fused_rope", ropeOp(x, 0, 0, 10000.0, 1.0, 0), 1);
                assertDenseOutputs(label + " cached fused_rope", cachedRopeOp(x, cos, sin, 0), 1);
                assertDenseOutputs(label + " fused_rope_bp", ropeBackwardOp(x, other, 0, 0, 10000.0, 1.0, 0), 1);
                assertDenseOutputs(label + " fused_bias_dropout_residual", DynamicCustomOp
                        .builder("fused_bias_dropout_residual").addInputs(x, bias, other)
                        .addIntegerArguments(0L).addFloatingPointArguments(0.0).addBooleanArguments(false).build(), 1);
                assertDenseOutputs(label + " fused_mrope", mropeOp(x, posT, posH, posW, 8, 4, 4, false, 10000.0), 1);
                assertDenseOutputs(label + " fused_gelu", new FusedGELU(x), 1);
                assertDenseOutputs(label + " fused_gelu_bp", new FusedGELUBp(x, other, null), 1);
            }
            for (int variant = 0; variant < 3; variant++) {
                INDArray text = variant == 0 ? random(type, 630, -2, 3, 2, 6, 16).get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, 2, 16))
                        : variant == 1 ? random(type, 631, -2, 3, 2, 6, 8).dup('f')
                        : random(type, 632, -2, 3, 4, 6, 8).get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                                NDArrayIndex.all());
                INDArray vision = random(type, 633, -2, 3, 2, 4, 8);
                INDArray tokens = Nd4j.createFromArray(new long[][]{{1, 9, 9, 2, 9, 3}, {9, 4, 9, 9, 9, 9}});
                assertDenseOutputs(type + " vision_embedding_merge variant " + variant + " strides "
                        + Arrays.toString(text.stride()), visionMergeOp(text, vision, tokens, 9), 1);
            }
        }
    }
}
