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
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An output the framework allocates for an op is a new array that owns exactly its length elements, dense in the
 * order of the shape its shape function describes, whatever layout the inputs have. Many shape functions return an
 * input's shape information for an output of their own (the {@code *_bp} ops, check_numerics, biasadd, the legacy
 * transform ops, the default shape functions of the single-input ops copy its flags), and that shape information
 * carries the input's strides and its view flag. An output allocated from it addresses the buffer of its input: for a
 * stepped view (a [4, 70] slice of every second column, strides [141, 2]) offset 561 of a 280-element buffer, a write
 * past the end that corrupts the heap (CPU) or device memory (CUDA); and the view flag marks an array that owns its
 * buffer as one that does not.
 * <p>
 * The inputs here are laid out as dense C order, F order, an offset view, a stepped view, permuted views (also one
 * that is neither C nor F) and a stepped F-order view, in NaN-filled parents so that a read outside the view shows.
 * Outputs are allocated through every route the framework has (the executioner's own, {@code allocateOutputArrays} and
 * {@code createFromDescriptor} over the shape function's descriptor) and inside a SameDiff graph with and without the
 * dynamic shape plan, where the plan's native executor allocates the outputs. Each output's strides must address
 * exactly its length elements, and its values must equal the op's on a dense copy of the same input. The
 * outputs that are views of their inputs (permute, reshape_no_copy) must stay views.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class DenseOpOutputTest extends BaseNd4jTestWithBackends {

    private static final long[][] SHAPES = {{4, 70}, {3, 4, 5}};

    // ------------------------------------------------------------------------------------------------ layouts

    private static final class Layout {
        final String name;
        final int minRank;
        final UnaryOperator<INDArray> of;

        Layout(String name, int minRank, UnaryOperator<INDArray> of) {
            this.name = name;
            this.minRank = minRank;
            this.of = of;
        }
    }

    private static INDArray parentOf(INDArray x, int dim, long size) {
        long[] s = x.shape().clone();
        s[dim] = size;
        return Nd4j.valueArrayOf(s, Double.NaN, x.dataType());
    }

    /** The indexes selecting {@code x}'s extent in every dimension but {@code dim}, and {@code selected} in it. */
    private static INDArrayIndex[] indexes(int rank, int dim, INDArrayIndex selected) {
        INDArrayIndex[] idx = new INDArrayIndex[rank];
        for (int i = 0; i < rank; i++)
            idx[i] = i == dim ? selected : NDArrayIndex.all();
        return idx;
    }

    private static final Layout[] LAYOUTS = {
            new Layout("C order", 1, x -> x.dup('c')),
            new Layout("F order", 1, x -> x.dup('f')),
            new Layout("offset view", 1, x -> {
                INDArray parent = parentOf(x, 0, x.size(0) + 1);
                return parent.get(indexes(x.rank(), 0, NDArrayIndex.interval(1, x.size(0) + 1))).assign(x);
            }),
            new Layout("stepped view", 1, x -> {
                int last = x.rank() - 1;
                long n = x.size(last);
                INDArray parent = parentOf(x, last, 2 * n + 1);
                return parent.get(indexes(x.rank(), last, NDArrayIndex.interval(1, 2, 2 * n + 1))).assign(x);
            }),
            new Layout("permuted view (F order)", 2, x -> {
                int r = x.rank();
                long[] reversed = new long[r];
                long[] perm = new long[r];
                for (int i = 0; i < r; i++) {
                    reversed[i] = x.size(r - 1 - i);
                    perm[i] = r - 1 - i;
                }
                return Nd4j.create(x.dataType(), reversed, 'c').permute(perm).assign(x);
            }),
            new Layout("permuted view (neither C nor F)", 3, x -> {
                long[] swapped = x.shape().clone();
                swapped[0] = x.size(1);
                swapped[1] = x.size(0);
                long[] perm = new long[x.rank()];
                for (int i = 0; i < perm.length; i++)
                    perm[i] = i;
                perm[0] = 1;
                perm[1] = 0;
                return Nd4j.create(x.dataType(), swapped, 'c').permute(perm).assign(x);
            }),
            new Layout("stepped F-order view", 2, x -> {
                INDArray parent = parentOf(x, 0, 2 * x.size(0) + 1).dup('f');
                return parent.get(indexes(x.rank(), 0, NDArrayIndex.interval(1, 2, 2 * x.size(0) + 1))).assign(x);
            }),
    };

    // -------------------------------------------------------------------------------------------- assertions

    /** Independent of the framework: do the strides pack the elements into consecutive offsets in the array's order? */
    private static boolean packed(long[] shape, long[] stride, char order) {
        long expected = 1;
        if (order == 'f') {
            for (int d = 0; d < shape.length; d++) {
                if (shape[d] != 1 && stride[d] != expected)
                    return false;
                expected *= shape[d];
            }
        } else {
            for (int d = shape.length - 1; d >= 0; d--) {
                if (shape[d] != 1 && stride[d] != expected)
                    return false;
                expected *= shape[d];
            }
        }
        return true;
    }

    /**
     * An output's strides address exactly as many elements as it has, in its order; it is no view, and its buffer is
     * exactly its length.
     */
    private static void assertDense(String label, INDArray out) {
        if (out.isEmpty())
            return;
        long[] shape = out.shape();
        long[] stride = out.stride();
        long span = 1;
        for (int d = 0; d < shape.length; d++)
            span += (shape[d] - 1) * stride[d];
        final long addressed = span;
        assertEquals(out.length(), addressed, () -> label + ": the strides " + Arrays.toString(stride) + " of shape "
                + Arrays.toString(shape) + " address " + addressed + " elements of an array of " + out.length());
        assertTrue(packed(shape, stride, out.ordering()), () -> label + ": the strides " + Arrays.toString(stride)
                + " of shape " + Arrays.toString(shape) + " do not pack the elements in order " + out.ordering());
        assertEquals(out.length(), out.data().length(), () -> label + ": the buffer holds " + out.data().length()
                + " elements, the array " + out.length());
        assertFalse(out.isView(), () -> label + ": a new output is not a view");
        assertFalse(ArrayOptionsHelper.isView(out.shapeInfoJava()), () -> label + ": the view flag is set");
    }

    private static void assertAllDense(String label, INDArray[] outputs) {
        for (int i = 0; i < outputs.length; i++)
            assertDense(label + ", output " + i, outputs[i]);
    }

    private static void assertSameValues(String label, INDArray[] expected, INDArray[] actual) {
        assertEquals(expected.length, actual.length, label + ": the number of outputs");
        for (int i = 0; i < expected.length; i++) {
            final int output = i;
            assertArrayEquals(expected[i].shape(), actual[i].shape(), label + ": shape of output " + i);
            assertEquals(expected[i].dataType(), actual[i].dataType(), label + ": type of output " + i);
            assertTrue(expected[i].equalsWithEps(actual[i], 1e-6),
                    () -> label + ": output " + output + " is " + actual[output] + ", expected " + expected[output]);
        }
    }

    // ---------------------------------------------------------------------------------------------- the ops

    private static INDArray dense(DataType type, long seed, long... shape) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.rand(type, shape).subi(0.5);
    }

    private static DynamicCustomOp op(String name, long[] iArgs, double[] tArgs, INDArray... inputs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(name).addInputs(inputs);
        if (iArgs.length > 0)
            builder.addIntegerArguments(iArgs);
        if (tArgs.length > 0)
            builder.addFloatingPointArguments(Arrays.stream(tArgs).boxed().toArray(Double[]::new));
        return builder.build();
    }

    private static final long[] NO_INTS = new long[0];
    private static final double[] NO_DOUBLES = new double[0];

    private static final class OpCase {
        final String name;
        /** The op over the laid-out first operand; the other operands are dense. */
        final Function<INDArray, DynamicCustomOp> build;

        OpCase(String name, Function<INDArray, DynamicCustomOp> build) {
            this.name = name;
            this.build = build;
        }
    }

    private static List<OpCase> opCases() {
        List<OpCase> cases = new ArrayList<>();
        // shape functions that return an input's shape information as it is (the _bp ops, biasadd)
        cases.add(new OpCase("reverse_bp", x -> op("reverse_bp", new long[]{x.rank() - 1}, NO_DOUBLES, x,
                dense(x.dataType(), 2, x.shape()))));
        cases.add(new OpCase("maximum_bp", x -> op("maximum_bp", NO_INTS, NO_DOUBLES, x,
                dense(x.dataType(), 3, x.shape()), dense(x.dataType(), 4, x.shape()))));
        cases.add(new OpCase("minimum_bp", x -> op("minimum_bp", NO_INTS, NO_DOUBLES, x,
                dense(x.dataType(), 3, x.shape()), dense(x.dataType(), 4, x.shape()))));
        cases.add(new OpCase("biasadd", x -> op("biasadd", NO_INTS, NO_DOUBLES, x,
                dense(x.dataType(), 5, x.size(x.rank() - 1)))));
        cases.add(new OpCase("biasadd_bp", x -> op("biasadd_bp", NO_INTS, NO_DOUBLES, x,
                dense(x.dataType(), 5, x.size(x.rank() - 1)), dense(x.dataType(), 6, x.shape()))));
        cases.add(new OpCase("cumsum_bp", x -> op("cumsum_bp", new long[]{0, 0, x.rank() - 1}, NO_DOUBLES, x,
                dense(x.dataType(), 7, x.shape()))));
        // the default shape functions of the single-input ops: dense strides, but the input's flags
        cases.add(new OpCase("softmax", x -> op("softmax", new long[]{x.rank() - 1}, NO_DOUBLES, x)));
        cases.add(new OpCase("cumsum", x -> op("cumsum", new long[]{0, 0, x.rank() - 1}, NO_DOUBLES, x)));
        cases.add(new OpCase("reverse", x -> op("reverse", new long[]{x.rank() - 1}, NO_DOUBLES, x)));
        cases.add(new OpCase("clipbyvalue", x -> op("clipbyvalue", NO_INTS, new double[]{-0.25, 0.25}, x)));
        return cases;
    }

    private enum Route {
        /** The executioner allocates the outputs of an op that has none. */
        EXEC,
        /** OpExecutioner.allocateOutputArrays, then the op runs over them. */
        ALLOCATE_OUTPUT_ARRAYS,
        /** Nd4j.createFromDescriptor over the shape function's descriptors, then the op runs over them. */
        DESCRIPTOR
    }

    private static INDArray[] run(String label, DynamicCustomOp op, Route route) {
        switch (route) {
            case EXEC: {
                INDArray[] outputs = Nd4j.exec(op);
                assertAllDense(label, outputs);
                return outputs;
            }
            case ALLOCATE_OUTPUT_ARRAYS: {
                INDArray[] outputs = Nd4j.getExecutioner().allocateOutputArrays(op);
                assertAllDense(label, outputs);
                op.addOutputArgument(outputs);
                Nd4j.exec(op);
                return outputs;
            }
            default: {
                List<DataBuffer> descriptors = op.calculateOutputShape();
                INDArray[] outputs = new INDArray[descriptors.size()];
                for (int i = 0; i < outputs.length; i++)
                    outputs[i] = Nd4j.createFromDescriptor(descriptors.get(i));
                assertAllDense(label, outputs);
                op.addOutputArgument(outputs);
                Nd4j.exec(op);
                return outputs;
            }
        }
    }

    // ----------------------------------------------------------------------------------------------- the tests

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void opsAllocateDenseOutputsForEveryInputLayout(Nd4jBackend backend) {
        List<String> failures = new ArrayList<>();
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (long[] shape : SHAPES) {
                INDArray base = dense(type, 1, shape);
                for (OpCase c : opCases()) {
                    INDArray[] expected = Nd4j.exec(c.build.apply(base.dup('c')));
                    Nd4j.getExecutioner().commit();
                    for (Layout layout : LAYOUTS) {
                        if (shape.length < layout.minRank)
                            continue;
                        for (Route route : Route.values()) {
                            String label = type + " " + c.name + " over a " + layout.name + " "
                                    + Arrays.toString(shape) + " input, " + route;
                            try {
                                INDArray[] actual = run(label, c.build.apply(layout.of.apply(base)), route);
                                Nd4j.getExecutioner().commit();
                                assertSameValues(label, expected, actual);
                            } catch (RuntimeException | AssertionError e) {
                                failures.add(label + ": " + e.getMessage());
                            }
                        }
                    }
                }
            }
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    /** Independent adjoint oracle: inclusive/exclusive prefix in the opposite direction. */
    private static INDArray scanAdjoint(DataType type, int axis, boolean exclusive, boolean reverse) {
        double[] values = {1, 2, 3, 4, 5, 6};
        double[] expected = new double[6];
        for (int i = 0; i < values.length; i++) {
            int coordinate = axis == 0 ? i / 3 : axis == 1 ? i % 3 : i;
            for (int j = 0; j < values.length; j++) {
                boolean sameTad = axis == 0 ? i % 3 == j % 3 : axis == 1 ? i / 3 == j / 3 : true;
                int other = axis == 0 ? j / 3 : axis == 1 ? j % 3 : j;
                boolean contributes = reverse ? (exclusive ? other < coordinate : other <= coordinate)
                        : (exclusive ? other > coordinate : other >= coordinate);
                if (sameTad && contributes) expected[i] += values[j];
            }
        }
        return Nd4j.createFromArray(expected).castTo(type).reshape(2, 3);
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void cumsumAdjointAxisTensorWritesBothStridedOutputs(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray base = Nd4j.createFromArray(new double[]{1, 2, 3, 4, 5, 6}).castTo(type).reshape(2, 3);
            for (Layout layout : LAYOUTS) {
                if (layout.minRank > 2) continue;
                for (DataType axisType : new DataType[]{DataType.INT32, DataType.INT64, DataType.FLOAT, DataType.DOUBLE}) {
                    for (boolean exclusive : new boolean[]{false, true}) {
                        for (boolean reverse : new boolean[]{false, true}) {
                            INDArray original = layout.of.apply(base);
                            INDArray gradient = layout.of.apply(base);
                            INDArray axes = Nd4j.createFromArray(-1).castTo(axisType);
                            for (int axis : new int[]{1, 0}) {
                                axes.assign(axis == 1 ? -1 : 0);
                                INDArray parent = Nd4j.valueArrayOf(new long[]{3, 7}, -777, type);
                                INDArray gradX = parent.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 7));
                                INDArray axisParent = Nd4j.valueArrayOf(new long[]{3}, -7, axisType);
                                INDArray gradAxis = axisParent.get(NDArrayIndex.interval(1, 2));
                                DynamicCustomOp scan = DynamicCustomOp.builder("cumsum_bp")
                                        .addInputs(original, axes, gradient).addOutputs(gradX, gradAxis)
                                        .addIntegerArguments(exclusive ? 1 : 0, reverse ? 1 : 0).build();
                                Nd4j.exec(scan);
                                String label = type + " " + axisType + " " + layout.name + " axis=" + axis
                                        + " exclusive=" + exclusive + " reverse=" + reverse;
                                assertEquals(scanAdjoint(type, axis, exclusive, reverse), gradX, label);
                                assertEquals(Nd4j.ones(axisType, 1), gradAxis, label + " axis gradient");
                                for (int row = 0; row < 3; row++) {
                                    for (int column = 0; column < 7; column++) {
                                        if (row == 0 || column % 2 == 0)
                                            assertEquals(-777.0, parent.getDouble(row, column), label + " sentinel");
                                    }
                                }
                                assertEquals(-7, axisParent.getInt(0), label);
                                assertEquals(-7, axisParent.getInt(2), label);
                                INDArray[] allocated = Nd4j.exec(op("cumsum_bp",
                                        new long[]{exclusive ? 1 : 0, reverse ? 1 : 0}, NO_DOUBLES,
                                        original, axes, gradient));
                                assertAllDense(label, allocated);
                                assertSameValues(label, new INDArray[]{scanAdjoint(type, axis, exclusive, reverse),
                                        Nd4j.ones(axisType, 1)}, allocated);
                            }
                        }
                    }
                }
            }
            for (boolean exclusive : new boolean[]{false, true}) {
                for (boolean reverse : new boolean[]{false, true}) {
                    INDArray[] result = Nd4j.exec(op("cumsum_bp",
                            new long[]{exclusive ? 1 : 0, reverse ? 1 : 0}, NO_DOUBLES, base.dup('f'), base.dup('f')));
                    assertSameValues("flattened adjoint", new INDArray[]{scanAdjoint(type, -1, exclusive, reverse)}, result);
                    INDArray axes = Nd4j.createFromArray(0, -1);
                    INDArray[] tensorAxes = Nd4j.exec(op("cumsum_bp",
                            new long[]{exclusive ? 1 : 0, reverse ? 1 : 0}, NO_DOUBLES, base.dup('f'), axes, base.dup('f')));
                    assertSameValues("multi-axis tensor adjoint", new INDArray[]{scanAdjoint(type, -1, exclusive, reverse),
                            Nd4j.ones(axes.dataType(), 2)}, tensorAxes);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void biasAdjointChannelLayoutsHaveIndependentOracles(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray nhwc = Nd4j.createFromArray(new double[]{1, 2, 3, 4, 5, 6}).castTo(type).reshape(1, 2, 3);
            for (boolean nchw : new boolean[]{false, true}) {
                INDArray base = nchw ? nhwc.permute(0, 2, 1) : nhwc;
                INDArray expectedBias = Nd4j.createFromArray(5.0, 7.0, 9.0).castTo(type);
                for (Layout layout : LAYOUTS) {
                    if (layout.minRank > 3) continue;
                    INDArray input = layout.of.apply(base);
                    INDArray grad = layout.of.apply(base);
                    INDArray bias = Nd4j.zeros(type, 3);
                    INDArray[] result = Nd4j.exec(DynamicCustomOp.builder("biasadd_bp")
                            .addInputs(input, bias, grad).addBooleanArguments(nchw).build());
                    assertSameValues(layout.name + " nchw=" + nchw, new INDArray[]{base, expectedBias}, result);
                    assertAllDense("bias adjoint", result);
                }
            }
        }
    }

    private static DataBuffer descriptor(DataType type, long[] shape, long[] stride, char order, long flags) {
        long extras = ArrayOptionsHelper.setOptionBit(0L, type) | flags;
        return Shape.createShapeInformation(shape, stride, 1, order, extras);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void createFromDescriptorKeepsTheShapeTypeAndOrderOnly(Nd4jBackend backend) {
        long copyOffset = ArrayOptionsHelper.ARRAY_COPY_OFFSET_INPUT_0;
        Object[][] cases = {
                // type, shape, descriptor strides, order, flags, expected strides
                {DataType.FLOAT, new long[]{4, 70}, new long[]{140, 2}, 'c', 0L, new long[]{70, 1}},
                {DataType.DOUBLE, new long[]{4, 70}, new long[]{141, 2}, 'c', ArrayOptionsHelper.IS_VIEW,
                        new long[]{70, 1}},
                {DataType.HALF, new long[]{3, 4, 5}, new long[]{5, 15, 1}, 'c', 0L, new long[]{20, 5, 1}},
                {DataType.FLOAT, new long[]{4, 70}, new long[]{2, 9}, 'f', 0L, new long[]{1, 4}},
                {DataType.FLOAT, new long[]{4, 70}, new long[]{1, 4}, 'f', 0L, new long[]{1, 4}},
                {DataType.INT32, new long[]{70}, new long[]{2}, 'c', 0L, new long[]{1}},
                {DataType.BOOL, new long[]{4, 70}, new long[]{70, 1}, 'c', ArrayOptionsHelper.ARRAY_NEEDS_COPY,
                        new long[]{70, 1}},
                {DataType.FLOAT, new long[]{4, 70}, new long[]{70, 1}, 'c', copyOffset | ArrayOptionsHelper.IS_VIEW,
                        new long[]{70, 1}},
                {DataType.FLOAT, new long[]{4, 70}, new long[]{140, 2}, 'c', copyOffset, new long[]{70, 1}},
                {DataType.UINT8, new long[]{2, 3}, new long[]{3, 1}, 'c', 0L, new long[]{3, 1}},
                {DataType.FLOAT, new long[0], new long[0], 'c', ArrayOptionsHelper.IS_VIEW, new long[0]},
        };
        for (Object[] c : cases) {
            DataType type = (DataType) c[0];
            long[] shape = (long[]) c[1];
            long[] stride = (long[]) c[2];
            char order = (Character) c[3];
            long flags = (Long) c[4];
            long[] expectedStride = (long[]) c[5];
            String label = type + " " + Arrays.toString(shape) + " strides " + Arrays.toString(stride) + " order "
                    + order + " flags " + flags;

            INDArray out = Nd4j.createFromDescriptor(descriptor(type, shape, stride, order, flags));
            assertEquals(type, out.dataType(), label + ": type");
            assertArrayEquals(shape, out.shape(), label + ": shape");
            assertEquals(order, out.ordering(), label + ": order");
            assertArrayEquals(expectedStride, out.stride(), label + ": strides");
            assertFalse(ArrayOptionsHelper.arrayNeedsCopy(out.shapeInfoJava()), label + ": needs-copy flag");
            assertEquals(-1, ArrayOptionsHelper.getCopyOffsetInputIndex(out.shapeInfoJava()), label + ": copy-offset flag");
            assertDense(label, out);
        }

        // an empty descriptor stays empty
        INDArray empty = Nd4j.createFromDescriptor(descriptor(DataType.FLOAT, new long[]{0, 5}, new long[]{0, 0}, 'c',
                ArrayOptionsHelper.ATYPE_EMPTY_BIT));
        assertTrue(empty.isEmpty());
        assertArrayEquals(new long[]{0, 5}, empty.shape());
    }

    /** An op's shape function and an array created from it: the outputs of ops over stepped views. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void anArrayCreatedFromAShapeFunctionOverASteppedViewIsDense(Nd4jBackend backend) {
        INDArray base = dense(DataType.FLOAT, 1, 4, 70);
        INDArray stepped = LAYOUTS[3].of.apply(base);
        assertArrayEquals(new long[]{141, 2}, stepped.stride(), "the input is a stepped view");
        for (OpCase c : opCases()) {
            for (DataBuffer descriptor : c.build.apply(stepped).calculateOutputShape()) {
                assertDense(c.name, Nd4j.createFromDescriptor(descriptor));
            }
        }
    }

    /** Populate the owner, not the view: a broken assignment cannot hide the read-offset regression. */
    @ParameterizedTest
    @MethodSource("configs")
    public void legacyArithmeticPreservesOffsetViewsAndOwnerCoherence(Nd4jBackend backend) {
        for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (char order : new char[]{'c', 'f'}) {
                double[] ownerValues = new double[21];
                for (int i = 0; i < ownerValues.length; i++) ownerValues[i] = i;
                INDArray owner = Nd4j.createFromArray(ownerValues).castTo(dtype).reshape(3, 7).dup(order);
                INDArray view = owner.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 7));
                String label = dtype + " " + order;
                assertSame(owner.data(), view.data(), label);
                assertTrue(view.offset() > 0, label);
                double[] values = {8, 10, 12, 15, 17, 19};
                assertSmallViewValues(label + " input", view, values);
                assertSmallViewValues(label + " negation", Transforms.neg(view, true),
                        new double[]{-8, -10, -12, -15, -17, -19});
                assertSmallViewValues(label + " scalar", view.add(2), new double[]{10, 12, 14, 17, 19, 21});
                assertArrayEquals(new double[]{30, 51}, view.sum(1).toDoubleVector(), 0.0, label);
                assertArrayEquals(new double[]{23, 27, 31}, view.sum(0).toDoubleVector(), 0.0, label);
                assertSmallOwnerValues(label + " read-only operations", owner, ownerValues);

                // A device write through the view must update its owner, including its coherence state.
                view.addi(2);
                for (int r = 1; r < 3; r++)
                    for (int c = 1; c < 7; c += 2) ownerValues[r * 7 + c] += 2;
                assertSmallOwnerValues(label + " in-place write", owner, ownerValues);

                // After reading the owner, a host write must be visible to the next view execution.
                owner.putScalar(1, 1, -5);
                ownerValues[8] = -5;
                assertSmallViewValues(label + " owner-to-view update", view.add(1),
                        new double[]{-4, 13, 15, 18, 20, 22});
                assertSmallOwnerValues(label + " owner sentinels", owner, ownerValues);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void legacyAssignmentAndCastPreserveSteppedViewDestinations(Nd4jBackend backend) {
        for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (char order : new char[]{'c', 'f'}) {
                double[] ownerValues = new double[21];
                for (int i = 0; i < ownerValues.length; i++) ownerValues[i] = i;
                INDArray owner = Nd4j.createFromArray(ownerValues).castTo(dtype).reshape(3, 7).dup(order);
                INDArray view = owner.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 7));
                String label = dtype + " " + order;
                double[] replacements = {-1, -2, -3, -4, -5, -6};
                view.assign(Nd4j.createFromArray(replacements).castTo(dtype).reshape(2, 3));
                for (int i = 0; i < replacements.length; i++)
                    ownerValues[(1 + i / 3) * 7 + 1 + 2 * (i % 3)] = replacements[i];
                assertSmallOwnerValues(label + " array assignment", owner, ownerValues);
                assertSmallViewValues(label + " copy", view.dup('c'), replacements);
                assertSmallViewValues(label + " cast", view.castTo(dtype == DataType.FLOAT
                        ? DataType.DOUBLE : DataType.FLOAT), replacements);

                view.assign(7);
                for (int r = 1; r < 3; r++)
                    for (int c = 1; c < 7; c += 2) ownerValues[r * 7 + c] = 7;
                assertSmallOwnerValues(label + " scalar assignment and untouched surroundings", owner, ownerValues);
            }
        }
    }

    private static void assertSmallViewValues(String label, INDArray actual, double[] expected) {
        assertArrayEquals(new long[]{2, 3}, actual.shape(), label);
        for (int i = 0; i < expected.length; i++)
            assertEquals(expected[i], actual.getDouble(i / 3, i % 3), 0.0, label + " element " + i);
    }

    private static void assertSmallOwnerValues(String label, INDArray owner, double[] expected) {
        for (int i = 0; i < expected.length; i++)
            assertEquals(expected[i], owner.getDouble(i / 7, i % 7), 0.0, label + " owner element " + i);
    }

    // --------------------------------------------------------------------------- SameDiff, plan on and off

    private interface SdBuilder {
        SDVariable build(SameDiff sd, SDVariable in, int rank);
    }

    private static final class SdCase {
        final String name;
        final SdBuilder build;

        SdCase(String name, SdBuilder build) {
            this.name = name;
            this.build = build;
        }
    }

    private static long[] reversedDimensions(int rank) {
        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++)
            dims[i] = rank - 1 - i;
        return dims;
    }

    private static List<SdCase> sdCases() {
        List<SdCase> cases = new ArrayList<>();
        cases.add(new SdCase("softmax", (sd, in, rank) -> sd.nn().softmax("out", in, rank - 1)));
        cases.add(new SdCase("tanh", (sd, in, rank) -> sd.math().tanh("out", in)));
        cases.add(new SdCase("relu", (sd, in, rank) -> sd.nn().relu("out", in, 0.0)));
        cases.add(new SdCase("cumsum", (sd, in, rank) -> sd.cumsum("out", in, false, false, rank - 1)));
        cases.add(new SdCase("reverse", (sd, in, rank) -> sd.reverse("out", in, rank - 1)));
        cases.add(new SdCase("clipByValue", (sd, in, rank) -> sd.math().clipByValue("out", in, -0.25, 0.25)));
        cases.add(new SdCase("cast", (sd, in, rank) -> in.castTo("out", DataType.DOUBLE)));
        // ops that may make views of their input: a stepped input cannot be viewed, the plan allocates their output
        cases.add(new SdCase("permute then add", (sd, in, rank) ->
                sd.permute(in, reversedDimensions(rank)).add("out", 1.0)));
        cases.add(new SdCase("expand_dims then mul", (sd, in, rank) -> sd.expandDims(in, 0).mul("out", 2.0)));
        cases.add(new SdCase("reshape then add", (sd, in, rank) ->
                sd.reshape(in, length(in.getShape())).add("out", 1.0)));
        // view ops over the outputs of view ops: the plan infers the shapes of the whole graph from placeholders it
        // publishes for the views before anything runs, and a view is only made over a contiguous input
        cases.add(new SdCase("permute twice then add", (sd, in, rank) ->
                sd.permute(sd.permute(in, reversedDimensions(rank)), reversedDimensions(rank)).add("out", 1.0)));
        cases.add(new SdCase("permute then reshape then add", (sd, in, rank) ->
                sd.reshape(sd.permute(in, reversedDimensions(rank)), length(in.getShape())).add("out", 1.0)));
        cases.add(new SdCase("slice then reshape then add", (sd, in, rank) -> {
            int[] size = firstHalfOfLastDimension(in.getShape());
            long[] sliced = new long[size.length];
            for (int i = 0; i < size.length; i++)
                sliced[i] = size[i];
            return sd.reshape(sd.slice(in, new int[rank], size), length(sliced)).add("out", 1.0);
        }));
        cases.add(new SdCase("slice then expand_dims then mul", (sd, in, rank) ->
                sd.expandDims(sd.slice(in, new int[rank], firstHalfOfLastDimension(in.getShape())), 0)
                        .mul("out", 2.0)));
        return cases;
    }

    private static long length(long[] shape) {
        long length = 1;
        for (long dim : shape)
            length *= dim;
        return length;
    }

    /** The extent of the first half of the last dimension (at least one element), the rest of the shape whole. */
    private static int[] firstHalfOfLastDimension(long[] shape) {
        int[] size = new int[shape.length];
        for (int i = 0; i < size.length; i++)
            size[i] = (int) shape[i];
        size[size.length - 1] = Math.max(1, size[size.length - 1] / 2);
        return size;
    }

    private static INDArray sdRun(SdCase c, INDArray input) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", input.dataType(), input.shape());
        c.build.build(sd, in, input.rank());
        return sd.output(Collections.singletonMap("in", input), "out").get("out");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameDiffOutputsDoNotDependOnTheInputLayout(Nd4jBackend backend) {
        boolean planBefore = InferenceSession.isDynamicShapePlanEnabled();
        List<String> failures = new ArrayList<>();
        try {
            for (boolean plan : new boolean[]{false, true}) {
                InferenceSession.setDynamicShapePlanEnabled(plan);
                for (long[] shape : SHAPES) {
                    INDArray base = dense(DataType.FLOAT, 1, shape);
                    for (SdCase c : sdCases()) {
                        INDArray expected;
                        try {
                            expected = sdRun(c, base.dup('c')).dup('c');
                        } catch (RuntimeException e) {
                            failures.add(c.name + " over a dense C-order input, plan " + plan + ": " + e);
                            continue;
                        }
                        for (Layout layout : LAYOUTS) {
                            if (shape.length < layout.minRank)
                                continue;
                            String label = c.name + " over a " + layout.name + " " + Arrays.toString(shape)
                                    + " input, dynamic shape plan " + (plan ? "on" : "off");
                            try {
                                INDArray actual = sdRun(c, layout.of.apply(base));
                                Nd4j.getExecutioner().commit();
                                assertSameValues(label, new INDArray[]{expected}, new INDArray[]{actual});
                                assertTrue(packed(actual.shape(), actual.stride(), actual.ordering()),
                                        () -> label + ": the output's strides are " + Arrays.toString(actual.stride()));
                            } catch (RuntimeException | AssertionError e) {
                                failures.add(label + ": " + e.getMessage());
                            }
                        }
                    }
                }
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(planBefore);
        }
        assertTrue(failures.isEmpty(), failures.size() + " failures:\n" + String.join("\n", failures));
    }

    // -------------------------------------------------------------- outputs that are views stay views

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void permuteOfAnyLayoutStaysAViewOfItsInput(Nd4jBackend backend) {
        for (Layout layout : LAYOUTS) {
            if (layout.minRank > 2)
                continue;
            INDArray base = dense(DataType.FLOAT, 1, 4, 70);
            INDArray x = layout.of.apply(base);
            INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("permute").addInputs(x).addIntegerArguments(1, 0).build());
            Nd4j.getExecutioner().commit();
            assertEquals(1, out.length, layout.name);
            assertArrayEquals(new long[]{70, 4}, out[0].shape(), layout.name);
            assertTrue(base.dup('c').permute(1, 0).equalsWithEps(out[0], 1e-6), layout.name + ": values");
            // a view shares the input's storage: a write through it shows in the input
            out[0].putScalar(0, 0, 42.0f);
            assertEquals(42.0f, x.getFloat(0, 0), 0.0f, layout.name + ": the output of permute is no view of its input");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void reshapeNoCopyViewsWhatItCanAndAllocatesDenselyWhatItCannot(Nd4jBackend backend) {
        INDArray base = dense(DataType.FLOAT, 1, 4, 70);
        INDArray stepped = LAYOUTS[3].of.apply(base);

        // [4, 70] with strides [141, 2] splits its first axis without a copy
        INDArray[] view = Nd4j.exec(new ReshapeNoCopy(stepped, new long[]{2, 2, 70}, null));
        Nd4j.getExecutioner().commit();
        // Check aliasing independently of putScalar: a bad write offset is not a failure to create a view.
        assertSame(stepped.data(), view[0].data(), "reshape shares the input buffer");
        assertEquals(stepped.offset(), view[0].offset(), "reshape preserves the input offset");
        assertArrayEquals(new long[]{282, 141, 2}, view[0].stride(), "split-axis view strides");
        assertTrue(view[0].isView());
        assertTrue(base.dup('c').reshape('c', 2, 2, 70).equalsWithEps(view[0], 1e-6), "values of the view");
        view[0].putScalar(0, 0, 0, 7.0f);
        assertEquals(7.0f, stepped.getFloat(0, 0), 0.0f, "a reshape that can be a view shares its input's storage");
        assertTrue(Float.isNaN(stepped.data().getFloat(0)), "the element before the offset view is untouched");
        view[0].putScalar(1, 1, 69, 8.0f);
        assertEquals(8.0f, stepped.getFloat(3, 69), 0.0f, "rank-three writes use the split-axis strides");
        stepped.putScalar(1, 1, 9.0f);
        assertEquals(9.0f, view[0].getFloat(0, 1, 1), 0.0f, "input writes are visible through the reshape");

        // The reference must include the alias writes before checking the copying reshape.
        base.putScalar(0, 0, 7.0f);
        base.putScalar(3, 69, 8.0f);
        base.putScalar(1, 1, 9.0f);

        // flattening it needs a copy: the output is a new dense array
        INDArray[] copy = Nd4j.exec(new ReshapeNoCopy(stepped, new long[]{280}, null));
        Nd4j.getExecutioner().commit();
        assertEquals(280, copy[0].length());
        assertTrue(base.dup('c').reshape('c', new long[]{280}).equalsWithEps(copy[0], 1e-6), "values of the copy");
        assertDense("reshape_no_copy that copies", copy[0]);
        assertFalse(copy[0].data() == stepped.data(), "a copying reshape owns separate storage");
        copy[0].putScalar(0L, -7.0);
        assertEquals(7.0f, stepped.getFloat(0, 0), 0.0f, "a copying reshape must not change its input");
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void singletonAxisCopiesPreserveCoordinatesAcrossOrders(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray parent = Nd4j.create(type, new long[]{2, 7}, 'c');
            INDArray stepped = parent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 7));
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++) stepped.putScalar(i, j, 10 * i + j + 1);
            for (INDArray input : new INDArray[]{stepped.dup('c'), stepped.dup('f'), stepped}) {
                for (int axis : new int[]{0, 1, -1}) {
                    int normalized = axis < 0 ? axis + 3 : axis;
                    long[] shape = normalized == 0 ? new long[]{1, 2, 3}
                            : normalized == 1 ? new long[]{2, 1, 3} : new long[]{2, 3, 1};
                    for (char outputOrder : new char[]{'c', 'f'}) {
                        String label = type + " " + input.ordering() + " axis=" + axis + " out=" + outputOrder;
                        INDArray expanded = Nd4j.create(type, shape, outputOrder);
                        Nd4j.exec(DynamicCustomOp.builder("expand_dims").addInputs(input)
                                .addOutputs(expanded).addIntegerArguments(axis).build());
                        INDArray squeezed = Nd4j.create(type, new long[]{2, 3}, outputOrder);
                        Nd4j.exec(DynamicCustomOp.builder("squeeze").addInputs(expanded)
                                .addOutputs(squeezed).addIntegerArguments(normalized).build());
                        Nd4j.getExecutioner().commit();
                        for (int i = 0; i < 2; i++) {
                            for (int j = 0; j < 3; j++) {
                                long[] coordinates = normalized == 0 ? new long[]{0, i, j}
                                        : normalized == 1 ? new long[]{i, 0, j} : new long[]{i, j, 0};
                                assertEquals(10 * i + j + 1, expanded.getDouble(coordinates), 0.0, label);
                                assertEquals(10 * i + j + 1, squeezed.getDouble(i, j), 0.0, label);
                            }
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void copyingReshapeHonorsRequestedTraversalAndOutputStrides(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray parent = Nd4j.create(type, new long[]{2, 7}, 'c');
            INDArray stepped = parent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 7));
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++) stepped.putScalar(i, j, 10 * i + j + 1);
            for (INDArray input : new INDArray[]{stepped.dup('c'), stepped.dup('f'), stepped}) {
                for (char requestedOrder : new char[]{'c', 'f'}) {
                    for (char outputOrder : new char[]{'c', 'f'}) {
                        String label = type + " input=" + input.ordering() + " requested=" + requestedOrder
                                + " output=" + outputOrder;
                        INDArray output = Nd4j.create(type, new long[]{3, 2}, outputOrder);
                        Nd4j.exec(new ReshapeNoCopy(input, new long[]{3, -1}, output, requestedOrder));
                        assertReshapeTraversal(output, requestedOrder, label + " iArgs");
                        INDArray tensorOutput = Nd4j.create(type, new long[]{3, 2}, outputOrder);
                        Nd4j.exec(DynamicCustomOp.builder("reshape_no_copy")
                                .addInputs(input, Nd4j.createFromArray(3L, 2L)).addOutputs(tensorOutput)
                                .addIntegerArguments(-(long) requestedOrder).build());
                        assertReshapeTraversal(tensorOutput, requestedOrder, label + " shape tensor");
                    }
                    for (boolean shapeTensor : new boolean[]{false, true}) {
                        INDArray outputParent = Nd4j.valueArrayOf(new long[]{3, 5}, -999., type);
                        INDArray outputView = outputParent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 5));
                        if (shapeTensor) {
                            Nd4j.exec(DynamicCustomOp.builder("reshape_no_copy")
                                    .addInputs(input, Nd4j.createFromArray(3L, 2L)).addOutputs(outputView)
                                    .addIntegerArguments(-(long) requestedOrder).build());
                        } else {
                            Nd4j.exec(new ReshapeNoCopy(input, new long[]{3, -1}, outputView, requestedOrder));
                        }
                        assertReshapeTraversal(outputView, requestedOrder, "stepped output shapeTensor=" + shapeTensor);
                        for (int row = 0; row < 3; row++)
                            for (int col = 0; col < 5; col += 2)
                                assertEquals(-999., outputParent.getDouble(row, col), 0.,
                                        "reshape wrote outside its output view");
                    }
                    INDArray allocated = Nd4j.exec(new ReshapeNoCopy(input, new long[]{3, 2}, null, requestedOrder))[0];
                    assertReshapeTraversal(allocated, requestedOrder, "allocated " + requestedOrder);
                }
                INDArray defaultOrder = Nd4j.exec(DynamicCustomOp.builder("reshape_no_copy")
                        .addInputs(input, Nd4j.createFromArray(3L, 2L)).build())[0];
                assertReshapeTraversal(defaultOrder, 'c', "default shape-tensor order");
                for (char order : new char[]{'c', 'f'}) {
                    INDArray zeroDimensionOutput = Nd4j.create(type, new long[]{2, 3}, order);
                    Nd4j.exec(new ReshapeNoCopy(input, new long[]{0, -1}, zeroDimensionOutput, order));
                    Nd4j.getExecutioner().commit();
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 3; j++)
                            assertEquals(10 * i + j + 1, zeroDimensionOutput.getDouble(i, j), 0.0,
                                    "zero copies input dimension " + type + " " + order);
                }
            }
        }
    }

    private static void assertReshapeTraversal(INDArray output, char order, String label) {
        Nd4j.getExecutioner().commit();
        assertArrayEquals(new long[]{3, 2}, output.shape(), label);
        for (int flat = 0; flat < 6; flat++) {
            int inputRow = order == 'f' ? flat % 2 : flat / 3;
            int inputColumn = order == 'f' ? flat / 2 : flat % 3;
            int outputRow = order == 'f' ? flat % 3 : flat / 2;
            int outputColumn = order == 'f' ? flat / 3 : flat % 2;
            assertEquals(10 * inputRow + inputColumn + 1, output.getDouble(outputRow, outputColumn), 0.0, label);
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void putScalarOnReshapedScalarViewsKeepsTheOffset(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            for (char order : new char[]{'c', 'f'}) {
                for (long[] shape : new long[][]{new long[0], {1}, {1, 1}}) {
                    INDArray parent = Nd4j.zeros(type, 3);
                    INDArray selected = parent.get(NDArrayIndex.interval(1, 2));
                    INDArray scalar = Nd4j.exec(new ReshapeNoCopy(selected, shape, null, order))[0];
                    String label = type + " " + order + " scalar view " + Arrays.toString(shape);
                    assertArrayEquals(shape, scalar.shape(), label);
                    assertSame(parent.data(), scalar.data(), label);
                    assertEquals(1, scalar.offset(), label);
                    scalar.putScalar(0L, 7.0);
                    assertEquals(7.0, parent.getDouble(1), 0.0, label + ": parent element");
                    assertEquals(7.0, scalar.getDouble(0L), 0.0, label + ": read after write");
                    assertEquals(0.0, parent.getDouble(0), 0.0, label + ": before view");
                    assertEquals(0.0, parent.getDouble(2), 0.0, label + ": after view");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void putScalarOnOffsetViewsWritesTheParentAtAnyRank(Nd4jBackend backend) {
        INDArray parent3 = Nd4j.zeros(DataType.FLOAT, 3, 2, 2);
        INDArray view3 = parent3.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(), NDArrayIndex.all());
        assertTrue(view3.offset() > 0);
        view3.putScalar(0, 1, 1, 3.0);
        assertEquals(3.0, parent3.getDouble(1, 1, 1), 0.0);
        assertEquals(0.0, parent3.getDouble(0, 1, 1), 0.0);

        INDArray parent4 = Nd4j.zeros(DataType.FLOAT, 3, 2, 2, 2);
        INDArray view4 = parent4.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all());
        assertTrue(view4.offset() > 0);
        view4.putScalar(1, 0, 1, 1, 4.0);
        assertEquals(4.0, parent4.getDouble(2, 0, 1, 1), 0.0);
        assertEquals(0.0, parent4.getDouble(1, 0, 1, 1), 0.0);

        INDArray parent5 = Nd4j.zeros(DataType.FLOAT, 3, 2, 2, 2, 2);
        INDArray view5 = parent5.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        assertTrue(view5.offset() > 0);
        view5.putScalar(new long[]{0, 1, 0, 1, 1}, 5.0);
        assertEquals(5.0, parent5.getDouble(1, 1, 0, 1, 1), 0.0);
        assertEquals(0.0, parent5.getDouble(0, 1, 0, 1, 1), 0.0);
        view5.putScalar(new int[]{1, 0, 1, 0, 0}, 6.0);
        assertEquals(6.0, parent5.getDouble(2, 0, 1, 0, 0), 0.0);
        assertEquals(0.0, parent5.getDouble(1, 0, 1, 0, 0), 0.0);
    }

    /** Independent references for Vulkan's coordinate-driven recipes, not a second backend path. */
    @ParameterizedTest
    @MethodSource("configs")
    public void coordinateRecipesRespectPayloadAndOutputStrides(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray a = Nd4j.createFromArray(1., 2., 3., 4., 5., 6.).castTo(type).reshape(2, 3);
            INDArray b = Nd4j.createFromArray(7., 8., 9., 2., 3., 4.).castTo(type).reshape(2, 3);
            INDArray cross = Nd4j.createFromArray(-6., 12., -6., 2., -4., 2.).castTo(type).reshape(2, 3);
            INDArray reversed = Nd4j.createFromArray(9., 8., 7., 4., 3., 2.).castTo(type).reshape(2, 3);
            for (Layout layout : LAYOUTS) {
                if (layout.minRank > 2) continue;
                INDArray x = layout.of.apply(a);
                INDArray y = LAYOUTS[3].of.apply(b);
                assertSameValues(layout.name + " cross", new INDArray[]{cross},
                        Nd4j.exec(op("cross", NO_INTS, NO_DOUBLES, x, y)));
                assertSameValues(layout.name + " reverse_bp", new INDArray[]{reversed},
                        Nd4j.exec(op("reverse_bp", new long[]{-1}, NO_DOUBLES, x, y)));
                for (boolean reverse : new boolean[]{false, true}) {
                    for (boolean exclusive : new boolean[]{false, true}) {
                        INDArray expected = Nd4j.create(type, 2, 3);
                        for (int row = 0; row < 2; ++row) {
                            double sum = 0;
                            for (int step = 0; step < 3; ++step) {
                                int col = reverse ? 2 - step : step;
                                double next = sum + (row * 3 + col + 1);
                                expected.putScalar(row, col, exclusive ? sum : next);
                                sum = next;
                            }
                        }
                        // Caller-provided output is a nonzero-offset stepped view.
                        INDArray parent = Nd4j.valueArrayOf(new long[]{2, 7}, -999., type);
                        INDArray output = parent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 7));
                        DynamicCustomOp scan = op("cumsum", new long[]{exclusive ? 1 : 0, reverse ? 1 : 0, -1},
                                NO_DOUBLES, x);
                        scan.addOutputArgument(output);
                        Nd4j.exec(scan);
                        Nd4j.getExecutioner().commit();
                        assertTrue(expected.equalsWithEps(output, 0), layout.name + " scan flags " + exclusive + "/" + reverse);
                        for (int row = 0; row < 2; ++row)
                            for (int col = 0; col < 7; col += 2)
                                assertEquals(-999., parent.getDouble(row, col), 0., "scan wrote outside its output view");
                    }
                }
            }
            // Argument-free scan covers the whole tensor in logical C order.
            INDArray all = Nd4j.exec(op("cumsum", new long[]{0, 0}, NO_DOUBLES, a.dup('f')))[0];
            INDArray allExpected = Nd4j.createFromArray(1., 3., 6., 10., 15., 21.).castTo(type).reshape(2, 3);
            assertTrue(allExpected.equalsWithEps(all, 0), "linear F-order scan");
            INDArray inplace = LAYOUTS[3].of.apply(a);
            DynamicCustomOp inplaceScan = op("cumsum", new long[]{1, 1, 1}, NO_DOUBLES, inplace);
            inplaceScan.addOutputArgument(inplace);
            Nd4j.exec(inplaceScan);
            INDArray inplaceExpected = Nd4j.createFromArray(5., 3., 0., 11., 6., 0.).castTo(type).reshape(2, 3);
            assertTrue(inplaceExpected.equalsWithEps(inplace, 0), "reverse exclusive scan in place");
            INDArray reverseInplace = LAYOUTS[3].of.apply(a);
            DynamicCustomOp reverse = op("reverse", new long[]{0, 1}, NO_DOUBLES, reverseInplace);
            reverse.addOutputArgument(reverseInplace);
            Nd4j.exec(reverse);
            INDArray reverseExpected = Nd4j.createFromArray(6., 5., 4., 3., 2., 1.).castTo(type).reshape(2, 3);
            assertTrue(reverseExpected.equalsWithEps(reverseInplace, 0), "reverse uses one owner per pair");
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void shapeVectorBroadcastRightAlignsAndKeepsZeroDimensions(Nd4jBackend backend) {
        for (DataType type : new DataType[]{DataType.INT32, DataType.LONG}) {
            INDArray x = Nd4j.createFromArray(1, 3).castTo(type);
            INDArray y = Nd4j.createFromArray(2, 0, 1).castTo(type);
            INDArray expected = Nd4j.createFromArray(2, 0, 3).castTo(type);
            for (boolean swapped : new boolean[]{false, true}) {
                INDArray[] actual = Nd4j.exec(op("broadcast_dynamic_shape", NO_INTS, NO_DOUBLES,
                        swapped ? y : x, swapped ? x : y));
                assertSameValues(type + " shape vector broadcast", new INDArray[]{expected}, actual);
            }
        }
    }

    /** Small deterministic proof of the requested-order reshape and frozen-IArg slice prepass. */
    @ParameterizedTest
    @MethodSource("configs")
    public void sameDiffReshapeSliceAndScalarUseLogicalCoordinates(Nd4jBackend backend) {
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        INDArray base = Nd4j.createFromArray(1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.)
                .castTo(DataType.FLOAT).reshape(2, 2, 3);
        INDArray reshaped = Nd4j.createFromArray(2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f);
        INDArray sliced = Nd4j.createFromArray(3.f, 12.f, 21.f, 30.f);
        INDArray permuted = Nd4j.createFromArray(2.f, 8.f, 5.f, 11.f, 3.f, 9.f, 6.f, 12.f, 4.f, 10.f, 7.f, 13.f);
        try {
            for (boolean plan : new boolean[]{false, true}) {
                InferenceSession.setDynamicShapePlanEnabled(plan);
                for (Layout layout : LAYOUTS) {
                    INDArray input = layout.of.apply(base);
                    INDArray actual = sdRun(new SdCase("reshape scalar",
                            (sd, in, rank) -> sd.reshape(in, 12).add("out", 1.)), input);
                    assertTrue(reshaped.equalsWithEps(actual, 0), layout.name + " reshape, plan=" + plan);
                    INDArray permute = sdRun(new SdCase("permute reshape scalar",
                            (sd, in, rank) -> sd.reshape(sd.permute(in, 2, 1, 0), 12).add("out", 1.)), input);
                    assertTrue(permuted.equalsWithEps(permute, 0), layout.name + " permute reshape, plan=" + plan);
                    INDArray slice = sdRun(new SdCase("slice scalar",
                            (sd, in, rank) -> sd.reshape(sd.slice(in, new int[]{0, 0, 0}, new int[]{2, 2, 1}), 4)
                                    .mul("out", 3.)), input);
                    assertTrue(sliced.equalsWithEps(slice, 0), layout.name + " slice, plan=" + plan);
                }
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    // ------------------------------------------------------------------------------- a helper found on the way

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void crossAllocatesItsOutputFromTheShapeDescriptor(Nd4jBackend backend) {
        INDArray a = dense(DataType.DOUBLE, 1, 4, 3);
        INDArray b = dense(DataType.DOUBLE, 2, 4, 3);
        INDArray expected = Nd4j.create(DataType.DOUBLE, 4, 3);
        for (int r = 0; r < 4; r++) {
            double a0 = a.getDouble(r, 0), a1 = a.getDouble(r, 1), a2 = a.getDouble(r, 2);
            double b0 = b.getDouble(r, 0), b1 = b.getDouble(r, 1), b2 = b.getDouble(r, 2);
            expected.putScalar(r, 0, a1 * b2 - a2 * b1);
            expected.putScalar(r, 1, a2 * b0 - a0 * b2);
            expected.putScalar(r, 2, a0 * b1 - a1 * b0);
        }
        INDArray out = Transforms.cross(a, b);
        assertArrayEquals(new long[]{4, 3}, out.shape());
        assertEquals(DataType.DOUBLE, out.dataType());
        assertTrue(expected.equalsWithEps(out, 1e-12), () -> out + " vs " + expected);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
