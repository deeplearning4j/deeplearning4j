/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.Roll;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@NativeTag
@Tag(TagNames.FULL_CI)
class RollRegressionTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    void testLinearShiftNormalizationAndOddLength(Nd4jBackend backend) {
        INDArray input = Nd4j.arange(8).castTo(DataType.INT64);
        assertEquals(Nd4j.createFromArray(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L), input, "linear test fixture");
        long[][] cases = {
                {3, 5, 6, 7, 0, 1, 2, 3, 4},
                {-3, 3, 4, 5, 6, 7, 0, 1, 2},
                {11, 5, 6, 7, 0, 1, 2, 3, 4},
                {-13, 5, 6, 7, 0, 1, 2, 3, 4},
                {0, 0, 1, 2, 3, 4, 5, 6, 7}
        };

        for (long[] testCase : cases) {
            INDArray actual = Nd4j.exec(new Roll(input, Math.toIntExact(testCase[0])))[0];
            INDArray expected = Nd4j.createFromArray(
                    testCase[1], testCase[2], testCase[3], testCase[4],
                    testCase[5], testCase[6], testCase[7], testCase[8]);
            assertEquals(expected, actual, "shift=" + testCase[0]);
        }

        INDArray tensorShiftActual = Nd4j.exec(DynamicCustomOp.builder("roll")
                .addInputs(input, Nd4j.scalar(DataType.INT64, 3))
                .build())[0];
        assertEquals(Nd4j.createFromArray(5L, 6L, 7L, 0L, 1L, 2L, 3L, 4L), tensorShiftActual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    void testMultiAxisLongArgumentsDoNotCancel(Nd4jBackend backend) {
        INDArray input = Nd4j.arange(12).reshape(3, 4).castTo(DataType.DOUBLE);
        INDArray shifts = Nd4j.createFromArray(1L, -1L);
        INDArray axes = Nd4j.createFromArray(0L, -1L);

        INDArray actual = Nd4j.exec(new Roll(input, shifts, axes))[0];
        INDArray expected = Nd4j.createFromArray(
                9.0, 10.0, 11.0, 8.0,
                1.0, 2.0, 3.0, 0.0,
                5.0, 6.0, 7.0, 4.0).reshape(3, 4);
        assertEquals(expected, actual);

        INDArray duplicateAxisActual = Nd4j.exec(new Roll(
                input, Nd4j.createFromArray(1L, 2L), Nd4j.createFromArray(1L, 1L)))[0];
        INDArray duplicateAxisExpected = Nd4j.createFromArray(
                1.0, 2.0, 3.0, 0.0,
                5.0, 6.0, 7.0, 4.0,
                9.0, 10.0, 11.0, 8.0).reshape(3, 4);
        assertEquals(duplicateAxisExpected, duplicateAxisActual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    void testNonContiguousAndFortranInputs(Nd4jBackend backend) {
        INDArray base = Nd4j.arange(24).reshape(4, 6).castTo(DataType.FLOAT);
        INDArray view = base.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 6));
        assertTrue(view.isView());

        INDArray viewActual = Nd4j.exec(new Roll(
                view, Nd4j.scalar(DataType.INT64, 1), Nd4j.scalar(DataType.INT64, 1)))[0];
        INDArray viewExpected = Nd4j.createFromArray(
                4f, 0f, 2f,
                10f, 6f, 8f,
                16f, 12f, 14f,
                22f, 18f, 20f).reshape(4, 3);
        assertEquals(viewExpected, viewActual);

        INDArray fortran = Nd4j.arange(8).reshape(2, 4).castTo(DataType.FLOAT).dup('f');
        INDArray fortranActual = Nd4j.exec(new Roll(fortran, 3))[0];
        INDArray fortranExpected = Nd4j.createFromArray(
                5f, 6f, 7f, 0f,
                1f, 2f, 3f, 4f).reshape(2, 4);
        assertEquals(fortranExpected, fortranActual);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    void testInPlaceContiguousAndNonContiguous(Nd4jBackend backend) {
        INDArray linear = Nd4j.arange(8).castTo(DataType.INT32);
        DynamicCustomOp linearOp = DynamicCustomOp.builder("roll")
                .addInputs(linear)
                .addIntegerArguments(3)
                .callInplace(true)
                .build();
        Nd4j.exec(linearOp);
        assertEquals(Nd4j.createFromArray(5, 6, 7, 0, 1, 2, 3, 4), linear);

        INDArray booleans = Nd4j.createFromArray(true, false, false, true, false, true, true, false);
        DynamicCustomOp booleanOp = DynamicCustomOp.builder("roll")
                .addInputs(booleans)
                .addIntegerArguments(3)
                .callInplace(true)
                .build();
        Nd4j.exec(booleanOp);
        assertEquals(Nd4j.createFromArray(true, true, false, true, false, false, true, false), booleans);

        INDArray base = Nd4j.arange(24).reshape(4, 6).castTo(DataType.INT64);
        INDArray view = base.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 6));
        DynamicCustomOp viewOp = DynamicCustomOp.builder("roll")
                .addInputs(view, Nd4j.scalar(DataType.INT64, 1), Nd4j.scalar(DataType.INT64, 1))
                .callInplace(true)
                .build();
        Nd4j.exec(viewOp);

        INDArray expectedBase = Nd4j.createFromArray(
                4L, 1L, 0L, 3L, 2L, 5L,
                10L, 7L, 6L, 9L, 8L, 11L,
                16L, 13L, 12L, 15L, 14L, 17L,
                22L, 19L, 18L, 21L, 20L, 23L).reshape(4, 6);
        assertEquals(expectedBase, base);
    }
}
