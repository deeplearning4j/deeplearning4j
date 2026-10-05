/*
 * SPDX-License-Identifier: Apache-2.0
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Small regressions for validation at the legacy op authority, not only at construction. */
@NativeTag
public class LegacyOpContextValidationTest extends BaseNd4jTestWithBackends {
    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void broadcastAxesAreValidatedBeforeShapeIndexing(Nd4jBackend backend) {
        INDArray x = Nd4j.ones(DataType.FLOAT, 3, 4, 2);
        INDArray y = Nd4j.ones(DataType.FLOAT, 3, 2);
        for (long[] axes : new long[][]{{3}, {-4}, {0, 0}, {0, -3}, {Integer.MAX_VALUE}}) {
            assertThrows(IllegalStateException.class, () -> new BroadcastMulOp(x, y, x, axes));
        }
        long[] axes = {2, -3};
        BroadcastMulOp op = new BroadcastMulOp(x, y, x, axes);
        assertTrue(op.validateDataTypes(false));
        assertArrayEquals(new long[]{2, -3}, axes, "Construction must not mutate caller-owned axes");
        assertArrayEquals(new long[]{0, 2}, op.dimensions().toLongVector());

        // Equal-rank Y may have singleton axes outside the selected TAD, not arbitrary sizes.
        INDArray fullY = Nd4j.ones(DataType.FLOAT, 3, 4, 2);
        assertThrows(IllegalStateException.class, () -> new BroadcastMulOp(x, fullY, x, 0, 2));
        assertTrue(new BroadcastMulOp(x, Nd4j.ones(DataType.FLOAT, 3, 1, 2), x, 0, 2)
                .validateDataTypes(false));
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void mutatedBroadcastOperandsAreRejectedBeforeExecution(Nd4jBackend backend) {
        INDArray x = Nd4j.ones(DataType.FLOAT, 3, 4, 2);
        INDArray before = x.dup();
        BroadcastMulOp op = new BroadcastMulOp(x, Nd4j.ones(DataType.FLOAT, 3, 2), x, 0, 2);
        op.setY(Nd4j.ones(DataType.FLOAT, 2, 3));
        assertThrows(IllegalStateException.class, () -> Nd4j.getExecutioner().exec(op));
        assertEquals(before, x);
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void dimensionMutationUpdatesTheNativeDimensionArray(Nd4jBackend backend) {
        INDArray x = Nd4j.ones(DataType.FLOAT, 2, 2);
        BroadcastMulOp op = new BroadcastMulOp(x, Nd4j.ones(DataType.FLOAT, 2), x, 0);
        op.setDimension(-1);
        assertTrue(op.validateDataTypes(false));
        assertArrayEquals(new long[]{1}, op.dimensions().toLongVector());
        op.getDimension()[0] = 0;
        assertTrue(op.validateDataTypes(false));
        assertArrayEquals(new long[]{0}, op.dimensions().toLongVector());
        assertThrows(IllegalStateException.class, () -> op.setDimension(0, 0));
        assertTrue(op.validateDataTypes(false));
        assertArrayEquals(new long[]{0}, op.dimensions().toLongVector());
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void broadcastValidationUsesContextArraysWithoutRebinding(Nd4jBackend backend) throws Exception {
        BroadcastMulOp op = new BroadcastMulOp();
        op.setDimension(0, -1);
        INDArray x = Nd4j.ones(DataType.FLOAT, 3, 4, 2);
        INDArray z = Nd4j.ones(DataType.FLOAT, 3, 4, 2);
        INDArray before = z.dup();
        try (OpContext context = Nd4j.getExecutioner().buildContext()) {
            context.setInputArray(0, x);
            context.setInputArray(1, Nd4j.ones(DataType.FLOAT, 2, 3));
            context.setOutputArray(0, z);
            assertThrows(IllegalStateException.class, () -> op.validateDataTypes(context, false));
            assertThrows(IllegalStateException.class, () -> Nd4j.getExecutioner().exec(op, context));
            assertEquals(before, z);

            context.setInputArray(1, Nd4j.ones(DataType.FLOAT, 3, 2));
            assertTrue(op.validateDataTypes(context, false));
            assertArrayEquals(new long[]{0, 2}, op.dimensions().toLongVector());
            context.setOutputArray(0, Nd4j.ones(DataType.FLOAT, 3, 2, 4));
            assertThrows(IllegalArgumentException.class, () -> op.validateDataTypes(context, false));
            context.setOutputArray(0, z);
            context.setInputArray(1, Nd4j.ones(DataType.INT, 3, 2));
            assertThrows(IllegalArgumentException.class, () -> op.validateDataTypes(context, false));
        }
        assertNull(op.x());
        assertNull(op.y());
        assertNull(op.z());
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void broadcastContextExecutionPreservesExplicitNonLastAxes(Nd4jBackend backend) throws Exception {
        for (char order : new char[]{'c', 'f'}) {
            for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                for (char yOrder : new char[]{'c', 'f'}) {
                    for (long[] yShape : new long[][]{{2, 2}, {2, 1, 2}}) {
                        BroadcastMulOp op = new BroadcastMulOp();
                        op.setDimension(0, -1);
                        INDArray x = Nd4j.ones(dtype, 2, 3, 2).dup(order);
                        INDArray y = Nd4j.createFromArray(2.0, 3.0, 5.0, 7.0)
                                .castTo(dtype).reshape(yShape).dup(yOrder);
                        INDArray z = Nd4j.create(dtype, new long[]{2, 3, 2}, order);
                        INDArray expected = Nd4j.createFromArray(
                                2.0, 3.0, 2.0, 3.0, 2.0, 3.0,
                                5.0, 7.0, 5.0, 7.0, 5.0, 7.0).castTo(dtype).reshape(2, 3, 2);
                        try (OpContext context = Nd4j.getExecutioner().buildContext()) {
                            context.setInputArray(0, x);
                            context.setInputArray(1, y);
                            context.setOutputArray(0, z);
                            assertSame(z, Nd4j.getExecutioner().exec(op, context));
                            Nd4j.getExecutioner().commit();
                            assertEquals(expected, z);
                            context.setOutputArray(0, x);
                            assertSame(x, Nd4j.getExecutioner().exec(op, context));
                            Nd4j.getExecutioner().commit();
                            assertEquals(expected, x);
                        }
                        assertNull(op.x());
                        assertNull(op.y());
                        assertNull(op.z());
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void scalarContextExecutionUsesContextInputAndPublishesItsAllocatedOutput(Nd4jBackend backend) throws Exception {
        for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            ScalarAdd op = new ScalarAdd();
            op.setScalar(Nd4j.scalar(dtype, 2));
            INDArray x = Nd4j.createFromArray(1.0, 2.0, 3.0).castTo(dtype);
            try (OpContext context = Nd4j.getExecutioner().buildContext()) {
                context.setInputArray(0, x);
                assertArrayEquals(x.shape(), Shape.shape(op.calculateOutputShape(context).get(0).asLong()));
                INDArray result = Nd4j.getExecutioner().exec(op, context);
                Nd4j.getExecutioner().commit();
                assertSame(context.getOutputArray(0), result);
                assertEquals(Nd4j.createFromArray(3.0, 4.0, 5.0).castTo(dtype), result);
                assertTrue(op.validateDataTypes(context, false));
            }
            assertNull(op.x());
            assertNull(op.z());
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void scalarValidationRejectsTheContextOutputType(Nd4jBackend backend) throws Exception {
        ScalarAdd op = new ScalarAdd();
        op.setScalar(Nd4j.scalar(DataType.FLOAT, 2));
        INDArray output = Nd4j.ones(DataType.INT, 3);
        try (OpContext context = Nd4j.getExecutioner().buildContext()) {
            context.setInputArray(0, Nd4j.ones(DataType.FLOAT, 3));
            context.setOutputArray(0, output);
            assertThrows(IllegalArgumentException.class, () -> op.validateDataTypes(context, false));
            assertThrows(IllegalArgumentException.class, () -> Nd4j.getExecutioner().exec(op, context));
            assertEquals(Nd4j.ones(DataType.INT, 3), output);
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void symbolicScalarShapeInferenceDoesNotDereferenceAMissingArray(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        ScalarAdd op = new ScalarAdd(sameDiff, sameDiff.placeHolder("input", DataType.FLOAT, 2, 3), 2);
        assertNull(op.x());
        assertArrayEquals(new long[]{2, 3}, Shape.shape(op.calculateOutputShape().get(0).asLong()));
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void matchConditionAllModesHaveTheSameUnaryAndPairwiseEquation(Nd4jBackend backend) {
        double[] values = {Double.NEGATIVE_INFINITY, -2, -1, -0.75, 0, 0.75,
                1, 1.25, 2, Double.POSITIVE_INFINITY, Double.NaN};
        double[] comparisons = {0, -1, 1, -0.75, 1, 1, 1, 1, 3, Double.POSITIVE_INFINITY, 0};
        for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT, DataType.UINT32}) {
            // Integer conditions cannot represent NaN/infinity or fractional epsilon.
            double[] inputValues = dtype == DataType.INT ? new double[]{-2, -1, 0, 1, 2, 3}
                    : dtype == DataType.UINT32 ? new double[]{0, 1, 2, 3, 4, 5} : values;
            double[] pairValues = dtype == DataType.INT || dtype == DataType.UINT32
                    ? new double[]{0, 1, 1, 1, 3, 4} : comparisons;
            double epsilon = dtype == DataType.INT || dtype == DataType.UINT32 ? 0 : 0.25;
            INDArray x = Nd4j.createFromArray(inputValues).castTo(dtype);
            INDArray y = Nd4j.createFromArray(pairValues).castTo(dtype);
            INDArray z = Nd4j.create(DataType.BOOL, inputValues.length);
            for (int mode = 0; mode < 16; mode++) {
                MatchConditionTransform op = new MatchConditionTransform(x, z, Conditions.equals(1));
                op.extraArgs()[0] = 1.0;
                op.extraArgs()[1] = epsilon;
                op.extraArgs()[2] = mode;
                boolean[] expected = new boolean[inputValues.length];
                for (int i = 0; i < expected.length; i++)
                    expected[i] = matchesCondition(inputValues[i], 1, epsilon, mode);
                Nd4j.getExecutioner().exec(op);
                assertCanonicalBooleanValues(z, expected);
                assertEquals(Nd4j.createFromArray(expected), z, dtype + " unary mode " + mode);

                op.setY(y);
                op.extraArgs()[0] = epsilon;
                op.extraArgs()[1] = mode;
                for (int i = 0; i < expected.length; i++)
                    expected[i] = matchesCondition(inputValues[i], pairValues[i], epsilon, mode);
                Nd4j.getExecutioner().exec(op);
                assertCanonicalBooleanValues(z, expected);
                assertEquals(Nd4j.createFromArray(expected), z, dtype + " pairwise mode " + mode);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void comparisonResultsKeepCanonicalBooleanValuesWhenCastAndReduced(Nd4jBackend backend) {
        for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.INT, DataType.UINT32}) {
            INDArray x = Nd4j.createFromArray(1, 2, 3).castTo(dtype);
            INDArray y = Nd4j.createFromArray(1, 0, 3).castTo(dtype);
            INDArray equal = x.eq(y);
            assertArrayEquals(new long[]{1, 0, 1}, equal.castTo(DataType.LONG).toLongVector());
            assertArrayEquals(new float[]{1, 0, 1}, equal.castTo(DataType.FLOAT).toFloatVector());
            assertEquals(2L, equal.castTo(DataType.LONG).sumNumber().longValue());
            assertEquals(Nd4j.createFromArray(true, false, true), equal);

            INDArray z = Nd4j.create(DataType.BOOL, 3);
            MatchConditionTransform op = new MatchConditionTransform(x, z, Conditions.greaterThan(1));
            Nd4j.getExecutioner().exec(op);
            assertArrayEquals(new long[]{0, 1, 1}, z.castTo(DataType.LONG).toLongVector());
            op.setY(y);
            Nd4j.getExecutioner().exec(op);
            assertArrayEquals(new long[]{0, 1, 0}, z.castTo(DataType.LONG).toLongVector());
            assertEquals(Nd4j.createFromArray(false, true, false), z);
        }
        // Logical widening must not change the signedness of ordinary integers.
        assertArrayEquals(new long[]{-2, -1, 0, 1},
                Nd4j.createFromArray(-2, -1, 0, 1).castTo(DataType.LONG).toLongVector());
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void scalarAssignmentWritesLongStructuralValues(Nd4jBackend backend) {
        INDArray scalar = Nd4j.create(DataType.LONG);
        INDArray vector = Nd4j.create(DataType.LONG, 3);
        for (long value : new long[]{0L, -3L, 17L}) {
            scalar.assign(value);
            vector.assign(value);
            assertArrayEquals(new long[]{value}, scalar.toLongVector());
            assertArrayEquals(new long[]{value, value, value}, vector.toLongVector());
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void longEqualityComparesIntegerPayloadWithoutFloatingPointRounding(Nd4jBackend backend) {
        long value = 9007199254740992L;
        INDArray x = Nd4j.createFromArray(value, value + 1L, -value - 1L, Long.MAX_VALUE);
        INDArray y = Nd4j.createFromArray(value, value, -value - 1L, Long.MAX_VALUE - 1L);
        assertArrayEquals(new long[]{1, 0, 1, 0}, x.eq(y).castTo(DataType.LONG).toLongVector());
        assertArrayEquals(new long[]{1, 1, 1, 1}, x.eq(x).castTo(DataType.LONG).toLongVector());
    }

    private static void assertCanonicalBooleanValues(INDArray actual, boolean[] expected) {
        long[] numeric = new long[expected.length];
        for (int i = 0; i < expected.length; i++) {
            numeric[i] = expected[i] ? 1L : 0L;
        }
        assertArrayEquals(numeric, actual.castTo(DataType.LONG).toLongVector());
    }

    private static boolean matchesCondition(double value, double comparison, double epsilon, int mode) {
        switch (mode) {
            case 0: return Math.abs(value - comparison) <= epsilon;
            case 1: return Math.abs(value - comparison) > epsilon;
            case 2: return value < comparison;
            case 3: return value > comparison;
            case 4: return value <= comparison;
            case 5: return value >= comparison;
            case 6: return Math.abs(value) < comparison;
            case 7: return Math.abs(value) > comparison;
            case 8: return Double.isInfinite(value);
            case 9: return Double.isNaN(value);
            case 10: return value == comparison;
            case 11: return value != comparison;
            case 12: return Math.abs(value) >= comparison;
            case 13: return Math.abs(value) <= comparison;
            case 14: return Double.isFinite(value);
            case 15: return !Double.isFinite(value);
            default: throw new IllegalArgumentException("Invalid condition mode " + mode);
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void matchConditionSwitchesItsAbiAndInvalidatesCachedExtraArguments(Nd4jBackend backend) {
        for (DataType dtype : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            INDArray x = Nd4j.createFromArray(1.0, 2.0, 3.0).castTo(dtype);
            INDArray y = Nd4j.createFromArray(0.0, 5.0, 2.0).castTo(dtype);
            INDArray z = Nd4j.create(DataType.BOOL, 3);
            MatchConditionTransform op = new MatchConditionTransform(x, z, Conditions.greaterThan(1.5));
            assertEquals(1.5, op.extraArgsDataBuff(dtype).getDouble(0), 0.0);
            Nd4j.getExecutioner().exec(op);
            assertCanonicalBooleanValues(z, new boolean[]{false, true, true});
            assertEquals(Nd4j.createFromArray(false, true, true), z);

            op.setY(y);
            assertEquals(2, op.extraArgs().length);
            assertEquals(Nd4j.EPS_THRESHOLD, op.extraArgsDataBuff(dtype).getDouble(0), 1e-10);
            assertEquals(Conditions.greaterThan().conditionType().index,
                    op.extraArgsDataBuff(dtype).getDouble(1), 0.0);
            Nd4j.getExecutioner().exec(op);
            assertEquals(Nd4j.createFromArray(true, false, true), z);

            op.setY(null);
            assertEquals(3, op.extraArgs().length);
            assertEquals(1.5, op.extraArgsDataBuff(dtype).getDouble(0), 0.0);
            Nd4j.getExecutioner().exec(op);
            assertCanonicalBooleanValues(z, new boolean[]{false, true, true});
            assertEquals(Nd4j.createFromArray(false, true, true), z);
        }
    }
}
