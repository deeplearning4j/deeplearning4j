package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Create;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.tests.tags.NativeTag;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Create op and ConstantOfShape/Expand import behavior.
 * Verifies:
 * 1. Create op iArguments are not duplicated (regression test for [99, 5, 99, 5] bug)
 * 2. sd.create() produces correct output in graph building mode (ConstantOfShape import path)
 * 3. Expand-style multiply-by-ones broadcasting works with sd.create()
 * 4. Create op execution produces arrays with correct shape and data type
 */
@NativeTag
@Tag("samediff")
public class TestCreateOpAndConstantOfShape {

    /**
     * Regression test: Create op constructor must not duplicate iArguments.
     * The bug was iArgs becoming [99, 5, 99, 5] instead of [99, 5] ('c'=99, FLOAT=5).
     */
    @Test
    public void testCreateOpIArgsNotDuplicated() {
        // Test the SameDiff constructor path used by ConstantOfShape hook
        SameDiff sd = SameDiff.create();
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(new long[]{3, 4}));
        Create createOp = new Create(sd, shape, DataType.FLOAT, "c", true);

        long[] iArgs = createOp.iArgs();
        assertEquals(2, iArgs.length,
                "Create op should have exactly 2 iArgs [order, dataType], got: " + Arrays.toString(iArgs));
        assertEquals((long) 'c', iArgs[0], "First iArg should be order char 'c' (99)");
        assertEquals((long) DataType.FLOAT.toInt(), iArgs[1], "Second iArg should be FLOAT dataType int");
    }

    /**
     * Test Create op with different data types to ensure iArgs are correct for each.
     */
    @Test
    public void testCreateOpIArgsMultipleDataTypes() {
        DataType[] types = {DataType.FLOAT, DataType.DOUBLE, DataType.INT, DataType.LONG, DataType.HALF};
        for (DataType dt : types) {
            SameDiff sd = SameDiff.create();
            SDVariable shape = sd.constant("shape", Nd4j.createFromArray(new long[]{2, 3}));
            Create createOp = new Create(sd, shape, dt, "c", false);

            long[] iArgs = createOp.iArgs();
            assertEquals(2, iArgs.length,
                    "Create op for " + dt + " should have exactly 2 iArgs, got: " + Arrays.toString(iArgs));
            assertEquals((long) 'c', iArgs[0]);
            assertEquals((long) dt.toInt(), iArgs[1]);
        }
    }

    /**
     * Test Create op with INDArray constructor (used in direct op execution).
     */
    @Test
    public void testCreateOpINDArrayConstructorIArgs() {
        INDArray shape = Nd4j.createFromArray(new long[]{5, 3});
        Create createOp = new Create(shape, DataType.FLOAT, "c", false);

        long[] iArgs = createOp.iArgs();
        assertEquals(2, iArgs.length,
                "INDArray constructor should have exactly 2 iArgs, got: " + Arrays.toString(iArgs));
        assertEquals((long) 'c', iArgs[0]);
        assertEquals((long) DataType.FLOAT.toInt(), iArgs[1]);
    }

    /**
     * Test Create op with the 2-arg DataType-only constructor.
     */
    @Test
    public void testCreateOpSimpleConstructorIArgs() {
        SameDiff sd = SameDiff.create();
        SDVariable shape = sd.constant("shape", Nd4j.createFromArray(new long[]{4, 5}));
        Create createOp = new Create(sd, shape, DataType.DOUBLE);

        long[] iArgs = createOp.iArgs();
        assertEquals(2, iArgs.length,
                "Simple constructor should have exactly 2 iArgs, got: " + Arrays.toString(iArgs));
        assertEquals((long) 'c', iArgs[0]);
        assertEquals((long) DataType.DOUBLE.toInt(), iArgs[1]);
    }

    /**
     * Simulates what ConstantOfShape does: creates array via sd.create() and executes.
     * Tests the "no value" path (zeros, FLOAT32).
     */
    @Test
    public void testConstantOfShapeZerosPath() {
        SameDiff sd = SameDiff.create();
        SDVariable inputShape = sd.constant("inputShape", Nd4j.createFromArray(new long[]{3, 4, 5}));
        // This mirrors ConstantOfShape.kt line 58: sd.create(outputVarName, inputShape, DataType.FLOAT, "c", true)
        SDVariable outputVar = sd.create("output", inputShape, DataType.FLOAT, "c", true);

        Map<String, INDArray> results = sd.output(Map.of(), "output");
        INDArray result = results.get("output");

        assertNotNull(result, "Output should not be null");
        assertArrayEquals(new long[]{3, 4, 5}, result.shape(), "Output shape should match input shape tensor");
        assertEquals(DataType.FLOAT, result.dataType(), "Output should be FLOAT");
        // initialize=true means zeros
        assertEquals(0.0, result.sumNumber().doubleValue(), 1e-6, "Initialized array should be zeros");
    }

    /**
     * Simulates ConstantOfShape with a value attribute: creates array and fills it.
     * Tests the "with value" path.
     */
    @Test
    public void testConstantOfShapeWithValuePath() {
        SameDiff sd = SameDiff.create();
        SDVariable inputShape = sd.constant("inputShape", Nd4j.createFromArray(new long[]{2, 3}));
        // This mirrors ConstantOfShape.kt lines 61-63
        SDVariable created = sd.create(inputShape, DataType.FLOAT, "c", false);
        double fillValue = 7.0;
        SDVariable filled = sd.assign(created, sd.constant(fillValue));
        SDVariable output = filled.castTo("output", DataType.FLOAT);

        Map<String, INDArray> results = sd.output(Map.of(), "output");
        INDArray result = results.get("output");

        assertNotNull(result, "Output should not be null");
        assertArrayEquals(new long[]{2, 3}, result.shape(), "Output shape should be [2, 3]");
        assertEquals(DataType.FLOAT, result.dataType());
        assertEquals(fillValue, result.getDouble(0, 0), 1e-6, "All values should be fill value");
        assertEquals(fillValue, result.getDouble(1, 2), 1e-6, "All values should be fill value");
        assertEquals(fillValue * 6, result.sumNumber().doubleValue(), 1e-5, "Sum should be fillValue * numElements");
    }

    /**
     * Simulates what Expand does: creates ones array and multiplies for broadcasting.
     */
    @Test
    public void testExpandStyleBroadcast() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var("input", Nd4j.createFromArray(new float[][]{{1, 2, 3}})); // [1, 3]
        SDVariable targetShape = sd.constant("targetShape", Nd4j.createFromArray(new long[]{4, 3}));

        // This mirrors Expand.kt lines 77-79
        SDVariable ones = sd.create(targetShape, input.dataType());
        SDVariable assignedOnes = ones.assign(1.0);
        SDVariable output = assignedOnes.mul("output", input);

        Map<String, INDArray> results = sd.output(Map.of(), "output");
        INDArray result = results.get("output");

        assertNotNull(result, "Expand output should not be null");
        assertArrayEquals(new long[]{4, 3}, result.shape(), "Expanded shape should be [4, 3]");
        // Each row should be [1, 2, 3] due to broadcasting
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, result.getDouble(i, 0), 1e-6);
            assertEquals(2.0, result.getDouble(i, 1), 1e-6);
            assertEquals(3.0, result.getDouble(i, 2), 1e-6);
        }
    }

    /**
     * Test Create op in graph building mode (the actual import scenario).
     * During ONNX import, graph building mode is active.
     */
    @Test
    public void testCreateOpInGraphBuildingMode() {
        SameDiff sd;
        SameDiff.setGraphBuildingMode(true);
        try {
            sd = SameDiff.create();
            SDVariable inputShape = sd.constant("inputShape", Nd4j.createFromArray(new long[]{2, 4}));
            SDVariable output = sd.create("output", inputShape, DataType.FLOAT, "c", true);
        } finally {
            SameDiff.setGraphBuildingMode(false);
        }

        // Execute outside building mode
        Map<String, INDArray> results = sd.output(Map.of(), "output");
        INDArray result = results.get("output");
        assertNotNull(result, "Create op should produce output after building mode");
        assertArrayEquals(new long[]{2, 4}, result.shape(), "Shape should match");
        assertEquals(DataType.FLOAT, result.dataType());
    }

    /**
     * Test ConstantOfShape + assign pattern in graph building mode.
     * This is the full pattern used during actual ONNX import.
     */
    @Test
    public void testConstantOfShapePatternInGraphBuildingMode() {
        SameDiff sd;
        SameDiff.setGraphBuildingMode(true);
        try {
            sd = SameDiff.create();
            SDVariable inputShape = sd.constant("inputShape", Nd4j.createFromArray(new long[]{3, 5}));
            // ConstantOfShape with value=2.0
            SDVariable created = sd.create(inputShape, DataType.FLOAT, "c", false);
            SDVariable filled = sd.assign(created, sd.constant(2.0));
            SDVariable output = filled.castTo("output", DataType.FLOAT);
        } finally {
            SameDiff.setGraphBuildingMode(false);
        }

        Map<String, INDArray> results = sd.output(Map.of(), "output");
        INDArray result = results.get("output");
        assertNotNull(result, "ConstantOfShape pattern should produce output");
        assertArrayEquals(new long[]{3, 5}, result.shape());
        assertEquals(2.0, result.getDouble(0, 0), 1e-6, "Values should be filled constant");
    }

    /**
     * Test Create op with various output data types to ensure calculateOutputDataTypes works.
     */
    @Test
    public void testCreateOpOutputDataTypes() {
        DataType[] types = {DataType.FLOAT, DataType.DOUBLE, DataType.INT, DataType.LONG};
        for (DataType dt : types) {
            SameDiff sd = SameDiff.create();
            SDVariable shape = sd.constant("shape", Nd4j.createFromArray(new long[]{2, 2}));
            SDVariable output = sd.create("output", shape, dt, "c", true);

            Map<String, INDArray> results = sd.output(Map.of(), "output");
            INDArray result = results.get("output");
            assertNotNull(result, "Create should produce output for " + dt);
            assertEquals(dt, result.dataType(), "Output dtype should be " + dt);
            assertArrayEquals(new long[]{2, 2}, result.shape());
        }
    }

    /**
     * Test Expand-style boolean broadcasting via sd.create().
     * Mirrors the Expand.kt DataType.BOOL branch.
     */
    @Test
    public void testExpandBoolBroadcast() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var("input", Nd4j.createFromArray(new boolean[]{true, false, true}));
        // Reshape to [1, 3] for broadcasting
        SDVariable reshaped = sd.reshape(input, 1, 3);
        SDVariable targetShape = sd.constant("targetShape", Nd4j.createFromArray(new long[]{3, 3}));

        // Mirrors Expand.kt lines 72-75 for BOOL path
        SDVariable ones = sd.create(targetShape, DataType.INT8);
        SDVariable assignedOnes = ones.assign(1.0);
        SDVariable casted = sd.castTo(reshaped, DataType.INT8);
        SDVariable multiplied = casted.mul(assignedOnes);
        SDVariable output = sd.castTo("output", multiplied, DataType.BOOL);

        Map<String, INDArray> results = sd.output(Map.of(), "output");
        INDArray result = results.get("output");
        assertNotNull(result, "Bool expand should produce output");
        assertArrayEquals(new long[]{3, 3}, result.shape());
        assertEquals(DataType.BOOL, result.dataType());
        // Each row should be [true, false, true]
        for (int i = 0; i < 3; i++) {
            assertTrue(result.getInt(i, 0) != 0, "Row " + i + " col 0 should be true");
            assertFalse(result.getInt(i, 1) != 0, "Row " + i + " col 1 should be false");
            assertTrue(result.getInt(i, 2) != 0, "Row " + i + " col 2 should be true");
        }
    }
}
