package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.tests.tags.NativeTag;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ONNX import optimizations:
 * 1. Graph building mode flag behavior
 * 2. calculateOutputShape works in building mode
 * 3. Graph execution after building-mode construction
 * 4. SameDiff graph construction in building mode
 */
@NativeTag
@Tag("samediff")
public class TestImportOptimizations {

    @Test
    public void testGraphBuildingModeFlag() {
        assertFalse(SameDiff.isInGraphBuildingMode(), "Should start false");
        SameDiff.setGraphBuildingMode(true);
        assertTrue(SameDiff.isInGraphBuildingMode(), "Should be true after set");
        SameDiff.setGraphBuildingMode(false);
        assertFalse(SameDiff.isInGraphBuildingMode(), "Should be false after reset");
    }

    @Test
    public void testGraphBuildingModeThreadLocal() {
        SameDiff.setGraphBuildingMode(true);
        assertTrue(SameDiff.isInGraphBuildingMode());

        final boolean[] otherThreadValue = {true};
        Thread t = new Thread(() -> otherThreadValue[0] = SameDiff.isInGraphBuildingMode());
        t.start();
        try { t.join(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

        assertFalse(otherThreadValue[0], "Graph building mode should be thread-local");
        SameDiff.setGraphBuildingMode(false);
    }

    @Test
    public void testCalculateOutputShapeWorksInBuildingMode() {
        // Shape calculation should work regardless of building mode
        INDArray x = Nd4j.ones(DataType.FLOAT, 3, 4);
        INDArray w = Nd4j.ones(DataType.FLOAT, 4, 5);

        DynamicCustomOp matmulOp = DynamicCustomOp.builder("matmul")
                .addInputs(x, w)
                .build();

        // Test in building mode
        SameDiff.setGraphBuildingMode(true);
        try {
            List<DataBuffer> shapes = Nd4j.getExecutioner().calculateOutputShape(matmulOp);
            assertFalse(shapes.isEmpty(), "Should calculate output shapes in building mode");
            INDArray dummy = Nd4j.createFromDescriptor(shapes.get(0));
            assertArrayEquals(new long[]{3, 5}, dummy.shape(), "Shape should be [3,5]");
        } finally {
            SameDiff.setGraphBuildingMode(false);
        }

        // Same test outside building mode
        List<DataBuffer> shapes2 = Nd4j.getExecutioner().calculateOutputShape(matmulOp);
        assertFalse(shapes2.isEmpty(), "Should calculate output shapes outside building mode");
        INDArray dummy2 = Nd4j.createFromDescriptor(shapes2.get(0));
        assertArrayEquals(new long[]{3, 5}, dummy2.shape(), "Shape should be [3,5]");
    }

    @Test
    public void testGraphBuiltInBuildingModeCanExecute() {
        // Build a graph in building mode, then verify it executes correctly
        SameDiff sd;
        SameDiff.setGraphBuildingMode(true);
        try {
            sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 3);
            SDVariable weights = sd.var("weights", Nd4j.ones(DataType.FLOAT, 3, 4));
            SDVariable output = sd.mmul("output", input, weights);
        } finally {
            SameDiff.setGraphBuildingMode(false);
        }

        // Execute outside building mode - should produce correct results
        Map<String, INDArray> results = sd.output(
                Map.of("input", Nd4j.ones(DataType.FLOAT, 2, 3)),
                "output");

        INDArray result = results.get("output");
        assertNotNull(result, "Output should be produced");
        assertArrayEquals(new long[]{2, 4}, result.shape(), "Output shape should be correct");
        assertEquals(3.0, result.getDouble(0, 0), 1e-5, "Values should be correct");
    }

    @Test
    public void testMultiOpGraphBuiltInBuildingModeCanExecute() {
        // Build a multi-op graph in building mode
        SameDiff sd;
        SameDiff.setGraphBuildingMode(true);
        try {
            sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
            SDVariable w1 = sd.var("w1", Nd4j.ones(DataType.FLOAT, 4, 8));
            SDVariable w2 = sd.var("w2", Nd4j.ones(DataType.FLOAT, 8, 2));
            SDVariable hidden = sd.nn.relu("hidden", sd.mmul("mm1", input, w1), 0);
            SDVariable output = sd.mmul("output", hidden, w2);
        } finally {
            SameDiff.setGraphBuildingMode(false);
        }

        Map<String, INDArray> results = sd.output(
                Map.of("input", Nd4j.ones(DataType.FLOAT, 3, 4)),
                "output");

        INDArray result = results.get("output");
        assertNotNull(result, "Output should be produced from multi-op graph");
        assertArrayEquals(new long[]{3, 2}, result.shape(), "Output shape should be [3,2]");
        // ones * ones -> 4s, relu(4s) -> 4s, 4s * ones -> 32s
        assertEquals(32.0, result.getDouble(0, 0), 1e-5, "Values should be computed correctly");
    }

    @Test
    public void testNormalEagerModeStillWorks() {
        // Verify that normal (non-building) eager mode still works
        assertFalse(SameDiff.isInGraphBuildingMode());
        SameDiff sd = SameDiff.create().enableEagerMode();
        SDVariable input = sd.var("input", Nd4j.ones(DataType.FLOAT, 3, 4));
        SDVariable weights = sd.var("weights", Nd4j.ones(DataType.FLOAT, 4, 5));
        SDVariable result = sd.mmul("matmul", input, weights);

        // Execute to verify
        INDArray output = sd.outputSingle(
                Map.of(), "matmul");
        assertNotNull(output, "Output should be produced in eager mode");
    }

    @Test
    public void testBuildingModeWithEagerDoesNotCrash() {
        // Building mode + eager mode should not crash (this is the import scenario)
        SameDiff.setGraphBuildingMode(true);
        try {
            SameDiff sd = SameDiff.create().enableEagerMode();
            SDVariable x = sd.var("x", Nd4j.ones(DataType.FLOAT, 2, 3));
            SDVariable w = sd.var("w", Nd4j.ones(DataType.FLOAT, 3, 4));
            // This should not crash - building mode should prevent actual execution
            SDVariable result = sd.mmul("result", x, w);
            // Graph structure should be intact
            assertTrue(sd.getOps().containsKey("result") || sd.hasVariable("result"),
                    "Graph should have the result op/variable");
        } finally {
            SameDiff.setGraphBuildingMode(false);
        }
    }
}
