/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused isolation tests for reshape changes on this branch.
 * Tests the scalar broadcast path and ONNX "0 = copy from input" semantics.
 */
@Slf4j
public class CudaReshapeIsolationTest {

    /** Standard reshape: same element count, different shape */
    @Test
    public void testReshapeBasic() {
        log.info("testReshapeBasic");
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(3, 4);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable reshaped = sd.reshape(in, 4, 3);
        INDArray result = sd.output(new HashMap<>(), reshaped.name()).get(reshaped.name());
        assertArrayEquals(new long[]{4, 3}, result.shape());
        assertEquals(1.0f, result.getFloat(0, 0), 1e-5);
        log.info("Basic reshape OK: {}", result.shape());
    }

    /** Reshape with -1 (infer dimension) */
    @Test
    public void testReshapeInfer() {
        log.info("testReshapeInfer");
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable reshaped = sd.reshape(in, 6, -1);
        INDArray result = sd.output(new HashMap<>(), reshaped.name()).get(reshaped.name());
        assertArrayEquals(new long[]{6, 4}, result.shape());
        log.info("Infer reshape OK: {}", result.shape());
    }

    /** Reshape with ONNX "0 = copy from input" semantics */
    @Test
    public void testReshapeOnnxZeroCopy() {
        log.info("testReshapeOnnxZeroCopy");
        // Input shape [2, 3, 4], reshape args [0, -1] should give [2, 12]
        // 0 at position 0 copies dim 2 from input
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable shape = sd.constant(Nd4j.createFromArray(new long[]{0, -1}));
        SDVariable reshaped = sd.reshape(in, shape);
        INDArray result = sd.output(new HashMap<>(), reshaped.name()).get(reshaped.name());
        log.info("ONNX zero-copy reshape result shape: {}", result.shape());
        // Expected: [2, 12] because 0 copies dim 2 from input, -1 infers 12
        assertArrayEquals(new long[]{2, 12}, result.shape());
        assertEquals(24, result.length());
    }

    /** Reshape scalar to larger - the new scalar broadcast path */
    @Test
    public void testReshapeScalarBroadcast() {
        log.info("testReshapeScalarBroadcast");
        // This is the NEW behavior on this branch: scalar reshaped to [4] broadcasts
        INDArray scalar = Nd4j.scalar(DataType.FLOAT, 5.0f);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", scalar);
        // Try to reshape scalar to [1] - should work
        SDVariable reshaped = sd.reshape(in, 1);
        INDArray result = sd.output(new HashMap<>(), reshaped.name()).get(reshaped.name());
        assertNotNull(result);
        assertEquals(5.0f, result.getFloat(0), 1e-5);
        log.info("Scalar to [1] reshape OK: {}", result);
    }

    /** Reshape length-1 tensor to larger shape - tests the broadcast path */
    @Test
    public void testReshapeLength1ToBroadcast() {
        log.info("testReshapeLength1ToBroadcast - testing scalar broadcast path");
        // length-1 input, larger output: the new code broadcasts
        INDArray input = Nd4j.createFromArray(new float[]{42.0f});
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        // This should trigger the scalar broadcast path: len==1 && zLen > 1
        // This is WRONG for a standard reshape but may be needed for ONNX
        SDVariable reshaped = sd.reshape(in, 4);
        try {
            INDArray result = sd.output(new HashMap<>(), reshaped.name()).get(reshaped.name());
            log.info("Length-1 -> [4] reshape: shape={}, values={}", result.shape(), result);
            // If this succeeds, check if it broadcast (fills with 42) or failed
            // Standard reshape semantics: this should FAIL because 1 != 4
            // ONNX broadcast semantics: this fills with 42
            // The result tells us which path was taken
            if (result.length() == 4) {
                log.warn("SCALAR BROADCAST: reshape(1 element) -> [4] succeeded with broadcasting!");
                log.warn("This changes reshape semantics and may cause downstream issues");
            }
        } catch (Exception e) {
            log.info("Length-1 -> [4] reshape correctly threw: {}", e.getMessage());
            // This is the expected behavior for standard reshape
        }
    }

    /** Repeated reshape in a chain - simulates VLM pipeline */
    @Test
    public void testReshapeChain() {
        log.info("testReshapeChain");
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1024, 768);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);

        // Chain of reshapes like in VLM: [1,1024,768] -> [1024,768] -> [1024,12,64] -> [12,1024,64]
        SDVariable r1 = sd.reshape(in, 1024, 768);
        SDVariable r2 = sd.reshape(r1, 1024, 12, 64);
        SDVariable p = sd.permute(r2, 1, 0, 2); // [12, 1024, 64]
        SDVariable r3 = sd.reshape(p, 12, 1024, 64);

        INDArray result = sd.output(new HashMap<>(), r3.name()).get(r3.name());
        assertNotNull(result);
        assertArrayEquals(new long[]{12, 1024, 64}, result.shape());
        assertFalse(Float.isNaN(result.getFloat(0)));
        log.info("Reshape chain OK: {}", result.shape());
    }

    /** Reshape with GC pressure - the race condition */
    @Test
    public void testReshapeGcPressure() {
        log.info("testReshapeGcPressure");
        for (int iter = 0; iter < 100; iter++) {
            SameDiff sd = SameDiff.create();
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, 256, 768);
            SDVariable in = sd.var("in", input);

            SDVariable r1 = sd.reshape(in, 1024, 768);
            SDVariable r2 = sd.reshape(r1, 1024, 12, 64);
            SDVariable result = sd.math.add(r2, sd.constant(Nd4j.ones(DataType.FLOAT, 1024, 12, 64)));

            INDArray out = sd.output(new HashMap<>(), result.name()).get(result.name());
            assertNotNull(out, "Null at iter " + iter);
            assertFalse(Float.isNaN(out.getFloat(0)), "NaN at iter " + iter);

            if (iter % 10 == 0) {
                System.gc();
                log.info("Reshape GC pressure iter {}: OK", iter);
            }
        }
        log.info("Reshape GC pressure test passed");
    }

    /** Reshape no-copy (modified on branch) */
    @Test
    public void testReshapeNoCopy() {
        log.info("testReshapeNoCopy");
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape('c', 2, 3, 4);
        INDArray reshaped = input.reshape('c', 6, 4);
        assertArrayEquals(new long[]{6, 4}, reshaped.shape());
        assertEquals(1.0f, reshaped.getFloat(0, 0), 1e-5);
        assertEquals(24.0f, reshaped.getFloat(5, 3), 1e-5);
        log.info("Reshape no-copy OK");
    }

    /** Reshape no-copy with non-contiguous (Fortran order) */
    @Test
    public void testReshapeNoCopyFortran() {
        log.info("testReshapeNoCopyFortran");
        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape('f', 3, 4);
        INDArray reshaped = input.reshape('c', new long[]{12});
        assertEquals(12, reshaped.length());
        log.info("Reshape no-copy Fortran -> C: values={}", reshaped);
    }

    /** Large reshape matching VLM vision encoder dimensions */
    @Test
    public void testReshapeVLMScale() {
        log.info("testReshapeVLMScale");
        // VLM vision encoder: patch embeddings [1, 1024, 768]
        // Reshape for attention heads: [1, 1024, 12, 64]
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1024, 768);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable reshaped = sd.reshape(in, 1, 1024, 12, 64);
        INDArray result = sd.output(new HashMap<>(), reshaped.name()).get(reshaped.name());
        assertNotNull(result);
        assertArrayEquals(new long[]{1, 1024, 12, 64}, result.shape());
        // Verify data integrity: first element should match
        assertEquals(input.getFloat(0), result.getFloat(0), 1e-5);
        log.info("VLM-scale reshape OK");
    }
}
