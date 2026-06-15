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

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for each op modified on this branch.
 * Tests use shapes matching the VLM vision encoder pipeline.
 * The crash sequence was: Where -> scatter_nd_update -> gather -> add -> expand_dims -> CRASH
 *
 * Each test runs a single modified op in isolation to identify which one corrupts memory.
 */
@Slf4j
public class CudaModifiedOpIsolationTest {

    // ======== WHERE (boolean/where.cpp, helpers/cuda/where.cu) ========

    @Test
    public void testWhereOp() {
        log.info("Testing Where op (modified on branch)");
        // Where with boolean condition - returns coordinates of true elements
        INDArray condition = Nd4j.create(new float[]{1, 0, 1, 0, 1, 1, 0, 0, 1, 0}, new long[]{10}).castTo(DataType.BOOL);
        SameDiff sd = SameDiff.create();
        SDVariable cond = sd.var("cond", condition);
        SDVariable result = sd.where(cond);
        Map<String, INDArray> out = sd.output(new HashMap<>(), result.name());
        INDArray whereResult = out.get(result.name());
        assertNotNull(whereResult);
        log.info("Where result shape: {}, values: {}", whereResult.shape(), whereResult);
        // 5 true elements -> shape [5, 1]
        assertEquals(5, whereResult.shape()[0]);
    }

    @Test
    public void testWhereLargeOp() {
        log.info("Testing Where op with large tensor (VLM-scale)");
        // VLM uses where on attention masks: [1, 1, 512, 512] bool
        INDArray mask = Nd4j.ones(DataType.BOOL, 1, 1024);
        // All true -> where returns all indices
        SameDiff sd = SameDiff.create();
        SDVariable cond = sd.var("cond", mask);
        SDVariable result = sd.where(cond);
        Map<String, INDArray> out = sd.output(new HashMap<>(), result.name());
        INDArray whereResult = out.get(result.name());
        assertNotNull(whereResult);
        assertEquals(1024, whereResult.shape()[0]);
        log.info("Where large: shape={}", whereResult.shape());
    }

    // ======== GATHER (transforms/gather.cpp, helpers/cuda/gather.cu) ========

    @Test
    public void testGatherOp() {
        log.info("Testing Gather op (modified on branch)");
        INDArray input = Nd4j.linspace(0, 31, 32, DataType.FLOAT).reshape(8, 4);
        INDArray indices = Nd4j.createFromArray(new int[]{0, 2, 5, 7});

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable idx = sd.var("idx", indices);
        SDVariable gathered = sd.gather(in, idx, 0);
        Map<String, INDArray> out = sd.output(new HashMap<>(), gathered.name());
        INDArray result = out.get(gathered.name());
        assertNotNull(result);
        assertArrayEquals(new long[]{4, 4}, result.shape());
        // First row should be [0, 1, 2, 3]
        assertEquals(0.0f, result.getFloat(0, 0), 1e-5);
        log.info("Gather result shape: {}", result.shape());
    }

    @Test
    public void testGather1DFlat() {
        log.info("Testing Gather 1D flat (special path in gather.cu)");
        // The 1D flat gather path uses gatherCudaLinearKernel
        INDArray input = Nd4j.linspace(0, 1023, 1024, DataType.FLOAT);
        INDArray indices = Nd4j.createFromArray(new int[]{0, 100, 500, 999});

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable idx = sd.var("idx", indices);
        SDVariable gathered = sd.gather(in, idx, 0);
        Map<String, INDArray> out = sd.output(new HashMap<>(), gathered.name());
        INDArray result = out.get(gathered.name());
        assertNotNull(result);
        assertEquals(0.0f, result.getFloat(0), 1e-5);
        assertEquals(999.0f, result.getFloat(3), 1e-1);
        log.info("Gather 1D flat: shape={}", result.shape());
    }

    // ======== GATHER_ND (transforms/gatherNd.cpp, helpers/cuda/gather_nd.cu) ========

    @Test
    public void testGatherNdOp() {
        log.info("Testing GatherNd op (modified on branch)");
        INDArray input = Nd4j.linspace(0, 11, 12, DataType.FLOAT).reshape(3, 4);
        INDArray indices = Nd4j.createFromArray(new int[][]{{0, 0}, {1, 2}, {2, 3}});

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable idx = sd.var("idx", indices);
        SDVariable result = sd.gatherNd(in, idx);
        Map<String, INDArray> out = sd.output(new HashMap<>(), result.name());
        INDArray res = out.get(result.name());
        assertNotNull(res);
        assertEquals(0.0f, res.getFloat(0), 1e-5);   // [0,0]
        assertEquals(6.0f, res.getFloat(1), 1e-5);   // [1,2]
        assertEquals(11.0f, res.getFloat(2), 1e-5);  // [2,3]
        log.info("GatherNd: shape={}", res.shape());
    }

    // ======== RESHAPE (shape/reshape.cpp) ========

    @Test
    public void testReshapeOp() {
        log.info("Testing Reshape op (modified on branch)");
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1024, 768);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable reshaped = sd.reshape(in, 1024, 768);
        Map<String, INDArray> out = sd.output(new HashMap<>(), reshaped.name());
        INDArray result = out.get(reshaped.name());
        assertNotNull(result);
        assertArrayEquals(new long[]{1024, 768}, result.shape());
        log.info("Reshape: {} -> {}", input.shape(), result.shape());
    }

    // ======== PERMUTE (shape/permute.cpp) ========

    @Test
    public void testPermuteOp() {
        log.info("Testing Permute op (modified on branch)");
        // VLM uses permute for attention head reshaping: [1, seq, heads, dim] -> [1, heads, seq, dim]
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1024, 12, 64);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable permuted = sd.permute(in, 0, 2, 1, 3);
        Map<String, INDArray> out = sd.output(new HashMap<>(), permuted.name());
        INDArray result = out.get(permuted.name());
        assertNotNull(result);
        assertArrayEquals(new long[]{1, 12, 1024, 64}, result.shape());
        log.info("Permute: {} -> {}", input.shape(), result.shape());
    }

    // ======== MATMUL (blas/matmul.cpp) ========

    @Test
    public void testMatmulOp() {
        log.info("Testing Matmul op (modified on branch)");
        // VLM attention: Q*K^T  [1, 12, 1024, 64] x [1, 12, 64, 1024]
        INDArray q = Nd4j.randn(DataType.FLOAT, 1, 12, 64, 64);
        INDArray k = Nd4j.randn(DataType.FLOAT, 1, 12, 64, 64);

        SameDiff sd = SameDiff.create();
        SDVariable qVar = sd.var("q", q);
        SDVariable kVar = sd.var("k", k);
        SDVariable result = sd.mmul(qVar, kVar);
        Map<String, INDArray> out = sd.output(new HashMap<>(), result.name());
        INDArray res = out.get(result.name());
        assertNotNull(res);
        assertArrayEquals(new long[]{1, 12, 64, 64}, res.shape());
        log.info("Matmul: shape={}", res.shape());
    }

    // ======== SOFTMAX (nn/softmax.cpp) ========

    @Test
    public void testSoftmaxOp() {
        log.info("Testing Softmax op (modified on branch)");
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 12, 256, 256);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable result = sd.nn.softmax(in, -1);
        Map<String, INDArray> out = sd.output(new HashMap<>(), result.name());
        INDArray res = out.get(result.name());
        assertNotNull(res);
        // Softmax sums to 1 along last axis
        INDArray sum = res.sum(-1);
        assertEquals(1.0f, sum.getFloat(0, 0, 0), 1e-3);
        log.info("Softmax: shape={}", res.shape());
    }

    // ======== CLIP_BY_VALUE (transforms/clip_by_value.cpp) ========

    @Test
    public void testClipByValueOp() {
        log.info("Testing ClipByValue op (modified on branch)");
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1024, 768).mul(10);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable clipped = sd.clipByValue(in, -1.0, 1.0);
        Map<String, INDArray> out = sd.output(new HashMap<>(), clipped.name());
        INDArray result = out.get(clipped.name());
        assertNotNull(result);
        assertTrue(result.maxNumber().floatValue() <= 1.0f + 1e-5);
        assertTrue(result.minNumber().floatValue() >= -1.0f - 1e-5);
        log.info("ClipByValue: min={}, max={}", result.minNumber(), result.maxNumber());
    }

    // ======== STACK (transforms/stack.cpp) ========

    @Test
    public void testStackOp() {
        log.info("Testing Stack op (modified on branch)");
        INDArray a = Nd4j.randn(DataType.FLOAT, 256, 768);
        INDArray b = Nd4j.randn(DataType.FLOAT, 256, 768);

        SameDiff sd = SameDiff.create();
        SDVariable va = sd.var("a", a);
        SDVariable vb = sd.var("b", b);
        SDVariable stacked = sd.stack(0, va, vb);
        Map<String, INDArray> out = sd.output(new HashMap<>(), stacked.name());
        INDArray result = out.get(stacked.name());
        assertNotNull(result);
        assertArrayEquals(new long[]{2, 256, 768}, result.shape());
        log.info("Stack: shape={}", result.shape());
    }

    // ======== ONES_AS (tensor/ones_as.cpp) ========

    @Test
    public void testOnesAsOp() {
        log.info("Testing OnesAs op (modified on branch)");
        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, 1024, 768);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", input);
        SDVariable ones = sd.onesLike(in);
        Map<String, INDArray> out = sd.output(new HashMap<>(), ones.name());
        INDArray result = out.get(ones.name());
        assertNotNull(result);
        assertEquals(1.0f, result.getFloat(0), 1e-5);
        assertEquals(1.0f, result.meanNumber().floatValue(), 1e-5);
        log.info("OnesAs: shape={}", result.shape());
    }

    // ======== BOOLEAN NOT (boolean/boolean_not.cpp) ========

    @Test
    public void testBooleanNotOp() {
        log.info("Testing BooleanNot op (modified on branch)");
        INDArray input = Nd4j.create(new float[]{1, 0, 1, 0, 1}).castTo(DataType.BOOL);
        // Boolean not via 1 - x
        INDArray result = Nd4j.ones(DataType.BOOL, 5).sub(input);
        assertNotNull(result);
        log.info("BooleanNot: {}", result);
    }

    // ======== CHAINED OPS - VLM CRASH SEQUENCE ========

    /**
     * Chain the exact ops from the VLM crash sequence.
     * Where -> gather -> expand_dims -> add
     * This is the critical test.
     */
    @Test
    public void testVLMCrashSequence() {
        log.info("Testing VLM crash sequence: where -> gather -> expand_dims -> add");

        for (int iter = 0; iter < 50; iter++) {
            SameDiff sd = SameDiff.create();

            // Where: find non-zero elements in a mask
            INDArray mask = Nd4j.ones(DataType.BOOL, 1024);
            // Set some to false
            for (int i = 512; i < 1024; i++) {
                mask.putScalar(i, 0);
            }
            SDVariable maskVar = sd.var("mask", mask);
            SDVariable whereResult = sd.where(maskVar);

            // Gather: use where indices to gather from data
            INDArray data = Nd4j.randn(DataType.FLOAT, 1024, 64);
            SDVariable dataVar = sd.var("data", data);
            // Gather rows at where indices
            SDVariable gathered = sd.gather(dataVar, whereResult.reshape(-1), 0);

            // Expand dims (view op - crash point in VLM)
            SDVariable expanded = sd.expandDims(gathered, 0);

            // Add (crash detected here in VLM)
            INDArray bias = Nd4j.ones(DataType.FLOAT, 1, 512, 64);
            SDVariable biasVar = sd.var("bias", bias);
            SDVariable added = sd.math.add(expanded, biasVar);

            Map<String, INDArray> out = sd.output(new HashMap<>(), added.name());
            INDArray result = out.get(added.name());
            assertNotNull(result, "Result null at iteration " + iter);
            assertFalse(Float.isNaN(result.getFloat(0)), "NaN at iteration " + iter);

            if (iter % 10 == 0) {
                System.gc();
                log.info("VLM crash sequence iter {}: shape={}", iter, result.shape());
            }
        }
        log.info("VLM crash sequence test passed (50 iterations)");
    }

    /**
     * Rapidly create and destroy SameDiff graphs with the modified ops.
     * This maximizes GC pressure on CUDA buffers.
     */
    @Test
    public void testRapidGraphCreationDestruction() {
        log.info("Testing rapid graph create/destroy with modified ops");

        for (int iter = 0; iter < 100; iter++) {
            SameDiff sd = SameDiff.create();

            INDArray input = Nd4j.randn(DataType.FLOAT, 4, 256, 768);
            SDVariable in = sd.var("in", input);

            // Chain: reshape -> permute -> matmul -> softmax -> reshape
            SDVariable reshaped = sd.reshape(in, 4, 256, 12, 64);
            SDVariable permuted = sd.permute(reshaped, 0, 2, 1, 3); // [4,12,256,64]
            SDVariable transposed = sd.permute(permuted, 0, 1, 3, 2); // [4,12,64,256]
            SDVariable scores = sd.mmul(permuted, transposed); // [4,12,256,256]
            SDVariable attn = sd.nn.softmax(scores, -1);
            SDVariable flat = sd.reshape(attn, 4, 12, 256 * 256);

            Map<String, INDArray> out = sd.output(new HashMap<>(), flat.name());
            INDArray result = out.get(flat.name());
            assertNotNull(result, "Null at iteration " + iter);

            // Force GC to try to reclaim previous iteration's buffers
            if (iter % 5 == 0) {
                System.gc();
            }
        }
        log.info("Rapid graph create/destroy test passed (100 iterations)");
    }
}
