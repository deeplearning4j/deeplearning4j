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

package org.eclipse.deeplearning4j.nd4j.linalg.crash;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal reproducer for SIGSEGV in CrashTest.testNonEWSViews2.
 *
 * Tests each operation in op() in isolation on non-contiguous f-order TAD views.
 *
 * The parent array is f-order [8, 128, 64] with strides [1, 8, 1024].
 * tensorAlongDimension(k, 1, 2) yields shape [128, 64] with strides [8, 1024].
 * These views are NON-CONTIGUOUS: elements in the row-direction are 8 floats apart,
 * elements in the column-direction are 1024 floats apart.
 *
 * The key hypotheses to test:
 * 1. BlasBufferUtil.getBlasStride() returns stride(-1) = stride(1) = 1024 for these views.
 *    BLAS sdot called with N=8192 elements and incX=1024 accesses 8192*1024 = 8M offsets,
 *    but the buffer only holds 65536 elements. This is an out-of-bounds access -> SIGSEGV.
 * 2. Other ops that treat non-contiguous views as contiguous may also crash.
 */
@Slf4j
@NativeTag
@Tag(TagNames.NDARRAY_INDEXING)
public class CrashTestMinimal extends BaseNd4jTestWithBackends {

    private static final int DIM_BATCH = 8;
    private static final int DIM_ROWS = 128;
    private static final int DIM_COLS = 64;

    /**
     * Verify the strides of an f-order TAD view are as expected.
     * This is a diagnostic test to confirm the view geometry.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFOrderTadViewStrides(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');

        // f-order [8, 128, 64]: strides should be [1, 8, 8*128=1024]
        log.info("Parent shape: {}, strides: {}, ordering: {}",
                java.util.Arrays.toString(x.shape()),
                java.util.Arrays.toString(x.stride()),
                x.ordering());
        assertEquals(1L, x.stride(0), "f-order: stride(0) should be 1");
        assertEquals(8L, x.stride(1), "f-order: stride(1) should be 8");
        assertEquals(1024L, x.stride(2), "f-order: stride(2) should be 1024");

        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        // TAD along {1,2} of [8,128,64] f-order gives shape [128,64] with strides [8, 1024]
        log.info("TAD shape: {}, strides: {}, ordering: {}, offset: {}",
                java.util.Arrays.toString(tad.shape()),
                java.util.Arrays.toString(tad.stride()),
                tad.ordering(),
                tad.offset());
        assertEquals(2, tad.rank(), "TAD should be rank 2");
        assertEquals(128L, tad.size(0), "TAD rows should be 128");
        assertEquals(64L, tad.size(1), "TAD cols should be 64");
        assertEquals(8L, tad.stride(0), "TAD stride(0) should be 8 (batch stride of parent)");
        assertEquals(1024L, tad.stride(1), "TAD stride(1) should be 1024 (col stride of parent)");

        // CRITICAL: getBlasStride returns stride(-1) = stride(1) = 1024 for this view
        // BLAS sdot called with N=8192 and incX=1024 accesses elements at offsets 0, 1024, 2048, ..., 8191*1024
        // Maximum offset = 8191*1024 = 8,387,584 — but buffer only has 65536 elements!
        // This is a massive out-of-bounds access that causes SIGSEGV in the BLAS library.
        int blasStride = BlasBufferUtil.getBlasStride(tad);
        log.info("BlasBufferUtil.getBlasStride(tad) = {}", blasStride);
        log.info("tad.length() = {}, buffer size = {}", tad.length(), x.data().length());
        log.info("Max BLAS access offset = {} (stride {} * length {})",
                (long) blasStride * tad.length(), blasStride, tad.length());
        log.info("Buffer elements available = {}", x.data().length());

        // The blas stride 1024 times the number of elements 8192 = 8,387,584
        // which vastly exceeds the buffer size of 65536
        long maxBlasOffset = (long) blasStride * tad.length();
        long bufferSize = x.data().length();
        log.warn("BLAS stride bug: max access offset {} >> buffer size {}",
                maxBlasOffset, bufferSize);
        assertTrue(maxBlasOffset > bufferSize,
                "Confirming the stride bug: BLAS will access out-of-bounds memory");
    }

    /**
     * Test broadcast operations (addiRowVector, addiColumnVector) on f-order TAD view.
     * These go through libnd4j ops, not BLAS, so they should handle strides correctly.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBroadcastOpsOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);

        INDArray row = Nd4j.ones(tad.columns());
        INDArray column = Nd4j.ones(tad.rows(), 1);

        log.info("Testing addiRowVector on f-order TAD view...");
        assertDoesNotThrow(() -> tad.addiRowVector(row),
                "addiRowVector should not throw on f-order TAD view");

        log.info("Testing addiColumnVector on f-order TAD view...");
        assertDoesNotThrow(() -> tad.addiColumnVector(column),
                "addiColumnVector should not throw on f-order TAD view");

        log.info("Broadcast ops passed on f-order TAD view");
    }

    /**
     * Test scalar op (addi) on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarOpOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);

        log.info("Testing addi scalar on f-order TAD view...");
        assertDoesNotThrow(() -> tad.addi(2.0f),
                "addi scalar should not throw on f-order TAD view");

        log.info("Scalar op passed on f-order TAD view");
    }

    /**
     * Test reduction ops (sum, max) on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionOpsOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        tad.addi(1.0f); // ensure positive values

        log.info("Testing sumNumber on f-order TAD view...");
        float[] sum = new float[1];
        assertDoesNotThrow(() -> sum[0] = tad.sumNumber().floatValue(),
                "sumNumber should not throw on f-order TAD view");
        log.info("sum = {}", sum[0]);

        log.info("Testing max(0) on f-order TAD view...");
        assertDoesNotThrow(() -> tad.max(0),
                "max(0) should not throw on f-order TAD view");

        log.info("Testing max(1) on f-order TAD view...");
        assertDoesNotThrow(() -> tad.max(1),
                "max(1) should not throw on f-order TAD view");

        log.info("Reduction ops passed on f-order TAD view");
    }

    /**
     * Test index reduction (ArgMax) on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgMaxOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);

        log.info("Testing ArgMax on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.getExecutioner().exec(new ArgMax(tad)),
                "ArgMax should not throw on f-order TAD view");

        log.info("Testing argMax(0) on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.argMax(tad, 0),
                "argMax(0) should not throw on f-order TAD view");

        log.info("Testing argMax(1) on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.argMax(tad, 1),
                "argMax(1) should not throw on f-order TAD view");

        log.info("ArgMax ops passed on f-order TAD view");
    }

    /**
     * Test transform op (Sqrt) on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSqrtOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        tad.addi(1.0f); // ensure positive values for sqrt

        log.info("Testing Sqrt on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.getExecutioner().exec(new Sqrt(tad, tad)),
                "Sqrt should not throw on f-order TAD view");

        log.info("Sqrt passed on f-order TAD view");
    }

    /**
     * Test dup operations on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDupOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);

        log.info("Testing dup(ordering) on f-order TAD view...");
        INDArray[] dups = new INDArray[4];
        assertDoesNotThrow(() -> dups[0] = tad.dup(tad.ordering()),
                "dup(ordering) should not throw on f-order TAD view");
        assertDoesNotThrow(() -> dups[1] = tad.dup(tad.ordering()),
                "dup(ordering) second call should not throw");
        assertDoesNotThrow(() -> dups[2] = tad.dup('c'),
                "dup('c') should not throw on f-order TAD view");
        assertDoesNotThrow(() -> dups[3] = tad.dup('f'),
                "dup('f') should not throw on f-order TAD view");

        // Verify dups are contiguous
        assertEquals(1L, dups[2].stride(1), "c-order dup stride(1) should be 1");
        assertEquals(1L, dups[3].stride(0), "f-order dup stride(0) should be 1");

        log.info("Dup operations passed on f-order TAD view");
    }

    /**
     * Test vstack/hstack on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStackOpsOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        INDArray x1 = tad.dup(tad.ordering());
        INDArray x2 = tad.dup(tad.ordering());

        log.info("Testing vstack on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.vstack(tad, x1, x2),
                "vstack should not throw on f-order TAD view");

        log.info("Testing hstack on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.hstack(tad, x1, x2),
                "hstack should not throw on f-order TAD view");

        log.info("Stack ops passed on f-order TAD view");
    }

    /**
     * Test reduce3 op (ManhattanDistance) on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testManhattanDistanceOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        INDArray x2 = tad.dup(tad.ordering());

        log.info("Testing ManhattanDistance on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.getExecutioner().exec(new ManhattanDistance(tad, x2)),
                "ManhattanDistance should not throw on f-order TAD view");

        log.info("ManhattanDistance passed on f-order TAD view");
    }

    /**
     * Test flatten on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFlattenOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        INDArray x1 = tad.dup(tad.ordering());
        INDArray x2 = tad.dup(tad.ordering());

        log.info("Testing toFlattened on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.toFlattened(tad, x1, x2),
                "toFlattened should not throw on f-order TAD view");

        log.info("Flatten passed on f-order TAD view");
    }

    /**
     * Test softmax / log-softmax on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);

        log.info("Testing SoftMax on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.getExecutioner().exec((CustomOp) new SoftMax(tad)),
                "SoftMax should not throw on f-order TAD view");

        log.info("Testing LogSoftMax on f-order TAD view...");
        assertDoesNotThrow(() -> Nd4j.getExecutioner().exec((CustomOp) new LogSoftMax(tad)),
                "LogSoftMax should not throw on f-order TAD view");

        log.info("Softmax ops passed on f-order TAD view");
    }

    /**
     * Test BooleanIndexing on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBooleanIndexingOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        INDArray x1 = tad.dup(tad.ordering());

        log.info("Testing BooleanIndexing.replaceWhere on f-order TAD view...");
        assertDoesNotThrow(() -> BooleanIndexing.replaceWhere(tad, 5f, Conditions.lessThan(8f)),
                "replaceWhere should not throw on f-order TAD view");

        log.info("Testing BooleanIndexing.assignIf on f-order TAD view...");
        assertDoesNotThrow(() -> BooleanIndexing.assignIf(tad, x1, Conditions.greaterThan(-1000000000f)),
                "assignIf should not throw on f-order TAD view");

        log.info("BooleanIndexing passed on f-order TAD view");
    }

    /**
     * Test std/var on f-order TAD view.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testStdVarOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);
        tad.addi(1.0f);

        log.info("Testing stdNumber on f-order TAD view...");
        float[] std = new float[1];
        assertDoesNotThrow(() -> std[0] = tad.stdNumber().floatValue(),
                "stdNumber should not throw on f-order TAD view");
        log.info("std = {}", std[0]);

        log.info("Testing std(0) on f-order TAD view...");
        assertDoesNotThrow(() -> tad.std(0),
                "std(0) should not throw on f-order TAD view");

        log.info("Testing std(1) on f-order TAD view...");
        assertDoesNotThrow(() -> tad.std(1),
                "std(1) should not throw on f-order TAD view");

        log.info("Std ops passed on f-order TAD view");
    }

    /**
     * Test BLAS dot product on f-order TAD view.
     *
     * ROOT CAUSE TEST: BlasBufferUtil.getBlasStride() returns stride(-1) = stride(1) = 1024
     * for an f-order TAD view [128,64] with strides [8, 1024].
     * BLAS sdot is called with N=8192 and incX=1024, accessing elements at offsets
     * 0, 1024, 2048, ..., 8191*1024 = 8,387,584 — but the buffer only has 65536 elements.
     * This causes SIGSEGV in the BLAS library.
     *
     * The correct behavior should be to copy non-contiguous arrays before calling BLAS dot.
     * The current GemmParams.copyIfNeccessary() only copies for matrix multiply (gemm),
     * not for dot product (level1 sdot).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBlassDotOnFOrderTadViewStrideDiagnostic(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);  // shape [128,64], strides [8, 1024]
        INDArray x1 = tad.dup(tad.ordering());            // contiguous copy

        int blasStrideForTad = BlasBufferUtil.getBlasStride(tad);
        int blasStrideForX1 = BlasBufferUtil.getBlasStride(x1);

        log.info("TAD: shape={}, strides={}, blasStride={}, offset={}, bufferLen={}",
                java.util.Arrays.toString(tad.shape()),
                java.util.Arrays.toString(tad.stride()),
                blasStrideForTad,
                tad.offset(),
                tad.data().length());
        log.info("x1 (contiguous dup): strides={}, blasStride={}",
                java.util.Arrays.toString(x1.stride()),
                blasStrideForX1);

        // For the TAD with stride [8, 1024], getBlasStride returns stride(-1)=stride(1)=1024
        // For the contiguous dup, getBlasStride should return stride(-1)=stride(1)=1
        assertEquals(1024, blasStrideForTad,
                "TAD blasStride should be 1024 (the bug: stride(-1) picks the column stride 1024)");
        // Contiguous f-order [128,64] has strides [1, 128], so stride(-1) = stride(1) = 128
        assertEquals(128, blasStrideForX1,
                "Contiguous f-order dup [128,64] blasStride should be 128 (stride(1))");
        // Note: even blasStrideForX1=128 is problematic: BLAS sdot with N=8192, incY=128 reads
        // offset + i*128 for i=0..8191, max offset = 8191*128 = 1,048,448 >> 8192 elements in x1 buffer.
        // However x1 is freshly dup'd so its buffer happens to be at a valid (if wasteful) memory region.
        // In practice the crash is from the TAD's incX=1024 causing reads far past the parent buffer end.

        // A contiguous f-order [128,64] array has strides [1, 128] so stride(-1) = stride(1) = 128.
        // BLAS sdot with N=8192, incY=128 accesses element i of Y at position offset + i*128,
        // max = 8191*128 = 1,048,448 >> 8192 elements in the new buffer -> ALSO out of bounds.
        // The correct BLAS incX for a 2D strided view is UNDEFINED — BLAS only handles 1D vectors.

        // The REAL issue: getBlasStride() for a non-vector 2D array returns stride(-1) which is
        // NOT a valid 1D element stride. It is the stride of the LAST axis, not the stride
        // between consecutive logical elements when traversing a 2D matrix in memory.
        // For a non-contiguous f-order view, the correct behavior is to copy to contiguous before
        // calling any BLAS level-1 op, or use the libnd4j Dot op which handles strides correctly.

        // The actual call that crashes: dot(x, x1) where x is a non-contiguous TAD view
        // BLAS sdot: N=8192, incX=1024, incY=128 -> max access at offset 8191*1024 = 8,387,584
        // TAD buffer has only 65536 elements -> OUT OF BOUNDS -> SIGSEGV
        long maxAccessOffset = (long) blasStrideForTad * tad.length();
        log.warn("BLAS dot would access offset up to {} but TAD buffer only has {} elements -> SIGSEGV",
                maxAccessOffset, tad.data().length());

        // This is the CRASH: Nd4j.getBlasWrapper().dot(tad, x1) will SIGSEGV
        // We do NOT call it here — we've already confirmed the bug analytically.
        // The fix should be in CpuLevel1.sdot: detect when incX or incY is NOT a valid 1D
        // element stride (i.e., when the array is a non-contiguous 2D view) and fall through
        // to the Dot op instead of calling cblas_sdot.
        // The safe condition is: the array must be 1D OR must be contiguous (EWS=1 or stride(0)=1
        // for f-order, stride(1)=1 for c-order), not just any non-negative stride value.
    }

    /**
     * Test GEMM on f-order TAD view — this should work because GemmParams.copyIfNeccessary()
     * detects and copies non-contiguous views before calling BLAS.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemmOnFOrderTadView(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        INDArray y = Nd4j.create(new int[]{DIM_BATCH, DIM_COLS, DIM_ROWS}, 'f');
        Nd4j.rand(x);
        Nd4j.rand(y);
        INDArray xTad = x.tensorAlongDimension(0, 1, 2);  // shape [128, 64]
        INDArray yTad = y.tensorAlongDimension(0, 1, 2);  // shape [64, 128]

        log.info("Testing gemm(xTad, yTad, false, false) on f-order TAD views...");
        // This goes through GemmParams which copies non-contiguous arrays
        // so it should NOT crash
        assertDoesNotThrow(() -> Nd4j.gemm(xTad, yTad, false, false),
                "gemm should not crash with non-contiguous f-order TAD views (GemmParams copies them)");

        log.info("GEMM passed on f-order TAD view");
    }

    /**
     * Diagnose the BLAS dot product stride bug for contiguous 2D f-order arrays.
     *
     * Even a "safe" contiguous f-order [128,64] copy has stride(1)=128. getBlasStride returns
     * stride(-1)=128. BLAS sdot with N=8192 and incX=128 accesses offset + i*128 for i=0..8191,
     * max = 8191*128 = 1,048,448, but the buffer only has 8192 elements -> OUT OF BOUNDS.
     *
     * This means the bug in BaseBlasWrapper.dot() affects ALL 2D arrays, not just non-contiguous
     * views. The BLAS dot product via cblas_sdot is only correct for 1D vectors or when the
     * array has been flattened. For 2D arrays, the libnd4j Dot op must be used instead.
     *
     * We do NOT call dot here — we diagnose analytically to avoid crashing the test JVM.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBlassDotStrideBugForContiguous2DArray(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);  // non-contiguous [128,64] strides=[8,1024]
        INDArray tadCopy = tad.dup('f');                   // contiguous [128,64] strides=[1,128]

        log.info("Contiguous f-order dup strides: {}", java.util.Arrays.toString(tadCopy.stride()));
        assertEquals(1L, tadCopy.stride(0), "f-order contiguous copy: stride(0) should be 1");
        assertEquals(128L, tadCopy.stride(1), "f-order contiguous copy: stride(1) should be 128");

        int blasStrideContiguous = BlasBufferUtil.getBlasStride(tadCopy);
        assertEquals(128, blasStrideContiguous,
                "Contiguous f-order [128,64]: blasStride=stride(-1)=stride(1)=128");

        // BLAS sdot with N=8192, incX=128 -> max access = offset + 8191*128 = 1,048,448
        // but tadCopy buffer has only 8192 elements -> OUT OF BOUNDS -> SIGSEGV
        long maxAccessContiguous = (long) blasStrideContiguous * tadCopy.length();
        log.warn("BLAS dot on contiguous copy: max access offset {} >> buffer size {}",
                maxAccessContiguous, tadCopy.data().length());
        assertTrue(maxAccessContiguous > tadCopy.data().length(),
                "Even contiguous 2D f-order copy is out-of-bounds with blasStride=stride(1)");

        // NON-contiguous TAD view is even worse: stride 1024 * 8192 elements = 8,388,608
        int blasStrideTad = BlasBufferUtil.getBlasStride(tad);
        long maxAccessTad = (long) blasStrideTad * tad.length();
        log.warn("BLAS dot on TAD view: max access offset {} >> buffer size {}",
                maxAccessTad, x.data().length());

        // Summary: the bug is in BaseBlasWrapper.dot() passing 2D arrays directly to BLAS Level1.
        // The fix must ensure that for 2D arrays (or any non-1D array), dot() uses the libnd4j
        // Dot op instead of cblas_sdot, since BLAS Level1 can only handle 1D strided vectors.
        log.info("Confirmed: getBlasStride returns stride(-1) which is NOT the element-wise stride for 2D arrays");
        log.info("Fix: BaseBlasWrapper.dot() must use libnd4j Dot op for any non-1D array");
    }

    /**
     * Full op() sequence from CrashTest but with non-contiguous f-order TAD view —
     * runs operations one at a time up to (but not including) the crashing dot call,
     * to identify the exact point of failure.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOpSequenceUpToBlassDot(Nd4jBackend backend) {
        INDArray x = Nd4j.create(new int[]{DIM_BATCH, DIM_ROWS, DIM_COLS}, 'f');
        Nd4j.rand(x);
        INDArray tad = x.tensorAlongDimension(0, 1, 2);

        INDArray row = Nd4j.ones(tad.columns());
        INDArray column = Nd4j.ones(tad.rows(), 1);

        log.info("Step 1: addiRowVector");
        tad.addiRowVector(row);

        log.info("Step 2: addiColumnVector");
        tad.addiColumnVector(column);

        log.info("Step 3: addi scalar");
        tad.addi(2);

        log.info("Step 4: sumNumber");
        float sum = tad.sumNumber().floatValue();
        log.info("sum = {}", sum);

        log.info("Step 5: ArgMax");
        Nd4j.getExecutioner().exec(new ArgMax(tad));

        log.info("Step 6: Sqrt");
        tad.addi(1.0f); // ensure positive
        Nd4j.getExecutioner().exec(new Sqrt(tad, tad));

        log.info("Step 7: dup");
        INDArray x1 = tad.dup(tad.ordering());
        INDArray x2 = tad.dup(tad.ordering());
        INDArray x3 = tad.dup('c');
        INDArray x4 = tad.dup('f');

        log.info("Step 8: vstack");
        Nd4j.vstack(tad, x1, x2, x3, x4);

        log.info("Step 9: hstack");
        Nd4j.hstack(tad, x1, x2, x3, x4);

        log.info("Step 10: ManhattanDistance");
        Nd4j.getExecutioner().exec(new ManhattanDistance(tad, x2));

        log.info("Step 11: toFlattened");
        Nd4j.toFlattened(tad, x1, x2, x3, x4);

        log.info("Step 12: max(0), max(1)");
        tad.max(0);
        tad.max(1);

        log.info("Step 13: argMax(0), argMax(1)");
        Nd4j.argMax(tad, 0);
        Nd4j.argMax(tad, 1);

        log.info("Step 14: SoftMax");
        Nd4j.getExecutioner().exec((CustomOp) new SoftMax(tad));

        log.info("Step 15: LogSoftMax");
        Nd4j.getExecutioner().exec((CustomOp) new LogSoftMax(tad));

        log.info("Step 16: BooleanIndexing.replaceWhere");
        BooleanIndexing.replaceWhere(tad, 5f, Conditions.lessThan(8f));

        log.info("Step 17: BooleanIndexing.assignIf");
        BooleanIndexing.assignIf(tad, x1, Conditions.greaterThan(-1000000000f));

        log.info("Step 18: stdNumber");
        tad.stdNumber().floatValue();

        log.info("Step 19: std(0), std(1)");
        tad.std(0);
        tad.std(1);

        // Step 20: THIS IS THE CRASH
        // float dot = (float) Nd4j.getBlasWrapper().dot(tad, x1);
        // BLAS sdot is called with N=8192, incX=1024 (stride(-1) of [128,64] f-order view with strides [8,1024])
        // This accesses memory at offsets 0, 1024, 2048, ..., 8191*1024 = 8,387,584
        // The buffer only has 65536 elements -> OUT OF BOUNDS -> SIGSEGV in libmkl_def.so
        log.warn("Step 20 (CRASH): Nd4j.getBlasWrapper().dot(tad, x1) would SIGSEGV here");
        log.warn("  TAD strides: {}", java.util.Arrays.toString(tad.stride()));
        log.warn("  getBlasStride(tad) = {} (returns stride(-1) = stride(1) = 1024 for 2D f-order)",
                BlasBufferUtil.getBlasStride(tad));
        log.warn("  BLAS sdot would access elements at offsets: 0, 1024, 2048, ..., {}*1024",
                tad.length() - 1);
        log.warn("  Buffer size: {} (only {} float elements)",
                tad.data().length(), tad.data().length());

        // Verify the bug analytically
        assertEquals(1024, BlasBufferUtil.getBlasStride(tad),
                "Confirming bug: getBlasStride returns 1024 (stride(1)) for non-contiguous f-order TAD view");

        log.info("All steps 1-19 passed. Step 20 (dot) confirmed to crash analytically.");
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
