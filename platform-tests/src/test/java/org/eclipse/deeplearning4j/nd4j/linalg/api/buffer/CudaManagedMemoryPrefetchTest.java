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

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for cudaMallocManaged prefetch behavior in CudaMemoryPool::allocateFailover().
 *
 * <p>Production failure mode (BGE-M3 embedding, June 2026):
 * <ol>
 *   <li>RTX 4090 (device 0) at 99.5% memory usage with large model loaded</li>
 *   <li>dot_product_attention_v2 allocates 256MB output buffer</li>
 *   <li>allocateFailover() routes to device 1 (RTX 3070 Ti) via cudaMallocManaged</li>
 *   <li>cudaMemAdvise sets preferred location to device 0 but does NOT prefetch pages</li>
 *   <li>Attention kernel on device 0 writes to managed memory — pages physically on device 1</li>
 *   <li>Non-peer devices (no NVLink): demand paging triggers CUDA error 700</li>
 *   <li>Error 700 is sticky — poisons ALL subsequent CUDA calls including cudaMemGetInfo</li>
 *   <li>Device scanning loop can't discover device 1's 7GB free → "All GPUs exhausted"</li>
 * </ol>
 *
 * <p>Fix: cudaMemPrefetchAsync after cudaMemAdvise pre-migrates pages to the requesting
 * device before the kernel touches them. Plus: cudaGetLastError() before cudaMemGetInfo
 * in the device scanning loop clears sticky errors.
 *
 * @see org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer
 */
@Slf4j
@DisplayName("CUDA Managed Memory Prefetch Tests")
public class CudaManagedMemoryPrefetchTest extends BaseNd4jTestWithBackends {

    private DeviceMemoryManager memoryManager;

    @Override
    public char ordering() {
        return 'c';
    }

    private boolean isCudaBackend() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        return backendName.contains("cuda") || backendName.contains("jcublas");
    }

    private int getNumDevices() {
        try {
            return Nd4j.getAffinityManager().getNumberOfDevices();
        } catch (Exception e) {
            return 0;
        }
    }

    @BeforeEach
    void setUp() {
        memoryManager = DeviceMemoryManager.getInstance();
    }

    @AfterEach
    void tearDown() {
        if (memoryManager != null) {
            memoryManager.clearAllMemorySimulation();
        }
        System.gc();
    }

    /**
     * Exercises the exact production failure: dot_product_attention with output buffers
     * allocated under memory pressure that forces failover to a non-peer device.
     *
     * On a multi-GPU system, this forces the allocateFailover path. On a single-GPU
     * system, the test still verifies that attention computes correctly under memory
     * pressure (the failover goes to pinned host memory instead).
     *
     * The key assertion: the attention output is NOT NaN/zero/corrupted, which would
     * indicate error 700 from accessing managed memory without prefetch.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention output correct under memory-pressure failover (production BGE-M3 scenario)")
    void testAttentionUnderMemoryPressureFailover(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        int numDevices = getNumDevices();
        assumeTrue(numDevices > 0, "Test requires at least 1 CUDA device");

        // Dimensions matching production BGE-M3: batch=1, seqQ=1 (decode step),
        // seqKV=512 (context), heads=16, headDim=64
        int batch = 1, seqQ = 1, seqKV = 512, heads = 16, headDim = 64;
        double scale = 1.0 / Math.sqrt(headDim);

        // First run without memory pressure to get a reference result
        INDArray refOutput;
        INDArray qData = Nd4j.randn(DataType.FLOAT, batch, seqQ, heads, headDim).muli(0.1);
        INDArray kData = Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1);
        INDArray vData = Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1);

        {
            SameDiff sd = SameDiff.create();
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, heads, headDim);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqKV, heads, headDim);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqKV, heads, headDim);
            SDVariable y = sd.nn.dotProductAttentionV2("y", q, v, k, null, null,
                    scale, 0.0, true, false);
            sd.setOutputs("y");

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("q", qData);
            inputs.put("k", kData);
            inputs.put("v", vData);

            Map<String, INDArray> out = sd.output(inputs, "y");
            refOutput = out.get("y").dup();
            sd.close();
        }

        assertFalse(refOutput.isNaN().any(), "Reference output should not contain NaN");
        assertFalse(refOutput.amean().getDouble(0) == 0.0, "Reference output should not be all zeros");
        log.info("Reference attention output: shape={}, mean={}, max={}",
                refOutput.shape(), refOutput.amean().getDouble(0), refOutput.amax().getDouble(0));

        // Now simulate memory pressure: device 0 nearly full, forcing failover
        memoryManager.setSimulatedFreeMemory(0, 50L * 1024 * 1024); // 50MB free on device 0
        if (numDevices > 1) {
            // Device 1 has plenty of free memory — this is the failover target
            memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
        }
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Run the same attention under memory pressure
        INDArray pressureOutput;
        {
            SameDiff sd = SameDiff.create();
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, heads, headDim);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqKV, heads, headDim);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqKV, heads, headDim);
            SDVariable y = sd.nn.dotProductAttentionV2("y", q, v, k, null, null,
                    scale, 0.0, true, false);
            sd.setOutputs("y");

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("q", qData);
            inputs.put("k", kData);
            inputs.put("v", vData);

            Map<String, INDArray> out = sd.output(inputs, "y");
            pressureOutput = out.get("y").dup();
            sd.close();
        }

        memoryManager.clearAllMemorySimulation();

        // The critical assertion: output must NOT be corrupted
        assertFalse(pressureOutput.isNaN().any(),
                "Attention output under memory pressure contains NaN — likely CUDA error 700 from unPrefetched managed memory");
        assertFalse(pressureOutput.amean().getDouble(0) == 0.0,
                "Attention output under memory pressure is all zeros — likely CUDA error 700 poisoned the context");
        log.info("Pressure attention output: shape={}, mean={}, max={}",
                pressureOutput.shape(), pressureOutput.amean().getDouble(0), pressureOutput.amax().getDouble(0));

        // Outputs should match (same inputs, same graph, memory pressure should not affect correctness)
        INDArray diff = refOutput.sub(pressureOutput).amean();
        double meanDiff = diff.getDouble(0);
        log.info("Mean absolute difference between reference and pressure output: {}", meanDiff);
        assertTrue(meanDiff < 1e-4,
                "Attention output differs under memory pressure (mean diff=" + meanDiff
                        + "). This indicates the failover allocation path corrupted the output.");
    }

    /**
     * Verifies that after a failover allocation (potentially via managed memory),
     * subsequent allocations on the original device still work. This catches the
     * "sticky error 700" problem: if error 700 was triggered by accessing managed
     * memory without prefetch, it poisons cudaMemGetInfo and prevents ALL future
     * allocations from discovering available devices.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Post-failover allocations succeed (sticky error 700 resilience)")
    void testPostFailoverAllocationsSucceed(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        int numDevices = getNumDevices();
        assumeTrue(numDevices > 0, "Test requires at least 1 CUDA device");

        // Phase 1: Force a failover allocation by simulating pressure on device 0
        memoryManager.setSimulatedFreeMemory(0, 10L * 1024 * 1024); // 10MB free
        if (numDevices > 1) {
            memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
        }
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        // Allocate a large buffer that should trigger failover
        INDArray failoverArray = Nd4j.create(DataType.FLOAT, 64 * 1024 * 1024); // 256MB
        assertNotNull(failoverArray, "Failover allocation should succeed");

        // Run a matmul on it to exercise the kernel path (forces actual device memory access)
        INDArray small = Nd4j.ones(DataType.FLOAT, 1);
        INDArray result = failoverArray.reshape(8192, 8192).mmul(Nd4j.ones(DataType.FLOAT, 8192, 1));
        assertFalse(result.isNaN().any(), "MatMul on failover-allocated buffer should not produce NaN");

        // Phase 2: Clear pressure simulation — device 0 has plenty of memory again
        memoryManager.clearAllMemorySimulation();

        // Phase 3: Allocate on the original device — should succeed if error 700
        // wasn't triggered (or if the sticky-error fix clears it properly)
        INDArray normalArray = Nd4j.create(DataType.FLOAT, 1024 * 1024); // 4MB
        assertNotNull(normalArray, "Post-failover allocation should succeed");
        normalArray.assign(42.0);
        assertEquals(42.0, normalArray.getDouble(0), 1e-6,
                "Post-failover array should be writable and readable");

        // Phase 4: Run another SameDiff graph to verify DSP execution is healthy
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 3, 4));
        SDVariable out = sd.mmul("out", x, w);
        sd.setOutputs("out");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, 2, 3));
        Map<String, INDArray> outputs = sd.output(inputs, "out");
        INDArray postFailoverOutput = outputs.get("out");

        assertFalse(postFailoverOutput.isNaN().any(),
                "SameDiff execution after failover should not produce NaN — indicates CUDA context corruption");
        log.info("Post-failover SameDiff execution succeeded: output shape={}, mean={}",
                postFailoverOutput.shape(), postFailoverOutput.amean().getDouble(0));

        sd.close();
    }

    /**
     * Verifies repeated failover allocations don't accumulate errors.
     * The production scenario involved multiple batches: batch 1 triggers failover,
     * batches 2-N trigger error 700 cascade from sticky errors.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Repeated failover allocations with kernel execution (multi-batch scenario)")
    void testRepeatedFailoverAllocationsWithKernelExecution(Nd4jBackend backend) {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");
        int numDevices = getNumDevices();
        assumeTrue(numDevices > 0, "Test requires at least 1 CUDA device");

        int numBatches = 5;
        int batchSize = 4;
        int seqLen = 128;
        int hiddenDim = 256;

        // Simulate memory pressure throughout
        memoryManager.setSimulatedFreeMemory(0, 30L * 1024 * 1024); // 30MB free
        if (numDevices > 1) {
            memoryManager.setSimulatedFreeMemory(1, 8L * 1024 * 1024 * 1024);
        }
        memoryManager.setSimulatedFreeMemory(DeviceMemoryManager.CPU_DEVICE_ID, 32L * 1024 * 1024 * 1024);
        memoryManager.setMemorySimulationEnabled(true);

        for (int batch = 0; batch < numBatches; batch++) {
            // Each batch: create input, run matmul, verify output
            INDArray input = Nd4j.randn(DataType.FLOAT, batchSize, seqLen, hiddenDim).muli(0.01);
            INDArray weight = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).muli(0.01);

            // Reshape for matmul: [batchSize*seqLen, hiddenDim] x [hiddenDim, hiddenDim]
            INDArray reshaped = input.reshape(batchSize * seqLen, hiddenDim);
            INDArray output = reshaped.mmul(weight);

            assertFalse(output.isNaN().any(),
                    "Batch " + batch + " matmul output contains NaN — failover allocation corrupted");
            assertTrue(output.amean().getDouble(0) > 0 || output.amean().getDouble(0) < 0,
                    "Batch " + batch + " matmul output is exactly zero — unexpected");

            log.info("Batch {}/{}: matmul under pressure succeeded, output mean={}",
                    batch + 1, numBatches, output.amean().getDouble(0));

            // Explicitly close intermediate arrays to exercise buffer return path
            input.close();
            weight.close();
            output.close();
        }

        memoryManager.clearAllMemorySimulation();

        // Final sanity: normal allocation still works
        INDArray sanity = Nd4j.create(DataType.FLOAT, 1000);
        sanity.assign(1.0);
        assertEquals(1.0, sanity.getDouble(0), 1e-6, "Post-batch allocations should work normally");
    }
}
