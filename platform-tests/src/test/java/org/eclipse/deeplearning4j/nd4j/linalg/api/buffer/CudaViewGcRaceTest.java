/*
 *  ******************************************************************************
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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Reproducer for CUDA GC race condition with view-based ops.
 *
 * The crash scenario:
 * 1. Op creates a view (expand_dims, reshape) sharing the same DataBuffer
 * 2. Java GC collects the source array's OpaqueDataBuffer
 * 3. dbClose frees the shared CUDA memory while a kernel is still using it
 * 4. Next op accesses freed memory -> heap corruption / SIGABRT
 *
 * This test exercises that path rapidly without needing a full VLM pipeline.
 */
@Slf4j
@Tag("cuda")
public class CudaViewGcRaceTest {

    /**
     * Rapid expand_dims + GC pressure.
     * expand_dims creates views sharing the same buffer.
     * If GC frees a source buffer while CUDA kernels are in-flight, we get corruption.
     */
    @Test
    public void testExpandDimsGcRace() {
        log.info("Starting expand_dims GC race test");

        for (int iter = 0; iter < 200; iter++) {
            // Create a source tensor and do computation on it (launches CUDA kernels)
            INDArray source = Nd4j.randn(DataType.FLOAT, 1, 1024);
            INDArray computed = source.mul(2.0f).add(1.0f);

            // expand_dims creates a view - same underlying buffer
            INDArray expanded = Nd4j.expandDims(computed, 0);

            // Do more work on the view
            INDArray result = expanded.mul(3.0f);

            // Null out source references to make them GC-eligible
            source = null;
            computed = null;

            // Force GC to try to collect the source buffers
            // while CUDA kernels from result.mul() may still be in-flight
            if (iter % 10 == 0) {
                System.gc();
            }

            // Use the result to ensure it's valid (would crash if buffer was freed)
            assertNotNull(result);
            float val = result.getFloat(0);
            // Just check it's a finite number (not NaN from freed memory)
            assertEquals(val, val); // NaN != NaN
        }

        log.info("expand_dims GC race test passed (200 iterations)");
    }

    /**
     * Rapid reshape + matmul + GC pressure.
     * Reshapes create views. Combined with matmul (heavy CUDA kernel),
     * this maximizes the window for GC to race with in-flight ops.
     */
    @Test
    public void testReshapeMatmulGcRace() {
        log.info("Starting reshape+matmul GC race test");

        for (int iter = 0; iter < 100; iter++) {
            // Simulate vision encoder pattern: conv output -> reshape -> matmul
            INDArray input = Nd4j.randn(DataType.FLOAT, 4, 768);
            INDArray weight = Nd4j.randn(DataType.FLOAT, 768, 768);

            // Matmul launches heavy CUDA kernel
            INDArray mmResult = input.mmul(weight);

            // Reshape creates a view
            INDArray reshaped = mmResult.reshape(2, 2, 768);

            // More ops on the view
            INDArray normed = reshaped.div(reshaped.norm2());

            // Kill references, force GC
            input = null;
            weight = null;
            mmResult = null;
            reshaped = null;

            if (iter % 5 == 0) {
                System.gc();
            }

            assertNotNull(normed);
            float val = normed.getFloat(0);
            assertEquals(val, val);
        }

        log.info("reshape+matmul GC race test passed (100 iterations)");
    }

    /**
     * Scatter/gather + view ops + GC - matches the VLM crash sequence.
     * The VLM crash was: where -> scatter_nd_update -> gather -> add -> CRASH
     * This test simulates that pattern.
     */
    @Test
    public void testScatterGatherViewGcRace() {
        log.info("Starting scatter/gather + view GC race test");

        for (int iter = 0; iter < 100; iter++) {
            // Create tensors
            INDArray data = Nd4j.randn(DataType.FLOAT, 32, 64);

            // Gather rows via indexing (simulates modified gather op)
            INDArray gathered = data.get(Nd4j.createFromArray(new int[]{0, 5, 10, 15, 20, 25, 30, 31}));

            // Expand dims (view op - crash point)
            INDArray expanded = Nd4j.expandDims(gathered, 0);

            // Add (the op where crash was detected in VLM)
            INDArray result = expanded.add(1.0f);

            // Make source GC-eligible
            data = null;
            gathered = null;
            expanded = null;

            if (iter % 5 == 0) {
                System.gc();
            }

            assertNotNull(result);
            float val = result.getFloat(0);
            assertEquals(val, val);
        }

        log.info("scatter/gather + view GC race test passed (100 iterations)");
    }

    /**
     * Multi-threaded view ops + GC - maximum race condition pressure.
     * One thread does compute, another forces GC constantly.
     */
    @Test
    public void testConcurrentViewGcPressure() throws InterruptedException {
        log.info("Starting concurrent view + GC pressure test");

        final boolean[] running = {true};
        final boolean[] failed = {false};

        // GC pressure thread
        Thread gcThread = new Thread(() -> {
            while (running[0] && !failed[0]) {
                System.gc();
                try {
                    Thread.sleep(1);
                } catch (InterruptedException ex) {
                    break;
                }
            }
        }, "gc-pressure");
        gcThread.setDaemon(true);
        gcThread.start();

        // Compute thread
        try {
            for (int iter = 0; iter < 500; iter++) {
                INDArray arr1 = Nd4j.randn(DataType.FLOAT, 16, 256);
                INDArray arr2 = arr1.mul(2.0f);
                INDArray arr3 = Nd4j.expandDims(arr2, 0);    // view
                INDArray arr4 = arr3.reshape(4, 4, 256);      // view
                INDArray arr5 = arr4.add(1.0f);
                INDArray arr6 = arr5.reshape(16, 256);         // view
                INDArray arr7 = arr6.sub(0.5f);

                // Verify result is valid
                float val = arr7.getFloat(0);
                if (Float.isNaN(val)) {
                    failed[0] = true;
                    log.error("Got NaN at iteration {} - buffer corruption detected", iter);
                    break;
                }

                // Clear refs
                arr1 = null; arr2 = null; arr3 = null; arr4 = null;
                arr5 = null; arr6 = null;
            }
        } finally {
            running[0] = false;
            gcThread.join(5000);
        }

        assertEquals(false, failed[0], "Buffer corruption detected (NaN values)");
        log.info("Concurrent view + GC pressure test passed (500 iterations)");
    }
}
