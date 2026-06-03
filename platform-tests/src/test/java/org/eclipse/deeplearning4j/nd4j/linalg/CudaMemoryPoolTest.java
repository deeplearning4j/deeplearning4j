package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal test to isolate CudaMemoryPool heap corruption.
 *
 * The VLM test crashes with "corrupted size vs. prev_size" right after
 * CudaMemoryPool prints its pinned host memory limit during initialization.
 * This test exercises the same code path in isolation.
 */
@Slf4j
@Tag(TagNames.SMOKE)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class CudaMemoryPoolTest {

    @Test
    @Order(1)
    @DisplayName("Basic CUDA init - triggers CudaMemoryPool singleton construction")
    public void testBasicCudaInit() {
        // This triggers full CUDA backend initialization including CudaMemoryPool
        log.info("Step 1: Creating a simple array to trigger CUDA init");
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 10);
        assertNotNull(arr);
        log.info("Step 1 passed: array shape={}, backend={}", arr.shape(), Nd4j.getBackend().getClass().getSimpleName());
    }

    @Test
    @Order(2)
    @DisplayName("Allocate and free small arrays")
    public void testSmallAllocFree() {
        log.info("Step 2: Small alloc/free cycle");
        for (int i = 0; i < 100; i++) {
            INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
            arr.close();
        }
        log.info("Step 2 passed: 100 small alloc/free cycles");
    }

    @Test
    @Order(3)
    @DisplayName("Allocate and free large arrays - exercises pool reuse")
    public void testLargeAllocFree() {
        log.info("Step 3: Large alloc/free cycle");
        for (int i = 0; i < 10; i++) {
            INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024 * 1024); // 4MB
            arr.close();
        }
        log.info("Step 3 passed: 10 large alloc/free cycles");
    }

    @Test
    @Order(4)
    @DisplayName("Trim memory pool")
    public void testTrimPool() {
        log.info("Step 4: Trim memory pool");
        // Allocate some memory then free it
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 10 * 1024 * 1024); // 40MB
        arr.close();

        // Trim pool to release memory back to driver
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPool(0);
        log.info("Step 4 passed: trim completed");
    }

    @Test
    @Order(5)
    @DisplayName("Concurrent alloc/free from multiple threads")
    public void testConcurrentAllocFree() throws Exception {
        log.info("Step 5: Concurrent alloc/free");
        int threadCount = 4;
        int opsPerThread = 50;
        Thread[] threads = new Thread[threadCount];
        Exception[] errors = new Exception[threadCount];

        for (int t = 0; t < threadCount; t++) {
            final int threadId = t;
            threads[t] = new Thread(() -> {
                try {
                    for (int i = 0; i < opsPerThread; i++) {
                        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024 * (threadId + 1));
                        // Do a simple op to ensure GPU execution
                        arr.addi(1.0);
                        arr.close();
                    }
                } catch (Exception e) {
                    errors[threadId] = e;
                }
            });
            threads[t].start();
        }

        for (Thread t : threads) {
            t.join();
        }

        for (int i = 0; i < threadCount; i++) {
            if (errors[i] != null) {
                fail("Thread " + i + " failed: " + errors[i].getMessage());
            }
        }
        log.info("Step 5 passed: {} threads x {} ops", threadCount, opsPerThread);
    }

    @Test
    @Order(6)
    @DisplayName("Multi-device allocation if available")
    public void testMultiDeviceAlloc() {
        int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
        log.info("Step 6: Multi-device test with {} devices", deviceCount);

        if (deviceCount < 2) {
            log.info("Skipping multi-device test (only {} device)", deviceCount);
            return;
        }

        // Allocate on device 0
        INDArray arr0 = Nd4j.zeros(DataType.FLOAT, 1024);
        log.info("Allocated on device 0: shape={}", arr0.shape());

        // Trim both device pools
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPool(0);
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPool(1);

        arr0.close();
        log.info("Step 6 passed");
    }

    @Test
    @Order(7)
    @DisplayName("Rapid alloc/free stress test - exposes heap corruption")
    public void testRapidAllocFreeStress() {
        log.info("Step 7: Rapid alloc/free stress test");
        // This pattern mimics what happens during ONNX model import
        // where many small arrays are created and destroyed rapidly
        for (int round = 0; round < 5; round++) {
            INDArray[] arrays = new INDArray[100];
            // Allocate batch
            for (int i = 0; i < arrays.length; i++) {
                arrays[i] = Nd4j.zeros(DataType.FLOAT, 64 + i * 16);
            }
            // Free in reverse order
            for (int i = arrays.length - 1; i >= 0; i--) {
                arrays[i].close();
            }
            log.info("  Round {} complete", round);
        }
        log.info("Step 7 passed");
    }

    @Test
    @Order(8)
    @DisplayName("Mixed size allocations - mimics model loading pattern")
    public void testMixedSizeModelLoadPattern() {
        log.info("Step 8: Mixed size model load pattern");
        // Model loading creates many different-sized constant arrays (weights)
        // plus shape info buffers, then runs ops creating intermediates
        INDArray[] weights = new INDArray[50];
        for (int i = 0; i < weights.length; i++) {
            // Simulate varying weight sizes: 256 to 4M elements
            int size = 256 * (1 << (i % 14));
            weights[i] = Nd4j.zeros(DataType.FLOAT, Math.min(size, 1024 * 1024));
        }

        // Simulate forward pass intermediates
        for (int pass = 0; pass < 3; pass++) {
            INDArray intermediate = Nd4j.zeros(DataType.FLOAT, 64 * 576);
            INDArray result = intermediate.addi(1.0f);
            result.close();
            intermediate.close();
        }

        // Free weights
        for (INDArray w : weights) {
            w.close();
        }
        log.info("Step 8 passed");
    }
}
