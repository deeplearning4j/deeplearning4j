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
package org.eclipse.deeplearning4j.nd4j.linalg.gpu;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.condition.EnabledIf;
import static org.junit.jupiter.api.Assertions.*;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Tests for CUDA shapeInfo race conditions.
 *
 * These tests target the following race conditions identified in NDArray:
 *
 * 1. TOCTOU (Time-of-Check-Time-of-Use) in shapeInfo():
 *    - Thread A reads _shapeInfoBuffer (non-null)
 *    - Thread B destructor clears _shapeInfo = nullptr
 *    - Thread A refreshes _shapeInfo from buffer
 *    - Result: Use-after-free or nullptr dereference
 *
 * 2. Non-atomic pointer clearing in destructor:
 *    - _shapeInfo is cleared before _shapeInfoBuffer
 *    - Creates window where buffer is valid but shapeInfo is null
 *
 * 3. Concurrent shape access during array operations:
 *    - Multiple threads access shapeInfo() while arrays are being created/destroyed
 *
 * Run with CUDA compute-sanitizer for memory checking:
 *   -Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer
 *
 * @see libnd4j/include/array/NDArray.h:2083-2133 (shapeInfo method)
 * @see libnd4j/include/array/NDArray.hXX:800-821 (destructor)
 */
@NativeTag
public class CudaShapeInfoRaceConditionTests {

    private static final int NUM_THREADS = 32;
    private static final int ITERATIONS_PER_THREAD = 100;

    static boolean isCudaAvailable() {
        try {
            return "CUDA".equalsIgnoreCase(Nd4j.getExecutioner().getEnvironmentInformation()
                    .getProperty("backend"));
        } catch (Exception e) {
            return false;
        }
    }

    @BeforeEach
    public void setUp() {
        // Force GC to clean up any dangling references
        System.gc();
    }

    @AfterEach
    public void tearDown() {
        System.gc();
    }

    /**
     * Test 1: Concurrent shapeInfo access while creating/destroying arrays
     *
     * This test creates a producer thread that rapidly creates and destroys arrays
     * while consumer threads try to access shapeInfo() on shared arrays.
     *
     * Targets: TOCTOU race in shapeInfo() refresh logic
     */
    @RepeatedTest(5)
    public void testConcurrentShapeInfoAccessDuringCreationDestruction() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<Throwable> firstError = new AtomicReference<>(null);

        // Shared volatile reference to array being tested
        final AtomicReference<INDArray> sharedArray = new AtomicReference<>();

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS + 1);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS + 1);

        // Producer thread: rapidly creates and destroys arrays
        executor.submit(() -> {
            try {
                startLatch.await();
                for (int i = 0; i < ITERATIONS_PER_THREAD * 10; i++) {
                    // Create array
                    INDArray arr = Nd4j.create(DataType.FLOAT, 100, 100);
                    arr.assign(i);
                    sharedArray.set(arr);

                    // Brief pause to allow other threads to access
                    Thread.yield();

                    // Clear and let GC potentially collect
                    sharedArray.set(null);

                    if (i % 50 == 0) {
                        System.gc();
                    }
                }
            } catch (Throwable e) {
                firstError.compareAndSet(null, e);
                failures.incrementAndGet();
            } finally {
                finishLatch.countDown();
            }
        });

        // Consumer threads: try to access shapeInfo on shared array
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        INDArray arr = sharedArray.get();
                        if (arr != null) {
                            try {
                                // This call should either succeed or throw a clean exception
                                // It should NOT cause a JVM crash (SIGSEGV)
                                long[] shape = arr.shape();
                                long rank = arr.rank();
                                long length = arr.length();

                                // Validate consistency
                                if (shape != null && shape.length != rank) {
                                    String msg = "Thread " + threadId + ": Shape length " +
                                                shape.length + " != rank " + rank;
                                    firstError.compareAndSet(null, new AssertionError(msg));
                                    failures.incrementAndGet();
                                }
                            } catch (IllegalStateException | NullPointerException e) {
                                // Expected if array was destroyed - this is OK
                            }
                        }
                        Thread.yield();
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, e);
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(60, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out - possible deadlock");

        if (firstError.get() != null) {
            fail("Concurrent access caused error: " + firstError.get().getMessage(), firstError.get());
        }
    }

    /**
     * Test 2: Stress test for shapeInfo refresh mechanism
     *
     * Multiple threads simultaneously access shapeInfo() on the same array
     * while one thread modifies it via reshape operations.
     *
     * Targets: _shapeInfo refresh race in NDArray::shapeInfo()
     */
    @RepeatedTest(5)
    public void testShapeInfoRefreshRace() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        // Create base array
        INDArray base = Nd4j.create(DataType.FLOAT, 12, 12).assign(1.0);

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        // Access shapeInfo multiple times
                        long[] shape1 = base.shape();
                        long rank1 = base.rank();

                        // Create view and access its shapeInfo
                        INDArray view = base.getRow(threadId % 12);
                        long[] viewShape = view.shape();
                        long viewRank = view.rank();

                        // Validate view properties
                        if (viewRank != 1) {
                            String msg = "Thread " + threadId + ": View rank " + viewRank + " != 1";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        if (viewShape.length != 1 || viewShape[0] != 12) {
                            String msg = "Thread " + threadId + ": View shape invalid at iteration " + i;
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Access shape again to trigger potential refresh race
                        long[] shape2 = base.shape();
                        if (!java.util.Arrays.equals(shape1, shape2)) {
                            String msg = "Thread " + threadId + ": Base shape changed unexpectedly";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(60, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }

    /**
     * Test 3: Concurrent array creation and destruction stress test
     *
     * Tests the destructor's non-atomic pointer clearing by having
     * many threads create and destroy arrays simultaneously.
     *
     * Targets: Race between _shapeInfo and _shapeInfoBuffer clearing in destructor
     */
    @RepeatedTest(5)
    public void testDestructorRace() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        // Rapid create/use/discard cycle
                        INDArray arr = Nd4j.create(DataType.FLOAT, 50 + threadId, 50);

                        // Access shape info (triggers shapeInfo() call)
                        long[] shape = arr.shape();
                        long rank = arr.rank();

                        // Do some operations that require shape access
                        INDArray result = arr.add(1.0);

                        // Validate
                        if (rank != 2) {
                            String msg = "Thread " + threadId + ": Rank " + rank + " != 2";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        if (shape[0] != 50 + threadId || shape[1] != 50) {
                            String msg = "Thread " + threadId + ": Shape mismatch";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Let array go out of scope, trigger potential destructor race
                        arr = null;
                        result = null;

                        if (i % 20 == 0) {
                            System.gc();
                        }
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(120, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }

    /**
     * Test 4: Multi-device concurrent access (for multi-GPU systems)
     *
     * Creates arrays on different devices and accesses them from multiple threads.
     *
     * Targets: Device-specific shapeInfoD race conditions
     */
    @Test
    public void testMultiDeviceShapeInfoAccess() throws Exception {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices < 1) {
            return; // Skip if no CUDA devices
        }

        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        // Create arrays per device
        List<INDArray> deviceArrays = new ArrayList<>();
        for (int d = 0; d < numDevices; d++) {
            INDArray arr = Nd4j.create(DataType.FLOAT, 100, 100).assign(d);
            deviceArrays.add(arr);
        }

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        // Access array from random device
                        int deviceIdx = (threadId + i) % deviceArrays.size();
                        INDArray arr = deviceArrays.get(deviceIdx);

                        // Multiple shape accesses
                        long[] shape = arr.shape();
                        long rank = arr.rank();
                        long length = arr.length();

                        // Validate
                        if (rank != 2) {
                            String msg = "Thread " + threadId + ": Device " + deviceIdx +
                                        " array rank " + rank + " != 2";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        if (length != 10000) {
                            String msg = "Thread " + threadId + ": Device " + deviceIdx +
                                        " array length " + length + " != 10000";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Cross-device operation (if available)
                        if (deviceArrays.size() > 1) {
                            int otherIdx = (deviceIdx + 1) % deviceArrays.size();
                            INDArray other = deviceArrays.get(otherIdx);

                            // This should trigger device synchronization
                            INDArray result = arr.add(other);
                            if (result.rank() != 2) {
                                String msg = "Thread " + threadId + ": Cross-device result rank invalid";
                                firstError.compareAndSet(null, msg);
                                failures.incrementAndGet();
                            }
                        }
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(120, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }

    /**
     * Test 5: View creation race condition
     *
     * Tests concurrent view creation on the same base array.
     * Views share shapeInfoBuffer with base array - potential for race.
     *
     * Targets: Shared ConstantShapeBuffer access between views and base
     */
    @RepeatedTest(5)
    public void testConcurrentViewCreation() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        // Create base array
        INDArray base = Nd4j.create(DataType.FLOAT, 100, 100).assign(1.0);

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        // Create various views concurrently
                        int rowIdx = (threadId + i) % 100;
                        int colIdx = (threadId * 7 + i) % 100;

                        // Row view
                        INDArray rowView = base.getRow(rowIdx);
                        if (rowView.length() != 100) {
                            String msg = "Thread " + threadId + ": Row view length " +
                                        rowView.length() + " != 100";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Column view
                        INDArray colView = base.getColumn(colIdx);
                        if (colView.length() != 100) {
                            String msg = "Thread " + threadId + ": Col view length " +
                                        colView.length() + " != 100";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Subarray view
                        int startRow = rowIdx;
                        int endRow = Math.min(startRow + 10, 100);
                        int startCol = colIdx;
                        int endCol = Math.min(startCol + 10, 100);

                        INDArray subView = base.get(
                                NDArrayIndex.interval(startRow, endRow),
                                NDArrayIndex.interval(startCol, endCol)
                        );

                        long[] expectedShape = new long[]{endRow - startRow, endCol - startCol};
                        if (!Shape.shapeEquals(subView.shape(), expectedShape)) {
                            String msg = "Thread " + threadId + ": Subview shape mismatch";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Operations on views
                        double rowSum = rowView.sumNumber().doubleValue();
                        double colSum = colView.sumNumber().doubleValue();

                        // All values are 1.0, so sums should be 100.0
                        if (Math.abs(rowSum - 100.0) > 0.01) {
                            String msg = "Thread " + threadId + ": Row sum " + rowSum + " != 100.0";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(60, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }

    /**
     * Test 6: ConstantShapeBuffer reference counting race
     *
     * Stress tests the addRef()/release() mechanism by creating many
     * arrays that share the same shape (and thus the same ConstantShapeBuffer).
     *
     * Targets: ConstantShapeBuffer reference counting thread safety
     */
    @RepeatedTest(5)
    public void testConstantShapeBufferRefCountRace() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        // All arrays will have same shape, sharing ConstantShapeBuffer
        final long[] sharedShape = new long[]{64, 64};

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();

                    List<INDArray> localArrays = new ArrayList<>();

                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        // Create array with shared shape
                        INDArray arr = Nd4j.create(DataType.FLOAT, sharedShape).assign(threadId);
                        localArrays.add(arr);

                        // Access shapeInfo to trigger buffer refresh
                        long[] shape = arr.shape();
                        if (!Shape.shapeEquals(shape, sharedShape)) {
                            String msg = "Thread " + threadId + ": Shape mismatch at iteration " + i;
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Periodically clear some arrays to trigger release()
                        if (i % 10 == 9) {
                            int toClear = Math.min(5, localArrays.size());
                            for (int j = 0; j < toClear; j++) {
                                localArrays.remove(0);
                            }
                            System.gc();
                        }
                    }

                    // Validate remaining arrays
                    for (INDArray arr : localArrays) {
                        long[] shape = arr.shape();
                        if (!Shape.shapeEquals(shape, sharedShape)) {
                            String msg = "Thread " + threadId + ": Final shape validation failed";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }

                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(120, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }

    /**
     * Test 7: Copy/move constructor race condition
     *
     * Tests concurrent copying of arrays to stress copy constructor
     * and move semantics.
     *
     * Targets: Copy constructor and operator= race conditions
     */
    @RepeatedTest(5)
    public void testCopyMoveRace() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        // Source arrays
        List<INDArray> sourceArrays = new ArrayList<>();
        for (int i = 0; i < NUM_THREADS; i++) {
            sourceArrays.add(Nd4j.create(DataType.FLOAT, 32, 32).assign(i));
        }

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
                        int sourceIdx = (threadId + i) % sourceArrays.size();
                        INDArray source = sourceArrays.get(sourceIdx);

                        // dup() triggers copy
                        INDArray copy = source.dup();

                        // Validate copy
                        if (!Shape.shapeEquals(copy.shape(), source.shape())) {
                            String msg = "Thread " + threadId + ": Copy shape mismatch";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // assign() may trigger move or copy
                        INDArray target = Nd4j.create(DataType.FLOAT, 32, 32);
                        target.assign(source);

                        // Validate assign
                        if (!Shape.shapeEquals(target.shape(), source.shape())) {
                            String msg = "Thread " + threadId + ": Assign shape mismatch";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        // Reshape (creates new shape buffer)
                        INDArray reshaped = source.reshape(1024);
                        if (reshaped.length() != 1024) {
                            String msg = "Thread " + threadId + ": Reshape length mismatch";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(120, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }

    /**
     * Test 8: CUDA stream concurrent operations
     *
     * Tests concurrent operations that use different CUDA streams,
     * which may access shapeInfo from different stream contexts.
     *
     * Targets: specialShapeInfo() race in CUDA path
     */
    @Test
    public void testCudaStreamConcurrentOperations() throws Exception {
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);

        // Create arrays for concurrent operations
        INDArray a = Nd4j.create(DataType.FLOAT, 256, 256).assign(1.0);
        INDArray b = Nd4j.create(DataType.FLOAT, 256, 256).assign(2.0);

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(NUM_THREADS);

        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    startLatch.await();

                    for (int i = 0; i < ITERATIONS_PER_THREAD / 2; i++) {
                        // Different operations that may use different streams
                        INDArray result;
                        switch (threadId % 4) {
                            case 0:
                                result = a.mmul(b); // Matrix multiply
                                break;
                            case 1:
                                result = a.add(b);  // Element-wise add
                                break;
                            case 2:
                                result = a.sub(b);  // Element-wise sub
                                break;
                            default:
                                result = a.mul(b);  // Element-wise mul
                                break;
                        }

                        // Validate result shape
                        if (result.rank() != 2) {
                            String msg = "Thread " + threadId + ": Result rank " + result.rank() + " != 2";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }

                        if (result.shape()[0] != 256 || result.shape()[1] != 256) {
                            String msg = "Thread " + threadId + ": Result shape mismatch";
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }
                } catch (Throwable e) {
                    if (!(e instanceof InterruptedException)) {
                        firstError.compareAndSet(null, "Thread " + threadId + ": " + e.getMessage());
                        failures.incrementAndGet();
                    }
                } finally {
                    finishLatch.countDown();
                }
            });
        }

        startLatch.countDown();
        boolean completed = finishLatch.await(120, TimeUnit.SECONDS);
        executor.shutdownNow();

        assertTrue(completed, "Test timed out");
        assertEquals(0, failures.get(), firstError.get());
    }
}
