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
package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import static org.junit.jupiter.api.Assertions.*;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Tests for TAD operations under parallel execution, specifically targeting race conditions
 * in the TAD creation and caching code related to broadcast operations.
 */
@Execution(ExecutionMode.CONCURRENT) // Force tests to run in parallel
public class TadParallelBroadcastTests {

    /**
     * Test for row view shape consistency under parallel execution
     * This test creates multiple threads that access the same row view and verifies shape
     */
    @Test
    public void testRowViewParallelShapeConsistency() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Run multiple threads that access the same row view
        int numThreads = 20;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    // Get row view and check its shape and length
                    INDArray rowView = matrix.getRow(1);
                    
                    if (rowView.rank() != 1) {
                        String msg = "Row view has incorrect rank: " + rowView.rank();
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                    if (rowView.length() != 3) {
                        String msg = "Row view has incorrect length: " + rowView.length() + " (expected 3)";
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                    if (!Shape.shapeEquals(rowView.shape(), new long[]{3})) {
                        String msg = "Row view has incorrect shape: " + rowView.shapeInfoToString();
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception occurred: " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "Row view parallel shape checks failed");
    }

    /**
     * Test for row view broadcast operations under parallel execution
     * This test creates multiple threads that perform broadcast operations on the same row view
     */
    @Test
    public void testRowViewParallelBroadcastOperations() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Create a vector for broadcasting
        INDArray vector = Nd4j.create(new double[] {10, 20, 30});
        
        int numThreads = 20;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            final int threadIdx = i;
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    // Get row view
                    INDArray rowView = matrix.getRow(1);
                    
                    // Perform different broadcast operations based on thread index
                    INDArray result;
                    INDArray expected;
                    
                    switch (threadIdx % 4) {
                        case 0: // Addition
                            result = rowView.add(vector);
                            expected = Nd4j.create(new double[] {14, 25, 36});
                            break;
                        case 1: // Subtraction
                            result = rowView.sub(vector);
                            expected = Nd4j.create(new double[] {-6, -15, -24});
                            break;
                        case 2: // Multiplication
                            result = rowView.mul(vector);
                            expected = Nd4j.create(new double[] {40, 100, 180});
                            break;
                        case 3: // Division
                            result = rowView.div(vector);
                            expected = Nd4j.create(new double[] {0.4, 0.25, 0.2});
                            break;
                        default:
                            throw new IllegalStateException("Unexpected thread index mod 4");
                    }
                    
                    if (!result.equalsWithEps(expected, 1e-5)) {
                        String msg = "Operation result mismatch for thread " + threadIdx + 
                                     ". Expected: " + expected + ", Got: " + result;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception in thread " + threadIdx + ": " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "Row view parallel broadcast operations failed");
    }

    /**
     * Test for in-place broadcast operations on row views in parallel
     */
    @Test
    public void testInPlaceRowViewParallelBroadcastOperations() throws Exception {
        int numMatrices = 20; // Create multiple independent matrices for parallel in-place operations
        List<INDArray> matrices = new ArrayList<>(numMatrices);
        
        // Create identical matrices
        for (int i = 0; i < numMatrices; i++) {
            matrices.add(Nd4j.create(new double[][] {
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
            }));
        }
        
        // Create a vector for broadcasting
        INDArray vector = Nd4j.create(new double[] {10, 20, 30});
        
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numMatrices);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numMatrices);
        
        for (int i = 0; i < numMatrices; i++) {
            final int matrixIdx = i;
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    INDArray matrix = matrices.get(matrixIdx);
                    INDArray rowView = matrix.getRow(1);
                    
                    // Perform in-place addition
                    rowView.addi(vector);
                    
                    // Verify row view result
                    INDArray expectedRow = Nd4j.create(new double[] {14, 25, 36});
                    if (!rowView.equalsWithEps(expectedRow, 1e-5)) {
                        String msg = "Row view result mismatch for matrix " + matrixIdx + 
                                     ". Expected: " + expectedRow + ", Got: " + rowView;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                    // Verify matrix was modified correctly
                    INDArray expectedMatrix = Nd4j.create(new double[][] {
                            {1, 2, 3},
                            {14, 25, 36},
                            {7, 8, 9}
                    });
                    
                    if (!matrix.equalsWithEps(expectedMatrix, 1e-5)) {
                        String msg = "Matrix modification mismatch for matrix " + matrixIdx + 
                                   ". Expected: " + expectedMatrix + ", Got: " + matrix;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception in matrix " + matrixIdx + ": " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "In-place row view parallel broadcast operations failed");
    }

    /**
     * Test for column view shape consistency under parallel execution
     */
    @Test
    public void testColumnViewParallelShapeConsistency() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Run multiple threads that access the same column view
        int numThreads = 20;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    // Get column view and check its shape and length
                    INDArray colView = matrix.getColumn(1);
                    
                    if (colView.rank() != 2) {
                        String msg = "Column view has incorrect rank: " + colView.rank();
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                    if (colView.length() != 3) {
                        String msg = "Column view has incorrect length: " + colView.length() + " (expected 3)";
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                    if (!Shape.shapeEquals(colView.shape(), new long[] {3, 1})) {
                        String msg = "Column view has incorrect shape: " + colView.shapeInfoToString();
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception occurred: " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "Column view parallel shape checks failed");
    }

    /**
     * Test for mixing row and column views in parallel
     */
    @Test
    public void testMixedRowColumnViewParallel() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Create vectors for broadcasting
        INDArray rowVector = Nd4j.create(new double[] {10, 20, 30});
        INDArray colVector = Nd4j.create(new double[] {10, 20, 30}).reshape(3, 1);
        
        int numThreads = 20;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            final int threadIdx = i;
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    if (threadIdx % 2 == 0) {
                        // Even threads work with row views
                        INDArray rowView = matrix.getRow(1);
                        INDArray result = rowView.add(rowVector);
                        INDArray expected = Nd4j.create(new double[] {14, 25, 36});
                        
                        if (!result.equalsWithEps(expected, 1e-5)) {
                            String msg = "Row operation result mismatch for thread " + threadIdx + 
                                       ". Expected: " + expected + ", Got: " + result;
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    } else {
                        // Odd threads work with column views
                        INDArray colView = matrix.getColumn(1);
                        INDArray result = colView.add(colVector);
                        INDArray expected = Nd4j.create(new double[] {12, 25, 38}).reshape(3, 1);
                        
                        if (!result.equalsWithEps(expected, 1e-5)) {
                            String msg = "Column operation result mismatch for thread " + threadIdx + 
                                       ". Expected: " + expected + ", Got: " + result;
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception in thread " + threadIdx + ": " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "Mixed row/column view parallel operations failed");
    }

    /**
     * Test for specifically reproducing the 'off by 1' issue in row views
     * This test repeatedly stresses the system by creating and using row views with
     * specific focus on checking shape information
     */
    @RepeatedTest(5) // Run multiple times to increase chance of reproducing the issue
    public void testRowViewOffByOneIssue() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        // Create vector for broadcasting
        INDArray rowVector = Nd4j.create(new double[] {10, 20, 30});
        
        int numThreads = 30; // More threads to increase contention
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            final int threadIdx = i;
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    // Get row view 
                    INDArray rowView = matrix.getRow(1);
                    
                    // Immediately check shape info (this is the critical test for the off-by-one issue)
                    if (rowView.length() != 3) {
                        String msg = "CRITICAL: Row view has incorrect length: " + rowView.length() + 
                                     " (expected 3) in thread " + threadIdx;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                        return; // Exit early on shape mismatch
                    }
                    
                    if (rowView.shape()[0] != 3) {
                        String msg = "CRITICAL: Row view has incorrect shape[0]: " + rowView.shape()[0] + 
                                     " (expected 3) in thread " + threadIdx;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                        return; // Exit early on shape mismatch
                    }
                    
                    // For this test we'll just create a temp copy and check shape is consistent
                    // to avoid the accumulation problem from all threads modifying the same matrix
                    INDArray rowViewCopy = rowView.dup();
                    
                    // Check the copy also has the right shape
                    if (rowViewCopy.length() != 3 || !Shape.shapeEquals(rowViewCopy.shape(), new long[] {3})) {
                        String msg = "Copied row view has incorrect shape in thread " + threadIdx;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                    // Do some non-mutating operations to stress the system
                    INDArray result = rowViewCopy.add(rowVector);
                    if (result.length() != 3) {
                        String msg = "Result has incorrect length: " + result.length() + 
                                     " (expected 3) in thread " + threadIdx;
                        firstError.compareAndSet(null, msg);
                        failures.incrementAndGet();
                    }
                    
                } catch (Exception e) {
                    firstError.compareAndSet(null, "General exception in thread " + threadIdx + ": " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(15, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "Row view off-by-one issue test failed");
    }

    /**
     * Test for multiple different views (rows, columns, and subarrays) accessed in parallel
     */
    @Test
    public void testMultipleViewTypesParallel() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}
        });

        int numThreads = 30;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            final int threadIdx = i;
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    switch (threadIdx % 5) {
                        case 0: // Row view check
                            INDArray rowView = matrix.getRow(2);
                            if (rowView.length() != 4 || !Shape.shapeEquals(rowView.shape(), new long[] {4})) {
                                String msg = "Row view has incorrect shape: " + rowView.shapeInfoToString();
                                firstError.compareAndSet(null, msg);
                                failures.incrementAndGet();
                            }
                            break;
                            
                        case 1: // Column view check
                            INDArray colView = matrix.getColumn(2);
                            if (colView.length() != 4 || !Shape.shapeEquals(colView.shape(), new long[] {4, 1})) {
                                String msg = "Column view has incorrect shape: " + colView.shapeInfoToString();
                                firstError.compareAndSet(null, msg);
                                failures.incrementAndGet();
                            }
                            break;
                            
                        case 2: // Subarray view check
                            INDArray subView = matrix.get(
                                    NDArrayIndex.interval(1, 3),
                                    NDArrayIndex.interval(1, 3)
                            );
                            if (!Shape.shapeEquals(subView.shape(), new long[] {2, 2})) {
                                String msg = "Subarray view has incorrect shape: " + subView.shapeInfoToString();
                                firstError.compareAndSet(null, msg);
                                failures.incrementAndGet();
                            }
                            break;
                            
                        case 3: // Get scalar
                            double scalar = matrix.getDouble(2, 2);
                            if (scalar != 11.0) {
                                String msg = "Scalar value incorrect: " + scalar + " (expected 11.0)";
                                firstError.compareAndSet(null, msg);
                                failures.incrementAndGet();
                            }
                            break;
                            
                        case 4: // Broadcast row operation
                            INDArray rowView2 = matrix.getRow(1);
                            INDArray rowVector = Nd4j.create(new double[] {10, 20, 30, 40});
                            INDArray result = rowView2.add(rowVector);
                            INDArray expected = Nd4j.create(new double[] {15, 26, 37, 48});
                            if (!result.equalsWithEps(expected, 1e-5)) {
                                String msg = "Row broadcast result mismatch. Expected: " + expected + ", Got: " + result;
                                firstError.compareAndSet(null, msg);
                                failures.incrementAndGet();
                            }
                            break;
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception in thread " + threadIdx + ": " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "Multiple view types parallel test failed");
    }

    /**
     * Test for shape consistency of views after in-place modifications
     */
    @Test
    public void testViewShapeAfterModifications() throws Exception {
        // Create a matrix
        INDArray matrix = Nd4j.create(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });

        int numThreads = 20;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch finishLatch = new CountDownLatch(numThreads);
        AtomicInteger failures = new AtomicInteger(0);
        AtomicReference<String> firstError = new AtomicReference<>(null);
        
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            final int threadIdx = i;
            executorService.submit(() -> {
                try {
                    startLatch.await(); // Wait for all threads to be ready
                    
                    // First thread modifies the matrix
                    if (threadIdx == 0) {
                        INDArray rowView = matrix.getRow(1);
                        rowView.addi(Nd4j.create(new double[] {10, 20, 30}));
                        
                        // Verify row view still has correct shape after modification
                        if (rowView.length() != 3 || !Shape.shapeEquals(rowView.shape(), new long[] {3})) {
                            String msg = "Row view has incorrect shape after modification: " + rowView.shapeInfoToString();
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    } else {
                        // Other threads check shape integrity of views
                        INDArray rowView = matrix.getRow(threadIdx % 3);
                        if (rowView.length() != 3 || !Shape.shapeEquals(rowView.shape(), new long[] {3})) {
                            String msg = "Row view has incorrect shape in thread " + threadIdx + ": " + rowView.shapeInfoToString();
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                        
                        INDArray colView = matrix.getColumn(threadIdx % 3);
                        if (colView.length() != 3 || !Shape.shapeEquals(colView.shape(), new long[] {3, 1})) {
                            String msg = "Column view has incorrect shape in thread " + threadIdx + ": " + colView.shapeInfoToString();
                            firstError.compareAndSet(null, msg);
                            failures.incrementAndGet();
                        }
                    }
                } catch (Exception e) {
                    firstError.compareAndSet(null, "Exception in thread " + threadIdx + ": " + e.getMessage());
                    failures.incrementAndGet();
                } finally {
                    finishLatch.countDown();
                }
            });
        }
        
        // Start all threads simultaneously
        startLatch.countDown();
        
        // Wait for all threads to complete
        finishLatch.await(10, TimeUnit.SECONDS);
        executorService.shutdown();
        
        assertEquals(0, failures.get(), firstError.get() != null ? firstError.get() : "View shape after modifications test failed");
    }
}
