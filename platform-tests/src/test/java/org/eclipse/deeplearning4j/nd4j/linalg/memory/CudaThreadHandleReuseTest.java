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

package org.eclipse.deeplearning4j.nd4j.linalg.memory;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Threads that use cuBLAS and then exit must hand their handle to later threads. Each handle owns a
 * device workspace of tens of MB, and with thread-pool churn a handle left behind per exited thread
 * grows device memory without bound.
 */
@NativeTag
@Tag(TagNames.MULTI_THREADED)
public class CudaThreadHandleReuseTest extends BaseNd4jTestWithBackends {

    // Java's Thread.join() returns before the native thread runs its thread_local destructors,
    // which is where the handle goes back to the pool.
    private static final long NATIVE_THREAD_EXIT_MS = 300;

    private static boolean isCudaBackend() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        return backendName.contains("cuda") || backendName.contains("jcublas");
    }

    private static long cublasHandleAddress() {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        return ops.lcBlasHandle(ops.defaultLaunchContext()).address();
    }

    private static void runThreads(int count, Runnable body) throws Exception {
        Thread[] threads = new Thread[count];
        Throwable[] failures = new Throwable[count];
        for (int t = 0; t < count; t++) {
            int index = t;
            threads[t] = new Thread(() -> {
                try {
                    body.run();
                } catch (Throwable e) {
                    failures[index] = e;
                }
            });
            threads[t].start();
        }
        for (Thread thread : threads) {
            thread.join();
        }
        Thread.sleep(NATIVE_THREAD_EXIT_MS);
        for (Throwable failure : failures) {
            if (failure != null) {
                throw new AssertionError("Worker thread failed", failure);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExitedThreadHandleIsReused(Nd4jBackend backend) throws Exception {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        Set<Long> handles = new HashSet<>();
        for (int i = 0; i < 16; i++) {
            long[] handle = new long[1];
            runThreads(1, () -> handle[0] = cublasHandleAddress());
            assertNotEquals(0L, handle[0], "cuBLAS handle was not initialized");
            handles.add(handle[0]);
        }
        // One thread at a time, so every thread after the first should get the handle its
        // predecessor returned. One extra allows for a handle some other thread took meanwhile.
        assertTrue(handles.size() <= 2,
                "16 sequential threads used " + handles.size() + " distinct cuBLAS handles; exited threads' handles are not reused");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGemmThreadChurnKeepsDeviceMemoryFlat(Nd4jBackend backend) throws Exception {
        assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        INDArray a = Nd4j.rand(DataType.FLOAT, 64, 64);
        INDArray b = Nd4j.rand(DataType.FLOAT, 64, 64);
        INDArray expected = a.mmul(b);

        int threads = 8;
        Runnable gemm = () -> {
            try (INDArray product = a.mmul(b)) {
                assertTrue(expected.equalsWithEps(product, 1e-4), "GEMM on a reused cuBLAS handle gave a wrong result");
            }
        };

        // The first round creates the handles the later rounds should reuse.
        runThreads(threads, gemm);
        int rounds = 4;
        long freeBefore = ops.getDeviceFreeMemory(device);
        for (int round = 0; round < rounds; round++) {
            runThreads(threads, gemm);
        }
        long freeAfter = ops.getDeviceFreeMemory(device);

        // A handle left behind per thread costs about 64 MB, 2 GB over these rounds. Free memory is
        // system-wide on unified-memory devices, so allow for unrelated movement well below that.
        long dropPerThread = (freeBefore - freeAfter) / ((long) threads * rounds);
        assertTrue(dropPerThread < 16L * 1024 * 1024,
                "Device free memory dropped " + dropPerThread / 1024 + " KB per exited GEMM thread");
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
