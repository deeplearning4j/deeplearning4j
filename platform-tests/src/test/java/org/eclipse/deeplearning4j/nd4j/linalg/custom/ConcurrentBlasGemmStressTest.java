/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.custom;

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
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Concurrency regression guard for the BLAS GEMM serialization path
 * (BlasHelper::lockBlas / setSerializeBlasCalls).
 *
 * Context: NativeOpsHolder used to unconditionally force serialization ON at
 * every JVM init, stomping the native BLAS-aware default (MKL runs
 * unserialized, OpenBLAS keeps the safety mutex; SD_BLAS_SERIALIZE /
 * ND4J_BLAS_SERIALIZE override). Now the native default stands unless
 * ND4J_BLAS_SERIALIZE is explicitly set — so this test pins the correctness
 * of concurrent GEMMs under (a) whatever default ships, and (b) serialization
 * explicitly disabled, with results checked against serially-computed
 * references.
 */
@NativeTag
@Tag(TagNames.FULL_CI)
public class ConcurrentBlasGemmStressTest extends BaseNd4jTestWithBackends {

    private static final int THREADS = 8;
    private static final int ITERS_PER_THREAD = 32;
    private static final int M = 96, K = 64, N = 80;
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    /** Concurrent GEMMs under the shipped default serialization setting. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentGemmDefaultMode(Nd4jBackend backend) throws Exception {
        runConcurrentGemmStress();
    }

    /**
     * Concurrent GEMMs with serialization explicitly DISABLED — the risk case
     * the mutex exists for. Restores serialization to enabled afterwards so no
     * state leaks into other tests (Surefire reuses the JVM).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConcurrentGemmSerializationDisabled(Nd4jBackend backend) throws Exception {
        try {
            NativeOpsHolder.getInstance().getDeviceNativeOps().setSerializeBlasCalls(false);
            runConcurrentGemmStress();
        } finally {
            // Restore the conservative setting — do NOT leak disabled
            // serialization into subsequent tests sharing this JVM.
            NativeOpsHolder.getInstance().getDeviceNativeOps().setSerializeBlasCalls(true);
        }
    }

    private static void runConcurrentGemmStress() throws Exception {
        // Per-thread deterministic inputs + serial reference results, computed
        // BEFORE any concurrency so the reference is unambiguous.
        List<INDArray> as = new ArrayList<>();
        List<INDArray> bs = new ArrayList<>();
        List<INDArray> refs = new ArrayList<>();
        for (int t = 0; t < THREADS; t++) {
            Nd4j.getRandom().setSeed(7000 + t);
            INDArray a = Nd4j.rand(DataType.FLOAT, M, K).subi(0.5);
            INDArray b = Nd4j.rand(DataType.FLOAT, K, N).subi(0.5);
            as.add(a);
            bs.add(b);
            refs.add(a.mmul(b)); // serial reference
        }

        ExecutorService pool = Executors.newFixedThreadPool(THREADS);
        try {
            CountDownLatch start = new CountDownLatch(1);
            List<Future<Double>> results = new ArrayList<>();
            for (int t = 0; t < THREADS; t++) {
                final int tid = t;
                results.add(pool.submit(() -> {
                    start.await();
                    double worst = 0.0;
                    for (int i = 0; i < ITERS_PER_THREAD; i++) {
                        INDArray c = as.get(tid).mmul(bs.get(tid));
                        double diff = c.sub(refs.get(tid)).amaxNumber().doubleValue();
                        if (diff > worst) worst = diff;
                    }
                    return worst;
                }));
            }
            start.countDown(); // fire all threads at once for maximal overlap

            double worstOverall = 0.0;
            for (Future<Double> f : results) {
                // Generous timeout: a deadlock in the BLAS lock path shows up
                // here as a timeout instead of hanging the whole suite.
                worstOverall = Math.max(worstOverall, f.get(120, TimeUnit.SECONDS));
            }
            assertTrue(worstOverall < TOL,
                    "concurrent GEMM deviated from serial reference: worstAbsDiff=" + worstOverall);
        } finally {
            pool.shutdownNow();
        }
    }
}
