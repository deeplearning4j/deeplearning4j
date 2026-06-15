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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Performance test for attention operations with realistic dimensions.
 * Use with nsys profiling: mvn test -Dtest=TestAttentionPerformance -Dtest.prefix=nsys
 */
@Slf4j
public class TestAttentionPerformance {

    @Test
    public void testAttentionPerformanceBGELike() {
        // BGE-base-en-v1.5 like dimensions
        // 12 attention layers, each with:
        // batch=1, seq=512, dim=768, heads=12, head_dim=64

        int batch = 1;
        int seqLen = 512;
        int dim = 768;
        int numLayers = 12;
        int warmupIterations = 5;
        int benchmarkIterations = 48;  // Match the 48 calls from real workload

        log.info("Creating attention test with BGE-like dimensions:");
        log.info("  batch={}, seqLen={}, dim={}, layers={}", batch, seqLen, dim, numLayers);

        // Create input arrays
        INDArray query = Nd4j.rand(DataType.FLOAT, batch, seqLen, dim);
        INDArray key = Nd4j.rand(DataType.FLOAT, batch, seqLen, dim);
        INDArray value = Nd4j.rand(DataType.FLOAT, batch, seqLen, dim);

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("query", query);
        SDVariable sdK = sd.var("key", key);
        SDVariable sdV = sd.var("value", value);

        // Use dotProductAttentionV2 with causal mask (like transformer decoder)
        SDVariable output = sd.nn.dotProductAttentionV2(
            sdQ, sdV, sdK,  // query, values, keys
            null, null,     // no masks
            1.0 / Math.sqrt(dim),  // scale
            0.0,            // dropout
            true,           // use causal mask
            false           // not training
        );

        // Warmup
        log.info("Warming up with {} iterations...", warmupIterations);
        for (int i = 0; i < warmupIterations; i++) {
            sd.output(java.util.Collections.emptyMap(), output.name());
        }

        // Synchronize GPU before timing
        Nd4j.getExecutioner().commit();

        // Benchmark
        log.info("Running {} benchmark iterations...", benchmarkIterations);
        long startTime = System.nanoTime();

        for (int i = 0; i < benchmarkIterations; i++) {
            sd.output(java.util.Collections.emptyMap(), output.name());
        }

        // Synchronize GPU after timing
        Nd4j.getExecutioner().commit();

        long endTime = System.nanoTime();
        double totalMs = (endTime - startTime) / 1_000_000.0;
        double avgMs = totalMs / benchmarkIterations;

        log.info("=== ATTENTION PERFORMANCE RESULTS ===");
        log.info("Total time: {:.2f} ms", totalMs);
        log.info("Average per call: {:.2f} ms", avgMs);
        log.info("Throughput: {:.2f} calls/sec", 1000.0 / avgMs);
    }

    @Test
    public void testAttentionPerformanceMultipleIterations() {
        // Simpler test that just runs attention many times for profiling
        int batch = 1;
        int seqLen = 512;
        int dim = 768;
        int iterations = 20;  // Reduced for profiling

        log.info("Running {} attention iterations for profiling", iterations);

        INDArray query = Nd4j.rand(DataType.FLOAT, batch, seqLen, dim);
        INDArray key = Nd4j.rand(DataType.FLOAT, batch, seqLen, dim);
        INDArray value = Nd4j.rand(DataType.FLOAT, batch, seqLen, dim);

        SameDiff sd = SameDiff.create();
        SDVariable sdQ = sd.var("query", query);
        SDVariable sdK = sd.var("key", key);
        SDVariable sdV = sd.var("value", value);

        SDVariable output = sd.nn.dotProductAttentionV2(
            sdQ, sdV, sdK,
            null, null,
            1.0 / Math.sqrt(dim),
            0.0,
            true,  // causal
            false
        );

        // Run iterations
        for (int i = 0; i < iterations; i++) {
            sd.output(java.util.Collections.emptyMap(), output.name());
            if (i % 10 == 0) {
                log.info("Iteration {}/{}", i, iterations);
            }
        }

        Nd4j.getExecutioner().commit();
        log.info("Done");
    }
}
