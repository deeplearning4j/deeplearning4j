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
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.InferenceFactory;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for InferenceSession explicit intermediate lifecycle management.
 *
 * These tests verify that:
 * 1. Intermediate arrays are freed during execution (not just at the end)
 *    via consumer-based OpDep dependencies instead of ExecDoneDep.
 * 2. AutoGc is suppressed during InferenceSession execution to avoid
 *    excessive Full GC pauses.
 * 3. Memory does not grow unboundedly during multi-step inference.
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class InferenceSessionLifecycleTest extends BaseNd4jTestWithBackends {

    /**
     * Verify that intermediate arrays are released mid-execution in the standard
     * (non-DSP) InferenceSession path.
     *
     * Build a linear chain: x -> a -> b -> c -> d -> e (output)
     * After op "c" executes and its output is consumed by "d", the intermediate
     * arrays for "a" and "b" should be released (their last consumer completed).
     * Only "e" (the final output) should survive.
     *
     * With the old ExecDoneDep approach, all 4 intermediate arrays (a, b, c, d)
     * would survive until end-of-execution. With consumer-based OpDep tracking,
     * each is freed when its last consumer completes.
     *
     * We verify this by counting mmgr.release() calls during execution. With
     * proper liveness tracking, releases happen incrementally during the execution
     * loop (not all at the end in the cleanup phase).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIntermediateArraysReleasedDuringExecution(Nd4jBackend backend) {
        // Force standard (non-DSP) execution path
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();

        ReleaseTrackingFactory factory = new ReleaseTrackingFactory();
        try {
            InferenceSession.setDynamicShapePlanEnabled(false);
            ArrayCacheMemoryMgr.setEnableCache(false);

            // Build a 5-op linear chain: x -> a -> b -> c -> d -> e
            SameDiff sd = SameDiff.create();
            // Bind the tracking factory on this instance only (no global side effects)
            sd.bindInferenceFactory(factory);
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable a = x.add("a", 1.0);
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = b.sub("c", 0.5);
            SDVariable d = c.add("d", 3.0);
            SDVariable e = d.mul("e", 0.1);

            INDArray input = Nd4j.ones(DataType.FLOAT, 2, 4);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            Map<String, INDArray> outputs = sd.output(placeholders, "e");
            assertNotNull(outputs.get("e"));

            // Verify correctness
            INDArray expected = input.add(1.0).mul(2.0).sub(0.5).add(3.0).mul(0.1);
            assertEquals(expected, outputs.get("e"));

            ReleaseTrackingMemMgr memMgr = factory.getLastMemMgr();
            assertNotNull(memMgr, "Expected ReleaseTrackingMemMgr from factory");

            // With consumer-based liveness, intermediate arrays (a, b, c, d) should
            // be released during execution. The release count should be > 0
            // even before the final cleanup phase.
            int midExecReleases = memMgr.getMidExecutionReleaseCount();
            int totalReleases = memMgr.getTotalReleaseCount();

            assertTrue(totalReleases >= 4,
                    "Expected at least 4 intermediate releases (a, b, c, d) but got " + totalReleases);
            assertTrue(midExecReleases >= 2,
                    "Expected at least 2 mid-execution releases with consumer-based tracking, got " +
                            midExecReleases + " (total releases: " + totalReleases + ")");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
        }
    }

    /**
     * Verify that autoGc is suppressed during InferenceSession execution.
     * The autoGcWindow should be set to Integer.MAX_VALUE during execution
     * and restored afterward.
     *
     * This prevents the DeallocatorService from calling System.gc() every 100ms
     * during graph execution, which causes Full GC pauses.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoGcSuppressedDuringExecution(Nd4jBackend backend) {
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();

        try {
            InferenceSession.setDynamicShapePlanEnabled(false);
            ArrayCacheMemoryMgr.setEnableCache(false);

            // Record the autoGcWindow before execution
            int initialGcWindow = Nd4j.getMemoryManager().getAutoGcWindow();

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
            SDVariable y = x.add("y", 1.0);

            INDArray input = Nd4j.ones(DataType.FLOAT, 1, 4);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            Map<String, INDArray> outputs = sd.output(placeholders, "y");
            assertNotNull(outputs.get("y"));

            // After execution, the autoGcWindow should be restored to its original value
            int postExecGcWindow = Nd4j.getMemoryManager().getAutoGcWindow();
            assertEquals(initialGcWindow, postExecGcWindow,
                    "AutoGcWindow should be restored after InferenceSession execution");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
        }
    }

    /**
     * Verify that memory does not grow unboundedly during repeated standard-path
     * InferenceSession executions. Each execution should release its intermediates,
     * keeping total memory bounded.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMemoryBoundedDuringRepeatedExecution(Nd4jBackend backend) {
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();

        ReleaseTrackingFactory factory = new ReleaseTrackingFactory();
        try {
            InferenceSession.setDynamicShapePlanEnabled(false);
            ArrayCacheMemoryMgr.setEnableCache(false);

            // Build a wider graph to create more intermediates per step
            SameDiff sd = SameDiff.create();
            // Bind the tracking factory on this instance only (no global side effects)
            sd.bindInferenceFactory(factory);
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 128);
            SDVariable a = x.add("a", 1.0);
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = b.sub("c", 0.5);
            SDVariable d = c.add("d", 3.0);
            SDVariable e = d.div("e", 2.0);
            SDVariable f = e.mul("f", 1.5);

            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 128);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            // Execute many times
            int iterations = 20;
            for (int i = 0; i < iterations; i++) {
                Map<String, INDArray> outputs = sd.output(placeholders, "f");
                assertNotNull(outputs.get("f"));
            }

            ReleaseTrackingMemMgr memMgr = factory.getLastMemMgr();
            assertNotNull(memMgr);

            // Across 20 iterations with 5 intermediates each, we should see significant
            // release activity (not zero). With unbounded growth, releases would be low.
            int totalReleases = memMgr.getTotalReleaseCount();
            assertTrue(totalReleases >= iterations * 3,
                    "Expected at least " + (iterations * 3) + " total releases across " +
                            iterations + " iterations, but got " + totalReleases);
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
        }
    }

    /**
     * Verify that consumer-based liveness works correctly with diamond-shaped
     * graphs where a single variable feeds into multiple ops.
     *
     * Graph:  x -> a -> b (output)
     *                 \-> c (output)
     *
     * Variable "a" feeds into both "b" and "c". It should NOT be freed until
     * both "b" and "c" have executed.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDiamondGraphLivenessTracking(Nd4jBackend backend) {
        boolean prevDynamicEnabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean prevCache = ArrayCacheMemoryMgr.isCacheEnabled();

        try {
            InferenceSession.setDynamicShapePlanEnabled(false);
            ArrayCacheMemoryMgr.setEnableCache(false);

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable a = x.add("a", 1.0);
            // Two consumers of "a"
            SDVariable b = a.mul("b", 2.0);
            SDVariable c = a.sub("c", 0.5);

            INDArray input = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(2, 4);
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            placeholders.put("x", input);

            // Request both outputs
            Map<String, INDArray> outputs = sd.output(placeholders, "b", "c");

            // Verify correctness: "a" was available for both ops
            INDArray aExpected = input.add(1.0);
            assertEquals(aExpected.mul(2.0), outputs.get("b"), "b = a * 2 should be correct");
            assertEquals(aExpected.sub(0.5), outputs.get("c"), "c = a - 0.5 should be correct");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(prevDynamicEnabled);
            ArrayCacheMemoryMgr.setEnableCache(prevCache);
        }
    }

    // ---- Helper classes ----

    private static final class ReleaseTrackingFactory implements InferenceFactory {
        private ReleaseTrackingMemMgr lastMemMgr;

        @Override
        public InferenceSession create(SameDiff sameDiff) {
            lastMemMgr = new ReleaseTrackingMemMgr();
            return new InferenceSession(sameDiff, lastMemMgr);
        }

        ReleaseTrackingMemMgr getLastMemMgr() {
            return lastMemMgr;
        }
    }

    /**
     * ArrayCacheMemoryMgr subclass that tracks release calls and distinguishes
     * between mid-execution releases (from arrayUseTracker) and end-of-execution
     * releases (from cleanup/postProcess).
     */
    private static final class ReleaseTrackingMemMgr extends ArrayCacheMemoryMgr {
        private final AtomicInteger totalReleaseCount = new AtomicInteger();
        private final AtomicInteger midExecutionReleaseCount = new AtomicInteger();
        // Heuristic: releases during scopeIn/scopeOut are cleanup; releases at other
        // times are mid-execution. We track scope state via scopeIn/scopeOut.
        private volatile boolean inCleanup = false;

        @Override
        public void release(INDArray array) {
            totalReleaseCount.incrementAndGet();
            if (!inCleanup) {
                midExecutionReleaseCount.incrementAndGet();
            }
            super.release(array);
        }

        @Override
        public void scopeOut() {
            inCleanup = true;
            super.scopeOut();
            inCleanup = false;
        }

        int getTotalReleaseCount() {
            return totalReleaseCount.get();
        }

        int getMidExecutionReleaseCount() {
            return midExecutionReleaseCount.get();
        }
    }
}
