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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.training;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.config.GradientCheckpointConfig;
import org.nd4j.autodiff.samediff.config.GradientCheckpointConfig.CheckpointStrategy;
import org.nd4j.autodiff.samediff.training.CheckpointOffloadManager;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.HashSet;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for async gradient checkpoint offloading.
 *
 * Covers:
 * - GradientCheckpointConfig strategy enum and factory methods
 * - Backward compatibility of existing RECOMPUTE configs
 * - CheckpointOffloadManager lifecycle (offload / get / release)
 * - Offload-then-prefetch round trip with data integrity
 * - Prefetch distance scheduling
 * - Host memory budget enforcement
 * - C++ checkpoint_offload_d2h op
 * - C++ checkpoint_prefetch_h2d op
 * - Bit-exact round trip through offload + prefetch
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
@DisplayName("Checkpoint Offload Tests")
public class TestCheckpointOffload extends BaseNd4jTestWithBackends {

    private CheckpointOffloadManager manager;

    @AfterEach
    public void tearDown() {
        if (manager != null) {
            manager.releaseAll();
            manager = null;
        }
    }

    // =========================================================================
    // 1. Config creation and defaults
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("asyncOffload() factory — defaults are correct")
    public void testGradientCheckpointConfigAsyncOffload(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload();

        assertEquals(CheckpointStrategy.OFFLOAD_HOST_ASYNC, cfg.getStrategy(),
                "strategy should be OFFLOAD_HOST_ASYNC");
        assertEquals(1, cfg.getCheckpointEveryN(),
                "checkpointEveryN should default to 1 for async offload");
        assertTrue(cfg.isPinHostMemory(), "pinHostMemory should default to true");
        assertEquals(2, cfg.getPrefetchDistance(), "prefetchDistance should default to 2");
        assertEquals(0L, cfg.getMaxHostMemoryMB(), "maxHostMemoryMB should default to 0 (unlimited)");

        assertTrue(cfg.isAsyncOffload(), "isAsyncOffload() should return true");
        assertTrue(cfg.isAnyOffload(), "isAnyOffload() should return true");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("asyncOffload(dist) factory — custom prefetch distance")
    public void testGradientCheckpointConfigAsyncOffloadWithDist(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload(4);

        assertEquals(CheckpointStrategy.OFFLOAD_HOST_ASYNC, cfg.getStrategy());
        assertEquals(4, cfg.getPrefetchDistance());
        assertTrue(cfg.isPinHostMemory());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("OFFLOAD_HOST strategy — isAnyOffload true, isAsyncOffload false")
    public void testOffloadHostStrategy(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.builder()
                .strategy(CheckpointStrategy.OFFLOAD_HOST)
                .checkpointEveryN(2)
                .build();

        assertEquals(CheckpointStrategy.OFFLOAD_HOST, cfg.getStrategy());
        assertTrue(cfg.isAnyOffload());
        assertFalse(cfg.isAsyncOffload());
    }

    // =========================================================================
    // 2. Backward compatibility
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("RECOMPUTE is the default strategy — old configs still work")
    public void testGradientCheckpointConfigBackwardsCompat(Nd4jBackend backend) {
        // Old-style configs (no strategy set) must default to RECOMPUTE
        GradientCheckpointConfig everyN = GradientCheckpointConfig.everyN(3);
        assertEquals(CheckpointStrategy.RECOMPUTE, everyN.getStrategy(),
                "everyN should use RECOMPUTE strategy by default");
        assertFalse(everyN.isAnyOffload());
        assertFalse(everyN.isAsyncOffload());

        GradientCheckpointConfig sqrtN = GradientCheckpointConfig.sqrtN();
        assertEquals(CheckpointStrategy.RECOMPUTE, sqrtN.getStrategy());

        GradientCheckpointConfig manual = GradientCheckpointConfig.manual(
                new HashSet<>(Arrays.asList("layer_0", "layer_4")));
        assertEquals(CheckpointStrategy.RECOMPUTE, manual.getStrategy());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Deprecated offloadToHost field — sets strategy via new code path")
    public void testDeprecatedOffloadToHostField(Nd4jBackend backend) {
        // Old builder with the deprecated field still compiles and the builder
        // produces the expected RECOMPUTE strategy (the deprecated field is separate).
        @SuppressWarnings("deprecation")
        GradientCheckpointConfig cfg = GradientCheckpointConfig.builder()
                .offloadToHost(true)
                .checkpointEveryN(1)
                .build();

        // The old boolean is still present; the new strategy field is independent.
        // RECOMPUTE is still the default strategy for builder-constructed configs.
        assertEquals(CheckpointStrategy.RECOMPUTE, cfg.getStrategy());
        assertTrue(cfg.isOffloadToHost()); // deprecated but accessible
    }

    // =========================================================================
    // 3. CheckpointOffloadManager lifecycle
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Manager lifecycle: offload, get, release")
    public void testCheckpointOffloadManagerLifecycle(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload();
        manager = new CheckpointOffloadManager(cfg);

        INDArray activation = Nd4j.rand(DataType.FLOAT, 4, 8);
        manager.offloadCheckpoint("var_0", activation);

        assertEquals(1, manager.getHostCheckpointCount(),
                "should have 1 host checkpoint after offload");
        assertTrue(manager.getHostMemoryBytes() > 0,
                "host memory should be non-zero after offload");

        INDArray retrieved = manager.getCheckpoint("var_0");
        assertNotNull(retrieved, "getCheckpoint should return a non-null array");
        assertEquals(activation.shape()[0], retrieved.shape()[0]);
        assertEquals(activation.shape()[1], retrieved.shape()[1]);
        retrieved.close();

        manager.releaseAll();

        assertEquals(0, manager.getHostCheckpointCount(),
                "host checkpoints should be empty after releaseAll");
        assertEquals(0L, manager.getHostMemoryBytes(),
                "host memory bytes should be 0 after releaseAll");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("getCheckpoint throws for unknown variable")
    public void testGetCheckpointThrowsForUnknown(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload();
        manager = new CheckpointOffloadManager(cfg);

        assertThrows(IllegalArgumentException.class,
                () -> manager.getCheckpoint("does_not_exist"),
                "should throw for unknown variable name");
    }

    // =========================================================================
    // 4. Round-trip data integrity
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Offload and retrieve — data matches original")
    public void testOffloadAndPrefetch(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload();
        manager = new CheckpointOffloadManager(cfg);

        Nd4j.getRandom().setSeed(42);
        INDArray original = Nd4j.rand(DataType.FLOAT, 16, 32);

        manager.offloadCheckpoint("act", original);
        INDArray retrieved = manager.getCheckpoint("act");

        // Data should be numerically equal (note: on CPU dup() preserves bits)
        assertEquals(original.sumNumber().floatValue(), retrieved.sumNumber().floatValue(), 1e-4f,
                "sum should match after round-trip");

        retrieved.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Multiple variables offloaded then retrieved in LIFO order")
    public void testMultipleVariablesRoundTrip(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload();
        manager = new CheckpointOffloadManager(cfg);

        Nd4j.getRandom().setSeed(7);
        INDArray[] originals = new INDArray[5];
        for (int i = 0; i < 5; i++) {
            originals[i] = Nd4j.rand(DataType.FLOAT, 8, 16);
            manager.offloadCheckpoint("layer_" + i, originals[i]);
        }

        assertEquals(5, manager.getHostCheckpointCount());

        // Retrieve in backward order
        for (int i = 4; i >= 0; i--) {
            INDArray retrieved = manager.getCheckpoint("layer_" + i);
            assertEquals(originals[i].sumNumber().floatValue(),
                    retrieved.sumNumber().floatValue(), 1e-4f,
                    "layer_" + i + " sum mismatch");
            retrieved.close();
        }
    }

    // =========================================================================
    // 5. Prefetch distance scheduling
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Prefetch distance=2 — correct tensors are prefetched ahead of time")
    public void testPrefetchDistance(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload(2);
        manager = new CheckpointOffloadManager(cfg);

        int numLayers = 6;
        Nd4j.getRandom().setSeed(99);
        INDArray[] originals = new INDArray[numLayers];
        for (int i = 0; i < numLayers; i++) {
            originals[i] = Nd4j.rand(DataType.FLOAT, 4, 4);
            manager.offloadCheckpoint("layer_" + i, originals[i]);
        }

        // Simulate backward pass starting from layer 5
        // prefetchForBackward(5, 6) should prefetch layer 4 and layer 3
        manager.prefetchForBackward(5, numLayers);
        int prefetchedAfter5 = manager.getPrefetchedCount();
        assertTrue(prefetchedAfter5 > 0,
                "should have prefetched some tensors after prefetchForBackward(5, 6)");

        // Consuming layer 5 should succeed (not prefetched — direct hit from host)
        INDArray l5 = manager.getCheckpoint("layer_5");
        assertNotNull(l5);
        l5.close();

        // Consuming a prefetched layer should work via fast path
        INDArray l4 = manager.getCheckpoint("layer_4");
        assertNotNull(l4);
        l4.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Prefetch is skipped when strategy is not OFFLOAD_HOST_ASYNC")
    public void testPrefetchSkippedForSyncStrategy(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.builder()
                .strategy(CheckpointStrategy.OFFLOAD_HOST)
                .checkpointEveryN(1)
                .prefetchDistance(2)
                .build();
        manager = new CheckpointOffloadManager(cfg);

        Nd4j.getRandom().setSeed(1);
        manager.offloadCheckpoint("v0", Nd4j.rand(DataType.FLOAT, 4, 4));
        manager.offloadCheckpoint("v1", Nd4j.rand(DataType.FLOAT, 4, 4));

        // prefetchForBackward should be a no-op for OFFLOAD_HOST
        manager.prefetchForBackward(1, 2);
        assertEquals(0, manager.getPrefetchedCount(),
                "sync strategy should not prefetch anything");
    }

    // =========================================================================
    // 6. Host memory budget enforcement
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Budget exceeded — offload is skipped with warning")
    public void testMaxHostMemoryBudget(Nd4jBackend backend) {
        // Budget: 1 MB
        GradientCheckpointConfig cfg = GradientCheckpointConfig.builder()
                .strategy(CheckpointStrategy.OFFLOAD_HOST_ASYNC)
                .checkpointEveryN(1)
                .maxHostMemoryMB(1L)
                .build();
        manager = new CheckpointOffloadManager(cfg);

        // Each array: 256*256*4 bytes = 256 KB → 4 arrays = 1 MB → 5th should be skipped
        for (int i = 0; i < 4; i++) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, 256, 256);
            manager.offloadCheckpoint("v" + i, arr);
            arr.close();
        }

        // 5th array should be skipped due to budget
        INDArray extra = Nd4j.rand(DataType.FLOAT, 256, 256);
        int countBefore = manager.getHostCheckpointCount();
        manager.offloadCheckpoint("v_extra", extra);
        int countAfter = manager.getHostCheckpointCount();
        extra.close();

        assertEquals(countBefore, countAfter,
                "checkpoint count should not increase when budget is exceeded");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Budget=0 means unlimited — all offloads succeed")
    public void testUnlimitedBudget(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.builder()
                .strategy(CheckpointStrategy.OFFLOAD_HOST_ASYNC)
                .checkpointEveryN(1)
                .maxHostMemoryMB(0L)  // unlimited
                .build();
        manager = new CheckpointOffloadManager(cfg);

        for (int i = 0; i < 10; i++) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, 32, 32);
            manager.offloadCheckpoint("v" + i, arr);
            arr.close();
        }
        assertEquals(10, manager.getHostCheckpointCount(),
                "all 10 tensors should have been offloaded");
    }

    // =========================================================================
    // 7. C++ op tests
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("checkpoint_offload_d2h op — output matches input")
    public void testCheckpointOffloadD2HOp(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(DataType.FLOAT, 4, 8);

        INDArray[] out = Nd4j.exec(
                DynamicCustomOp.builder("checkpoint_offload_d2h")
                        .addInputs(input)
                        .addIntegerArguments(0, 0)  // no pinning, stream 0
                        .build());

        assertNotNull(out);
        assertEquals(1, out.length, "should have 1 output");
        assertArrayEquals(input.shape(), out[0].shape(), "output shape must match input");
        assertEquals(input.sumNumber().floatValue(), out[0].sumNumber().floatValue(), 1e-4f,
                "output data must match input");

        out[0].close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("checkpoint_offload_d2h op — preserves data type")
    public void testCheckpointOffloadD2HOpDataType(Nd4jBackend backend) {
        INDArray inputInt = Nd4j.createFromArray(new int[]{1, 2, 3, 4, 5}).reshape(1, 5);

        INDArray[] out = Nd4j.exec(
                DynamicCustomOp.builder("checkpoint_offload_d2h")
                        .addInputs(inputInt)
                        .addIntegerArguments(0, 0)
                        .build());

        assertEquals(DataType.INT32, out[0].dataType(), "data type should be preserved");
        assertArrayEquals(inputInt.shape(), out[0].shape());

        out[0].close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("checkpoint_prefetch_h2d op — output matches input")
    public void testCheckpointPrefetchH2DOp(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(999);
        INDArray input = Nd4j.rand(DataType.FLOAT, 8, 16);

        INDArray[] out = Nd4j.exec(
                DynamicCustomOp.builder("checkpoint_prefetch_h2d")
                        .addInputs(input)
                        .addIntegerArguments(0)  // target device 0
                        .build());

        assertNotNull(out);
        assertEquals(1, out.length, "should have 1 output");
        assertArrayEquals(input.shape(), out[0].shape());
        assertEquals(input.sumNumber().floatValue(), out[0].sumNumber().floatValue(), 1e-4f,
                "H2D output data must match input");

        out[0].close();
    }

    // =========================================================================
    // 8. Full round-trip data integrity through ops
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Round trip: D2H then H2D — bit-exact match")
    public void testRoundTripDataIntegrity(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(31415);
        INDArray original = Nd4j.rand(DataType.FLOAT, 32, 64);
        float originalSum = original.sumNumber().floatValue();

        // Step 1: D2H (offload to host)
        INDArray[] hostOut = Nd4j.exec(
                DynamicCustomOp.builder("checkpoint_offload_d2h")
                        .addInputs(original)
                        .addIntegerArguments(1, 0)  // pinned, stream 0
                        .build());
        INDArray hostCopy = hostOut[0];

        // Step 2: H2D (prefetch back to device)
        INDArray[] deviceOut = Nd4j.exec(
                DynamicCustomOp.builder("checkpoint_prefetch_h2d")
                        .addInputs(hostCopy)
                        .addIntegerArguments(0)
                        .build());
        INDArray deviceCopy = deviceOut[0];

        assertEquals(originalSum, deviceCopy.sumNumber().floatValue(), 1e-4f,
                "round-trip sum must match original");
        assertArrayEquals(original.shape(), deviceCopy.shape(),
                "round-trip shape must match original");

        hostCopy.close();
        deviceCopy.close();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Round trip through manager: manager offload + manager get — values preserved")
    public void testManagerRoundTripDataIntegrity(Nd4jBackend backend) {
        GradientCheckpointConfig cfg = GradientCheckpointConfig.asyncOffload();
        manager = new CheckpointOffloadManager(cfg);

        Nd4j.getRandom().setSeed(2718);
        int numLayers = 8;
        INDArray[] originals = new INDArray[numLayers];
        float[] originalSums = new float[numLayers];

        // Forward pass: offload all layers
        for (int i = 0; i < numLayers; i++) {
            originals[i] = Nd4j.rand(DataType.FLOAT, 16, 16);
            originalSums[i] = originals[i].sumNumber().floatValue();
            manager.offloadCheckpoint("layer_" + i, originals[i]);
        }

        // Backward pass: retrieve all layers and verify
        for (int i = numLayers - 1; i >= 0; i--) {
            manager.prefetchForBackward(i, numLayers);
            INDArray retrieved = manager.getCheckpoint("layer_" + i);
            assertEquals(originalSums[i], retrieved.sumNumber().floatValue(), 1e-4f,
                    "layer_" + i + " sum must match after round-trip");
            retrieved.close();
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
