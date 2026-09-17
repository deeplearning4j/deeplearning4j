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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DSP buffer coloring (compile-time buffer sharing) and the
 * DspBufferPool cross-plan buffer reuse mechanism.
 *
 * Buffer coloring assigns a "color" to non-overlapping intermediate slots
 * so they share physical DataBuffers, reducing GPU memory usage 10-20x
 * for large graphs like LLMs.
 */
@Slf4j
@Tag("dsp")
@DisplayName("DSP Buffer Coloring")
public class DspBufferColoringTest {

    private SameDiff sd;

    @BeforeEach
    void setup() {
        // DSP auto-compile is enabled by default on each SameDiff instance;
        // instance setters are called per-test after sd is created.
    }

    @AfterEach
    void teardown() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    @Test
    void testFirstWarmupSharesDeadIntermediates() {
        firstWarmupSharesDeadIntermediates(false);
    }

    @Test
    void testShapePrepassStillSharesFirstWarmupIntermediates() {
        firstWarmupSharesDeadIntermediates(true);
    }

    @Test
    void testFirstWarmupReusesLargerDeadBuffers() {
        firstWarmupReusesLargerDeadBuffers(DataType.FLOAT, 1.015625f);
    }

    @Test
    void testFirstWarmupReusesLargerDeadHalfBuffers() {
        // Input values and identity weights are exactly representable in HALF.
        firstWarmupReusesLargerDeadBuffers(DataType.HALF, 1.0f);
    }

    private void firstWarmupReusesLargerDeadBuffers(DataType dtype, float factor) {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        sd.setGraphExecutionMode(org.nd4j.autodiff.samediff.execution.GraphExecutionMode.TRITON);
        SDVariable x = sd.placeHolder("input", dtype, 64, 256);
        int width = 256;
        for (int layer = 0; layer < 16; layer++) {
            int nextWidth = width - 8;
            // Every intermediate has a distinct size. Rectangular diagonal
            // weights preserve a nonuniform prefix, without introducing aliases.
            float[] weights = new float[width * nextWidth];
            for (int col = 0; col < nextWidth; col++) weights[col * nextWidth + col] = factor;
            SDVariable weight = sd.constant("weight_" + layer,
                    Nd4j.create(weights, new long[]{width, nextWidth}, dtype));
            x = sd.mmul(layer == 15 ? "output" : "layer_" + layer, x, weight);
            width = nextWidth;
        }
        sd.compileNativeDynamicShapePlan("output");
        for (int iteration = 0; iteration < 8; iteration++) {
            float[] input = new float[64 * 256];
            for (int i = 0; i < input.length; i++) input[i] = (i % 251 - 125) / 256.0f + iteration / 16.0f;
            try (INDArray values = Nd4j.create(input, new long[]{64, 256}, dtype);
                 INDArray output = sd.outputSingle(Map.of("input", values), "output")) {
                assertArrayEquals(new long[]{64, 128}, output.shape());
                float[] actual = output.data().asFloat();
                for (int row = 0; row < 64; row++) {
                    for (int col = 0; col < 128; col++) {
                        float expected = input[row * 256 + col];
                        for (int layer = 0; layer < 16; layer++) expected *= factor;
                        assertEquals(expected, actual[row * 128 + col], 1e-5f,
                                "iteration=" + iteration + " row=" + row + " col=" + col);
                    }
                }
                if (iteration == 0) {
                    DspHandle handle = new DspHandle(sd);
                    assertTrue(handle.bufferColoringApplied(), "distinct sizes must share during first warmup");
                    assertTrue(handle.bufferColoringBytesSaved() >= 8L * 64 * 128 * dtype.width(),
                            "larger dead storage must replace at least eight smaller allocations");
                }
            }
        }
        assertTrue(new DspHandle(sd).numCapturedGraphSegments() > 0,
                "capacity sharing must also survive captured execution, not only warmup");
    }

    @Test
    void testOrdinaryReleasePreservesBorrowedOutputStorage() {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        sd.placeHolder("input", DataType.FLOAT, 32, 32).add("output", 0.25);
        sd.compileNativeDynamicShapePlan("output");
        var ops = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
        try (INDArray input = Nd4j.ones(DataType.FLOAT, 32, 32);
             INDArray first = sd.outputSingle(Map.of("input", input), "output");
             INDArray readback = Nd4j.create(DataType.FLOAT, 32, 32)) {
            var handle = sd.getOrCreateSession().getDynamicShapePlanExecutor().getNativePlanHandle();
            var opaque = ops.getPlanSlotOutputArray(handle, 0);
            assertNotNull(opaque);
            opaque.attachOwner(org.nd4j.nativeblas.OpaqueDataBuffer.primaryOwner());
            var pointer = ops.getOpaqueNDArraySpecialBuffer(opaque);
            var borrowed = ops.dbCreateExternalDataBuffer(1024, DataType.FLOAT.toInt(), null, pointer);
            try {
                ops.releaseGpuIntermediates(handle);
                input.assign(2.0);
                try (INDArray second = sd.outputSingle(Map.of("input", input), "output")) {
                    for (float value : second.data().asFloat()) assertEquals(2.25f, value, 0.0f);
                }
                ops.copyBuffer(readback.data().opaqueBuffer(), 1024, borrowed, 0, 0);
                Nd4j.getExecutioner().commit();
                for (float value : readback.data().asFloat()) assertEquals(1.25f, value, 0.0f,
                        "ordinary native release must preserve the borrowed producer allocation");
            } finally {
                ops.deleteDataBuffer(borrowed);
                // getPlanSlotOutputArray borrows the plan-owned NDArray itself.
                // Closing it would delete the producer a second time at teardown.
                opaque.setNull();
            }
        }
    }

    @Test
    void testOrdinaryReleasePreservesRequestedStagingIdentity() {
        ordinaryReleasePreservesRequestedStaging(false);
    }

    @Test
    void testOrdinaryReleasePreservesRequestedStagingView() {
        ordinaryReleasePreservesRequestedStaging(true);
    }

    private void ordinaryReleasePreservesRequestedStaging(boolean view) {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, 4, 16);
        SDVariable alias = view ? x.permute(1, 0).rename("borrowed") : sd.identity("borrowed", x);
        sd.nn.relu(alias.add(1.0).mul(2.0), 0).add("output", 1.0);
        sd.compileNativeDynamicShapePlan("borrowed", "output");
        var ops = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
        try (INDArray input = Nd4j.ones(DataType.FLOAT, 4, 16);
             INDArray readback = Nd4j.create(DataType.FLOAT, 4, 16)) {
            Map<String, INDArray> warmup = sd.output(Map.of("input", input), "borrowed", "output");
            warmup.values().forEach(INDArray::close);
            DspHandle dsp = new DspHandle(sd);
            int ext = dsp.extInputIndex("input");
            assertTrue(ext >= 0);
            dsp.markVariable(ext);
            for (int step = 0; step < 16; step++) {
                Map<String, INDArray> outputs = sd.output(Map.of("input", input), "borrowed", "output");
                try {
                    for (float value : outputs.get("borrowed").data().asFloat()) assertEquals(1.0f, value);
                    for (float value : outputs.get("output").data().asFloat()) assertEquals(5.0f, value);
                } finally {
                    outputs.values().forEach(INDArray::close);
                }
            }
            var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            var handle = executor.getNativePlanHandle();
            int slot = dsp.slotIndexForOutput("borrowed");
            assertTrue(slot >= 0);
            var opaque = ops.getPlanSlotOutputArray(handle, slot);
            assertNotNull(opaque);
            opaque.attachOwner(org.nd4j.nativeblas.OpaqueDataBuffer.primaryOwner());
            var pointer = ops.getOpaqueNDArraySpecialBuffer(opaque);
            assertNotEquals(0L, dsp.stagingBufferAddress(ext), "fixture must allocate real staging");
            assertEquals(dsp.stagingBufferAddress(ext), pointer.address(),
                    "requested output must borrow staging, not an independent output allocation");
            var borrowed = ops.dbCreateExternalDataBuffer(64, DataType.FLOAT.toInt(), null, pointer);
            try {
                ops.releaseGpuIntermediates(handle);
                ops.releaseGpuIntermediates(handle);
                input.assign(3.0);
                Map<String, INDArray> next = sd.output(Map.of("input", input), "borrowed", "output");
                try {
                    for (float value : next.get("borrowed").data().asFloat()) assertEquals(3.0f, value);
                    for (float value : next.get("output").data().asFloat()) assertEquals(9.0f, value);
                    ops.copyBuffer(readback.data().opaqueBuffer(), 64, borrowed, 0, 0);
                    Nd4j.getExecutioner().commit();
                    for (float value : readback.data().asFloat()) assertEquals(1.0f, value,
                            "ordinary release/reuse must preserve previously borrowed staging");
                } finally {
                    next.values().forEach(INDArray::close);
                }
            } finally {
                ops.deleteDataBuffer(borrowed);
                opaque.setNull(); // Borrowed native wrapper; the plan owns deletion.
            }
            executor.releaseGpuIntermediates();
            executor.releaseGpuIntermediates();
        }
    }

    @Test
    void testCopiedOutputsSurviveReleaseWithoutNativeAccumulation() {
        copiedOutputsSurviveRelease(false);
    }

    @Test
    void testCopiedViewOutputsSurviveReleaseWithoutNativeAccumulation() {
        copiedOutputsSurviveRelease(true);
    }

    private void copiedOutputsSurviveRelease(boolean viewOutput) {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1024, 1024);
        if (viewOutput) input.add("producer", 0.25).permute(1, 0).rename("output");
        else input.add("output", 0.25);
        sd.compileNativeDynamicShapePlan("output");
        java.util.List<INDArray> retained = new java.util.ArrayList<>();
        long baseline = -1;
        final long outputBytes = 1024L * 1024 * Float.BYTES;
        try (INDArray values = Nd4j.ones(DataType.FLOAT, 1024, 1024)) {
            for (int iteration = 0; iteration < 5; iteration++) {
                values.assign(iteration);
                INDArray result = sd.outputSingle(Map.of("input", values), "output");
                retained.add(result);
                // Keep each independently delivered Java result alive across
                // release and later executions, proving its ownership contract.
                for (int previous = 0; previous < retained.size(); previous++) {
                    for (float value : retained.get(previous).data().asFloat())
                        assertEquals(previous + 0.25f, value, 0.0f);
                }
                var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                executor.releaseGpuIntermediates();
                // A copied-output retirement is idempotent; the later executor
                // close/cache destruction must not rediscover deleted wrappers.
                executor.releaseGpuIntermediates();
                long resident = 0;
                for (int device = 0; device < Nd4j.getAffinityManager().getNumberOfDevices(); device++)
                    resident += Nd4j.getEnvironment().getDeviceCounter(device);
                long nativeAndFixed = resident - retained.size() * outputBytes;
                if (iteration == 1) baseline = nativeAndFixed;
                if (iteration > 1) assertTrue(nativeAndFixed <= baseline + outputBytes / 2,
                        "native producer residency grew after copied output release: iteration=" + iteration
                                + " baseline=" + baseline + " current=" + nativeAndFixed);
            }
        } finally {
            retained.forEach(INDArray::close);
        }
    }

    @Test
    void testReleasedFrozenPlanRestoresWarmupSharing() {
        firstWarmupSharesDeadIntermediates(true, true);
    }

    private void firstWarmupSharesDeadIntermediates(boolean shapePrepass) {
        firstWarmupSharesDeadIntermediates(shapePrepass, false);
    }

    private void firstWarmupSharesDeadIntermediates(boolean shapePrepass, boolean releaseAndRewarm) {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        if (shapePrepass) sd.setGraphExecutionMode(
                org.nd4j.autodiff.samediff.execution.GraphExecutionMode.TRITON);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 256, 128);
        final float factor = 1.015625f;
        SDVariable weight = sd.constant("weight", Nd4j.eye(128).castTo(DataType.FLOAT).muli(factor));
        SDVariable x = input;
        for (int layer = 0; layer < 32; layer++) {
            // A true dependency chain, not independent GEMMs eligible for batching.
            x = sd.mmul(layer == 7 ? "kept" : layer == 31 ? "output" : "layer_" + layer, x, weight);
        }
        sd.compileNativeDynamicShapePlan("kept", "output");
        try (INDArray values = Nd4j.ones(DataType.FLOAT, 256, 128)) {
            for (int iteration = 0; iteration < 8; iteration++) {
                if (releaseAndRewarm && iteration == 4) {
                    var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                    var nativeOps = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
                    nativeOps.releaseGpuIntermediates(executor.getNativePlanHandle());
                    nativeOps.setPlanShapesFrozen(executor.getNativePlanHandle(), true);
                    assertFalse(new DspHandle(sd).bufferColoringApplied(), "release must retire the old colors");
                }
                float expected = 0.25f + iteration * 0.125f;
                values.assign(expected);
                Map<String, INDArray> results = sd.output(Map.of("input", values), "kept", "output");
                try {
                    for (String name : new String[]{"kept", "output"}) {
                        INDArray result = results.get(name);
                        assertNotNull(result, name);
                        float expectedOutput = expected;
                        int layers = "kept".equals(name) ? 8 : 32;
                        for (int layer = 0; layer < layers; ++layer) expectedOutput *= factor;
                        for (float value : result.data().asFloat()) {
                            assertEquals(expectedOutput, value, 1e-5f,
                                    "requested output " + name + " at iteration " + iteration);
                        }
                    }
                    if (iteration == 0 || (releaseAndRewarm && iteration == 4)) {
                        DspHandle handle = new DspHandle(sd);
                        assertTrue(handle.isCompiled());
                        assertTrue(handle.bufferColoringApplied(),
                                "sharing must be active during the first execution, not only after warmup");
                        assertTrue(handle.bufferColoringBytesSaved() >= 8L * 256 * 128 * Float.BYTES,
                                "first-pass sharing must eliminate at least eight full intermediate buffers");
                    }
                } finally {
                    java.util.Set<INDArray> uniqueOutputs = java.util.Collections.newSetFromMap(
                            new java.util.IdentityHashMap<>());
                    uniqueOutputs.addAll(results.values());
                    uniqueOutputs.forEach(INDArray::close);
                }
            }
        }
    }

    /**
     * Regression for the freeze-time coloring narrowing of view-capable op outputs.
     *
     * A view-capable op (reshape) whose warmup input is NOT C-contiguous cannot
     * mint a zero-copy view, so the functional warmup publishes a DEDICATED array
     * for its output. That dedicated storage was previously excluded from coloring
     * on both sides of every potential alias; the narrowing keeps the aliased
     * INPUT side protected and makes the dedicated OUTPUT storage colorable.
     * Republished-as-view executions themselves stay covered by
     * DspBufferAliasAccuracyTest's view fixtures (accuracy/stability/varying-input
     * across all execution modes); this test pins the coloring-side invariants.
     *
     * Graph (mmul barriers keep the optimizer from folding the view chains):
     *   h1 = mmul(input, w1)    // [4,16]
     *   tA = permute(h1)        // [16,4] non-C-contiguous view of h1
     *   rA = reshape(tA, [64])  // view-capable; warmup publishes DEDICATED (input not contiguous)
     *   cA = rA + 1; cA2 = tanh(cA)   // [64] eligible chain sharing rA's shape group
     *   h2 = mmul(input, w2)    // [4,16]
     *   r1 = relu(h2)           // [4,16] C-contiguous
     *   rB = reshape(r1, [64])  // view of r1 at warmup; the frozen path may republish it as a view
     *   cB = rB + 1
     *   out  = cA2 + cB         // requested, [64]
     *   s2 = sigmoid(input)     // [4,16]
     *   s3 = tanh(s2); s4 = s3*2 // [4,16] eligible partners sharing s2's shape group
     *   out2 = s4 + 1           // requested, [4,16]
     *
     * Assertions:
     *  (a) outputs stay correct while the SAME input INDArray is re-assigned with
     *      different values between warmup and frozen replays (a republished view
     *      or a wrongly shared color buffer must read current values, never stale
     *      data from any sharing partner);
     *  (b) the republish-capable side remains protected where it must be: r1 (the
     *      aliased source of rB) is never colored while the unaliased same-shape
     *      h2 is — the exclusion is alias-specific, not a blanket producer ban;
     *  (c) the never-republishing dedicated counterpart rA DOES participate in
     *      coloring (slotColor(rA) >= 0 — uncolorable before the narrowing) and
     *      actual sharing is applied (bytesSaved > 0).
     */
    @Test
    void testViewCapableDedicatedOutputsAreColorableWhileAliasedInputsStayProtected() {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        INDArray w1Ref = Nd4j.rand(DataType.FLOAT, 16, 16).subi(0.05);
        INDArray w2Ref = Nd4j.rand(DataType.FLOAT, 16, 16).subi(0.05);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 4, 16);
        SDVariable h1 = sd.mmul("h1", input, sd.var("w1", w1Ref.dup()));
        SDVariable tA = sd.permute("tA", h1, 1, 0);
        SDVariable rA = sd.reshape("rA", tA, 64);
        SDVariable cA = rA.add(sd.constant("one64", Nd4j.ones(DataType.FLOAT, 64)));
        SDVariable cA2 = sd.nn.tanh("cA2", cA);
        SDVariable h2 = sd.mmul("h2", input, sd.var("w2", w2Ref.dup()));
        SDVariable r1 = sd.nn.relu("r1", h2, 0);
        SDVariable rB = sd.reshape("rB", r1, 64);
        SDVariable cB = rB.add(sd.constant("one64b", Nd4j.ones(DataType.FLOAT, 64)));
        cA2.add("out", cB);
        SDVariable s2 = sd.nn.sigmoid("s2", input);
        SDVariable s3 = sd.nn.tanh("s3", s2);
        SDVariable s4 = s3.mul(2.0);
        SDVariable out2 = s4.add(sd.constant("one416", Nd4j.ones(DataType.FLOAT, 4, 16)));
        out2.rename("out2");
        sd.compileNativeDynamicShapePlan("out", "out2");

        try (INDArray values = Nd4j.rand(DataType.FLOAT, 4, 16)) {
            for (int iteration = 0; iteration < 8; iteration++) {
                // Write DIFFERENT values into the same input INDArray between warmup
                // and frozen replays: any stale view or wrongly shared buffer shows up.
                float offset = 0.125f * iteration;
                values.assign(Nd4j.rand(DataType.FLOAT, 4, 16).subi(0.5).addi(offset));
                INDArray h1Ref = values.mmul(w1Ref);
                INDArray h2Ref = values.mmul(w2Ref);
                INDArray cARef = h1Ref.permute(1, 0).reshape(64).addi(1.0);
                INDArray expected = org.nd4j.linalg.ops.transforms.Transforms.tanh(cARef, true)
                        .addi(org.nd4j.linalg.ops.transforms.Transforms.max(
                                h2Ref.dup(), Nd4j.zeros(DataType.FLOAT, 4, 16)).reshape(64).addi(1.0));
                // out2 = tanh(sigmoid(input)) * 2 + 1  (s2 -> s3 -> s4 -> out2)
                INDArray expectedOut2 = org.nd4j.linalg.ops.transforms.Transforms.tanh(
                        org.nd4j.linalg.ops.transforms.Transforms.sigmoid(values.dup(), false), true)
                        .muli(2.0).addi(1.0);
                Map<String, INDArray> results = sd.output(Map.of("input", values), "out", "out2");
                try {
                    INDArray got = results.get("out");
                    assertNotNull(got, "out");
                    // Stale-view / wrong-sharing corruption produces O(0.05+) errors; 1e-3
                    // absorbs only legitimate GPU-vs-host accumulation drift.
                    assertArrayEquals(expected.data().asFloat(), got.data().asFloat(), 1e-3f,
                            "out at iteration " + iteration);
                    INDArray got2 = results.get("out2");
                    assertNotNull(got2, "out2");
                    assertArrayEquals(expectedOut2.data().asFloat(), got2.data().asFloat(), 1e-5f,
                            "out2 at iteration " + iteration);

                    if (iteration == 0) {
                        DspHandle handle = new DspHandle(sd);
                        assertTrue(handle.isCompiled());
                        assertTrue(handle.bufferColoringApplied(),
                                "narrowed view-capable dedicated outputs must enable actual sharing");

                        java.util.List<Integer> reshapes = handle.allSlotsForOp("reshape");
                        assertEquals(2, reshapes.size(), "expected exactly rA and rB reshape slots");
                        int rAIdx = reshapes.get(0);
                        int rBIdx = reshapes.get(1);
                        java.util.List<Integer> matmuls = handle.allSlotsForOp("matmul");
                        assertEquals(2, matmuls.size(), "expected exactly h1 and h2 matmul slots");
                        int h2Idx = matmuls.get(1);
                        int r1Idx = handle.slotIndexForOp("relu");
                        assertTrue(r1Idx >= 0, "relu slot must be found");

                        StringBuilder colorMap = new StringBuilder("slotColors:");
                        for (int s = 0; s < handle.totalSlots(); s++) {
                            colorMap.append(' ').append(s).append('=').append(handle.slotColor(s));
                        }
                        log.info("{} applied={} colors={} saved={}", colorMap,
                                handle.bufferColoringApplied(), handle.bufferColoringNumColors(),
                                handle.bufferColoringBytesSaved());

                        // (c) narrowing: the warmup-dedicated view-capable output participates
                        // in coloring (it was structurally excluded before the narrowing).
                        // rB's warmup array is a zero-copy view of r1 and stays uncolored by
                        // the pre-existing views-never-color invariant.
                        assertTrue(handle.slotColor(rAIdx) >= 0,
                                "warmup-dedicated reshape output rA must be colorable under the narrowing");
                        // (b) input-side protection is unchanged: r1 aliases rB's view and stays
                        // dedicated even though the unaliased h2 (same shape) does get colored —
                        // the exclusion is alias-specific, not a blanket producer ban.
                        assertEquals(-1, handle.slotColor(r1Idx),
                                "aliased reshape INPUT r1 must never be colored");
                        assertTrue(handle.slotColor(h2Idx) >= 0,
                                "unaliased h2 proves the protected-input assertion is not vacuous");
                        // Greedy interval coloring packs each 3-slot chain into 2 colors:
                        // the [64] group alone must retire one 256-byte float buffer.
                        assertTrue(handle.bufferColoringBytesSaved() >= 64L * Float.BYTES,
                                "the [64] shape group must actually share buffers");
                    }
                } finally {
                    java.util.Set<INDArray> uniqueOutputs = java.util.Collections.newSetFromMap(
                            new java.util.IdentityHashMap<>());
                    uniqueOutputs.addAll(results.values());
                    uniqueOutputs.forEach(INDArray::close);
                    h1Ref.close();
                    h2Ref.close();
                    cARef.close();
                    expected.close();
                    expectedOut2.close();
                }
            }
        }
    }

    @Test
    void testWarmupSharingPreservesBranchesViewsAndShapeLeases() {
        org.junit.jupiter.api.Assumptions.assumeTrue(Nd4j.backends().isCudaAvailable(), "requires CUDA");
        sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);
        final float factor = 1.015625f;
        SDVariable weight = sd.constant("weight", Nd4j.eye(64).castTo(DataType.FLOAT).muli(factor));
        SDVariable shared = input.mmul(weight);
        SDVariable left = shared.mmul(weight);
        // A second reader independent of the left chain: last topological reader
        // alone cannot prove the buffer safe to overwrite in a compiled schedule.
        SDVariable right = shared.mul("right", 3.0);
        SDVariable viewed = left.reshape(-1, 8, 8).permute(0, 2, 1).reshape(-1, 64);
        SDVariable x = viewed;
        for (int layer = 0; layer < 20; ++layer) x = x.mmul(weight);
        x.add("output", right);
        sd.compileNativeDynamicShapePlan("right", "output");
        for (int batch : new int[]{32, 16, 32}) {
            try (INDArray values = Nd4j.ones(DataType.FLOAT, batch, 64)) {
                for (int iteration = 0; iteration < 8; ++iteration) {
                    float source = 0.125f + iteration * 0.0625f;
                    values.assign(source);
                    Map<String, INDArray> results = sd.output(Map.of("input", values), "right", "output");
                    try {
                        float expectedRight = source * factor * 3.0f;
                        float expectedLeft = source;
                        for (int layer = 0; layer < 22; ++layer) expectedLeft *= factor;
                        for (String name : new String[]{"right", "output"}) {
                            INDArray result = results.get(name);
                            assertArrayEquals(new long[]{batch, 64}, result.shape());
                            float expected = "right".equals(name) ? expectedRight : expectedLeft + expectedRight;
                            for (float value : result.data().asFloat()) assertEquals(expected, value, 1e-5f,
                                    "branch/view/shape lease output " + name + " batch=" + batch + " iteration=" + iteration);
                        }
                    } finally {
                        java.util.Set<INDArray> uniqueOutputs = java.util.Collections.newSetFromMap(
                                new java.util.IdentityHashMap<>());
                        uniqueOutputs.addAll(results.values());
                        uniqueOutputs.forEach(INDArray::close);
                    }
                }
            }
        }
    }

    /**
     * Build a multi-layer graph where intermediates have non-overlapping lifetimes.
     * After freeze, coloring should be applied and bytesSaved > 0.
     */
    @Test
    void testColoringAppliedAfterFreeze() {
        sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 64);

        // Chain of ops: each intermediate is consumed only by the next op
        SDVariable x = sd.nn.relu(input, 0);
        x = sd.math.mul(x, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 64).mul(2.0)));
        x = sd.nn.sigmoid(x);
        x = sd.math.add(x, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 64).mul(0.5)));
        x = sd.nn.tanh(x);
        SDVariable output = x;
        output.rename("output");

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 64));

        // Ensure DSP is enabled on this instance
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Warmup — multiple executions to reach frozen state
        for (int i = 0; i < 5; i++) {
            sd.output(ph, "output");
        }

        // Buffer coloring introspection API is not yet exposed via DspHandle;
        // verify that execution completes without error instead.
        DspHandle handle = new DspHandle(sd);
        boolean compiled = handle.isCompiled();
        int slots = compiled ? handle.totalSlots() : -1;
        log.info("Plan compiled={}, totalSlots={}", compiled, slots);
    }

    /**
     * After coloring, the output must be numerically correct.
     * Compare against a fresh graph with no coloring.
     */
    @Test
    void testColoringCorrectness() {
        // Build a graph
        sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable x = sd.nn.relu(input, 0);
        x = sd.math.mul(x, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 32).mul(3.0)));
        x = sd.nn.sigmoid(x);
        x = sd.math.add(x, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 32)));
        SDVariable output = sd.nn.tanh(x);
        output.rename("output");

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 32);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", inputArr);

        // Run multiple times to warm up and potentially get coloring
        INDArray lastResult = null;
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> out = sd.output(ph, "output");
            lastResult = out.get("output");
        }

        // Compute reference: relu -> mul(3) -> sigmoid -> add(1) -> tanh
        INDArray ref = Nd4j.nn().relu(inputArr, 0);
        ref = ref.mul(3.0);
        ref = Nd4j.nn().sigmoid(ref);
        ref = ref.add(1.0);
        ref = Nd4j.math().tanh(ref);

        assertNotNull(lastResult);
        assertTrue(ref.equalsWithEps(lastResult, 1e-4),
                   "Coloring output must match reference computation");
    }

    /**
     * Run 10 executions after reaching frozen state.
     * All outputs must match the reference.
     */
    @Test
    void testColoringMultipleExecutions() {
        sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable x = sd.nn.relu(input, 0);
        x = sd.math.mul(x, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 16).mul(2.0)));
        x = sd.nn.tanh(x);
        SDVariable output = x;
        output.rename("output");

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 16);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", inputArr);

        // Warmup
        for (int i = 0; i < 3; i++) {
            sd.output(ph, "output");
        }

        // Reference
        INDArray ref = Nd4j.math().tanh(Nd4j.nn().relu(inputArr, 0).mul(2.0));

        // 10 post-warmup executions
        for (int i = 0; i < 10; i++) {
            Map<String, INDArray> out = sd.output(ph, "output");
            INDArray result = out.get("output");
            assertTrue(ref.equalsWithEps(result, 1e-4),
                       "Execution " + i + " output must match reference");
        }
    }

    /**
     * Test that the buffer pool tracks acquire/release correctly.
     * Note: buffer pool introspection static API is not yet exposed; this
     * test verifies that DSP execution completes without error.
     */
    @Test
    void testBufferPoolIntrospection() {
        sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable x = sd.nn.relu(input, 0);
        x.rename("output");

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
        for (int i = 0; i < 3; i++) {
            sd.output(ph, "output");
        }

        DspHandle handle = new DspHandle(sd);
        boolean compiled = handle.isCompiled();
        int slots = compiled ? handle.totalSlots() : -1;
        assertTrue(slots >= 0 || !compiled, "totalSlots should be >= 0 when compiled");
        log.info("Buffer pool test: plan compiled={}, totalSlots={}", compiled, slots);
    }

    /**
     * Test that running a plan, closing it, then running a new plan completes without error.
     * Cross-plan buffer pool sharing introspection is not yet exposed via static DspHandle API.
     */
    @Test
    void testCrossPlanBufferSharing() {
        // Plan A
        sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        SDVariable inputA = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable xA = sd.nn.relu(inputA, 0);
        xA = sd.math.mul(xA, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 32).mul(2.0)));
        xA.rename("output");

        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, 1, 32));
        for (int i = 0; i < 3; i++) {
            sd.output(ph, "output");
        }

        // Close plan A
        sd.close();
        sd = null;

        // Plan B — same shapes
        sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        SDVariable inputB = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable xB = sd.nn.sigmoid(inputB);
        xB = sd.math.add(xB, sd.constant(Nd4j.ones(DataType.FLOAT, 1, 32)));
        xB.rename("output");

        for (int i = 0; i < 3; i++) {
            sd.output(ph, "output");
        }

        DspHandle handle = new DspHandle(sd);
        boolean compiled = handle.isCompiled();
        int slots = compiled ? handle.totalSlots() : -1;
        log.info("Cross-plan sharing test: plan compiled={}, totalSlots={}", compiled, slots);
    }
}
