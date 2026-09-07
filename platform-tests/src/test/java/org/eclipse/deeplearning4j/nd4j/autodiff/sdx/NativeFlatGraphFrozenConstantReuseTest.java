/*
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for copyright ownership.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.sdx;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;
import com.sun.jna.ptr.IntByReference;
import org.bytedeco.javacpp.Loader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * CPU numerical regression for constant values whose colored storage is transient.
 * NativePlanCompiler, unlike the Java serialized-plan route, supplies slot liveness.
 * No SameDiff execution, optimizer, intermediate requested outputs, or execution-mode
 * overrides are used. CPU scope is intentional: raw JNA host reads below are not a
 * device synchronization API and must not be presented as CUDA/Vulkan coverage.
 */
public class NativeFlatGraphFrozenConstantReuseTest extends BaseND4JTest {
    private static final int WIDTH = 8;
    private static final int LAYERS = 3;
    private static final int FROZEN_CONSTANT = 3; // NativeOpsDsp.h SlotState

    @TempDir
    Path tempDir;

    @Override
    public long getTimeoutMilliseconds() {
        return 60_000L;
    }

    /** Existing exported C functions only; C++ bool is one byte, not JNA BOOL. */
    public interface DspApi extends Library {
        Pointer loadModelFromFile(String path);
        Pointer compileModelPlan(Pointer model, StringArray outputNames, int numOutputs);
        void freeLoadedModel(Pointer model);
        void freeDynamicShapePlan(Pointer plan);
        void setPlanShapesFrozen(Pointer plan, byte frozen);
        int executeDynamicShapePlan(Pointer plan, Pointer context, Pointer stream);
        int getPlanNumSlots(Pointer plan);
        int getTotalPlanOutputSlots(Pointer plan);
        int getPlanNumRequestedOutputs(Pointer plan);
        int getPlanNumExternalInputs(Pointer plan);
        String getPlanExternalInputName(Pointer plan, int index);
        Pointer getLoadedModelVariable(Pointer model, String name);
        byte getPlanIsExternalInputVariable(Pointer plan, int index);
        String getPlanSlotOpName(Pointer plan, int slot);
        int getPlanSlotIOCounts(Pointer plan, int slot, IntByReference inputs, IntByReference outputs);
        Pointer getPlanSlotOutputArray(Pointer plan, int slot);
        byte getPlanBufferColoringApplied(Pointer plan);
        int getPlanSlotColor(Pointer plan, int slot);
        int getPlanSlotState(Pointer plan, int slot);
        Pointer createGraphContext(int nodeId);
        void deleteGraphContext(Pointer context);
        void setGraphContextInputArray(Pointer context, int index, Pointer array);
        Pointer getOutputArrayNative(Pointer context, int index);
        Pointer getOpaqueNDArrayShapeInfo(Pointer array);
        Pointer getOpaqueNDArrayBuffer(Pointer array);
    }

    @Test
    public void testNativeFlatGraphReusedBf16CastsRemainNumericallyLive() throws Exception {
        assumeTrue(Nd4j.getBackend().getClass().getName().toLowerCase(Locale.ROOT).contains("cpu"),
                "CPU-native regression; run with -Dbackend.artifactId=nd4j-native");
        // Bind exactly the library JavaCPP loaded, not an arbitrary old cache hit
        // or a concurrently-produced build-tree .so. Input NDArray pointers cross
        // this boundary, so both bindings must address the same library instance.
        Nd4j.getNativeOps();
        Set<String> libraries = new LinkedHashSet<>();
        String libraryName = System.mapLibraryName("nd4jcpu");
        for (String path : Loader.getLoadedLibraries().values()) {
            if (path != null && new File(path).getName().equals(libraryName)) {
                libraries.add(new File(path).getCanonicalPath());
            }
        }
        assertEquals(1, libraries.size(), "Need one loaded CPU library, found " + libraries);
        DspApi api = Native.load(libraries.iterator().next(), DspApi.class);

        SameDiff sd = SameDiff.create();
        SDVariable activation = sd.placeHolder("x", DataType.FLOAT, 1, WIDTH);
        float[][] weights = new float[LAYERS][WIDTH * WIDTH];
        INDArray[] weightArrays = new INDArray[LAYERS];
        Pointer model = null;
        Pointer plan = null;
        Pointer context = null;
        INDArray input = Nd4j.create(DataType.FLOAT, 1, WIDTH);
        try {
            for (int layer = 0; layer < LAYERS; layer++) {
                for (int row = 0; row < WIDTH; row++) {
                    for (int col = 0; col < WIDTH; col++) {
                        // Small dyadic values, exactly representable in BF16.
                        weights[layer][row * WIDTH + col] =
                                (((layer + 1) * (row + 1) + 3 * col) % 9 - 4) / 8.0f
                                        + (row == col ? 1.0f : 0.0f);
                    }
                }
                try (INDArray fp32 = Nd4j.createFromArray(weights[layer]).reshape('c', WIDTH, WIDTH)) {
                    weightArrays[layer] = fp32.castTo(DataType.BFLOAT16);
                }
                SDVariable cast = sd.constant("w" + layer, weightArrays[layer])
                        .castTo("cast" + layer, DataType.FLOAT);
                // Interleave construction naturally; do not reorder serialized nodes.
                activation = sd.linalg.mmul(layer == LAYERS - 1 ? "output" : "h" + layer,
                        activation, cast);
            }
            File bundle = tempDir.resolve("frozen-constant-reuse.sdz").toFile();
            SDZSerializer.save(sd, bundle, false, Collections.emptyMap()); // NOT saveOptimized
            model = api.loadModelFromFile(bundle.getAbsolutePath());
            assertNotNull(model, "native loadModelFromFile failed");
            plan = api.compileModelPlan(model, new StringArray(new String[]{"output"}), 1);
            assertNotNull(plan, "native FlatGraph compilation failed");
            assertEquals(1, api.getPlanNumRequestedOutputs(plan), "Only the final output may be pinned");
            assertEquals(2 * LAYERS, api.getPlanNumSlots(plan), "Casts must survive compilation");
            assertEquals(2 * LAYERS, api.getTotalPlanOutputSlots(plan));
            for (int slot = 0; slot < 2 * LAYERS; slot++) {
                assertEquals(slot % 2 == 0 ? "cast" : "matmul", api.getPlanSlotOpName(plan, slot),
                        "Expected cast/consumer interleaving at slot " + slot);
                IntByReference inputs = new IntByReference();
                IntByReference outputs = new IntByReference();
                assertEquals(0, api.getPlanSlotIOCounts(plan, slot, inputs, outputs));
                assertEquals(slot % 2 == 0 ? 1 : 2, inputs.getValue());
                // One output per op establishes op-index == output-slot-index.
                assertEquals(1, outputs.getValue());
            }

            context = api.createGraphContext(0);
            assertNotNull(context);
            assertEquals(LAYERS + 1, api.getPlanNumExternalInputs(plan));
            int liveInput = -1;
            for (int i = 0; i < api.getPlanNumExternalInputs(plan); i++) {
                String name = api.getPlanExternalInputName(plan, i);
                if ("x".equals(name)) {
                    assertEquals(-1, liveInput, "Duplicate placeholder");
                    liveInput = i;
                    assertNotEquals(0, api.getPlanIsExternalInputVariable(plan, i), "x must remain live");
                } else {
                    assertTrue(name != null && name.matches("w[0-2]"), "Unexpected external " + name);
                    assertEquals(0, api.getPlanIsExternalInputVariable(plan, i), "Weights must be constant");
                    Pointer weight = api.getLoadedModelVariable(model, name);
                    assertNotNull(weight, "Missing model-owned " + name);
                    assertNativeShapeAndType(api, weight, WIDTH, WIDTH, DataType.BFLOAT16);
                    api.setGraphContextInputArray(context, i, weight);
                }
            }
            assertTrue(liveInput >= 0, "Native compiler lost the live input");
            api.setPlanShapesFrozen(plan, (byte) 1); // compile -> freeze -> functional warmup
            for (int execution = 0; execution < 6; execution++) {
                double[] expected = new double[WIDTH];
                for (int col = 0; col < WIDTH; col++) {
                    // Repeat one input, then vary it, then return to the first input.
                    float value = (col - 3) / 4.0f + (execution % 3 == 2 ? 0.5f : 0.0f);
                    input.putScalar(0, col, value);
                    expected[col] = value;
                }
                for (float[] weight : weights) {
                    double[] next = new double[WIDTH];
                    for (int col = 0; col < WIDTH; col++) {
                        for (int row = 0; row < WIDTH; row++) {
                            next[col] += expected[row] * weight[row * WIDTH + col];
                        }
                    }
                    expected = next;
                }
                api.setGraphContextInputArray(context, liveInput,
                        new Pointer(input.getOrCreateOpaqueNDArray().address()));
                assertEquals(0, api.executeDynamicShapePlan(plan, context, null),
                        "Native execution " + execution + " failed");
                assertNotEquals(0, api.getPlanBufferColoringApplied(plan),
                        "Fixture did not exercise native coloring (not a numerical pass)");
                int color = api.getPlanSlotColor(plan, 0);
                assertTrue(color >= 0, "First BF16->FLOAT cast was not colored");
                Pointer shared = null;
                for (int layer = 0; layer < LAYERS; layer++) {
                    int castSlot = 2 * layer;
                    assertEquals(color, api.getPlanSlotColor(plan, castSlot),
                            "Cast lifetimes must share one color, layer " + layer);
                    Pointer castArray = api.getPlanSlotOutputArray(plan, castSlot);
                    assertNativeShapeAndType(api, castArray, WIDTH, WIDTH, DataType.FLOAT);
                    Pointer buffer = api.getOpaqueNDArrayBuffer(castArray);
                    assertNotNull(buffer, "Colored cast has no backing storage");
                    if (shared == null) shared = buffer;
                    else assertEquals(Pointer.nativeValue(shared), Pointer.nativeValue(buffer),
                            "Equal color must actually share native storage");
                }
                Pointer output = api.getOutputArrayNative(context, 0);
                assertNativeShapeAndType(api, output, 1, WIDTH, DataType.FLOAT);
                Pointer data = api.getOpaqueNDArrayBuffer(output);
                assertNotNull(data);
                // CPU output is synchronous. Read only the requested output; merely
                // inspect cast metadata/addresses without wrapping or retaining them.
                float[] actual = data.getFloatArray(0, WIDTH);
                for (int col = 0; col < WIDTH; col++) {
                    assertEquals(expected[col], actual[col], 1e-5,
                            "execution=" + execution + ", col=" + col + ", castColor=" + color);
                }
                // Keep the numerical assertion first: OLD binaries must demonstrate
                // corrupt output, not just a changed classification enum.
                if (execution >= 2) {
                    for (int layer = 0; layer < LAYERS; layer++) {
                        int state = api.getPlanSlotState(plan, 2 * layer);
                        assertTrue(state >= 0 && state < FROZEN_CONSTANT,
                                "Recycled cast storage cannot be FROZEN_CONSTANT: layer=" + layer);
                    }
                }
            }
        } finally {
            if (context != null) api.deleteGraphContext(context);
            if (plan != null) api.freeDynamicShapePlan(plan);
            if (model != null) api.freeLoadedModel(model);
            input.close(); // only after the native plan releases its input references
            sd.close(); // SameDiff owns the constant arrays (plain weight.close() is blocked)
            for (INDArray weight : weightArrays) {
                if (weight != null && !weight.wasClosed() && weight.closeable()) weight.close();
            }
        }
    }

    private static void assertNativeShapeAndType(DspApi api, Pointer array, long rows, long cols,
                                                 DataType dtype) {
        assertNotNull(array, "Missing native array");
        Pointer info = api.getOpaqueNDArrayShapeInfo(array);
        assertNotNull(info);
        assertEquals(2L, info.getLong(0), "Expected rank two");
        long[] shapeInfo = info.getLongArray(0, 8); // rank * 2 + 4
        assertArrayEquals(new long[]{rows, cols}, Shape.shape(shapeInfo));
        assertEquals(dtype, ArrayOptionsHelper.dataType(Shape.extras(shapeInfo)));
        if (rows == 1) assertEquals(1L, Shape.stride(shapeInfo)[1], "Output must be contiguous");
    }
}
