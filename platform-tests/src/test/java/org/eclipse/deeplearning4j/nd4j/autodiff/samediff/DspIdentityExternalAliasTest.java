/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Named;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.NativeExecutionBinding;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * An identity over an external input publishes the caller's array as its own output.
 * Callers routinely pass a new array on every call, close their inputs once output()
 * returns, and execute the same plan through execution bindings whose input wrappers die
 * with the binding. None of that may leave the identity's output on an earlier call's
 * array: every call must compute from the array it passed.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DspIdentityExternalAliasTest {
    private static final int STEPS = 32;
    private static final int ROWS = 4;
    private static final int COLS = 16;
    private static final int OUT_COLS = 8;

    enum Graph {
        /** identity feeds elementwise ops that Triton can lower. */
        ELEMENTWISE,
        /** identity feeds a matmul, which runs as a native op. */
        MATMUL,
        /** the identity itself is requested next to its consumer. */
        REQUESTED_ALIAS,
        /** an identity of an identity. */
        CHAIN,
        /** a reshape of the input: a view the plan creates over the caller's buffer. */
        RESHAPE,
        /** a strided view of the identity, transposed back for the output. */
        VIEW_CHAIN,
        /** each call's input is the output the previous call returned. */
        PING_PONG
    }

    static Stream<Arguments> cases() {
        List<Arguments> cases = new ArrayList<>();
        for (Graph graph : Graph.values()) {
            for (boolean compileAll : new boolean[]{false, true}) {
                cases.add(Arguments.of(
                        Named.of(graph + (compileAll ? " compileAll" : " default"), graph), compileAll));
            }
        }
        return cases.stream();
    }

    @ParameterizedTest
    @MethodSource("cases")
    @Order(1)
    void freshCallerArraysKeptAlive(Graph graph, boolean compileAll) {
        runFreshCallerArrays(graph, compileAll, false);
    }

    @Test
    @Order(2)
    void executionBindingsDoNotPinTheAliasToTheirWrappers() {
        var environment = Nd4j.getEnvironment();
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean capture = environment.tritonGraphCapture();
        List<INDArray> callerArrays = new ArrayList<>();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            environment.setTritonGraphCapture(true);
            try (SameDiff sd = build(Graph.ELEMENTWISE)) {
                int step = 0;
                for (int round = 0; round < 3; round++) {
                    for (int call = 0; call < 8; call++, step++) {
                        INDArray x = freshInput(step);
                        callerArrays.add(x);
                        assertOutputs(sd, Graph.ELEMENTWISE, step,
                                sd.output(Map.of("x", x), "out"), "round=" + round + " output step=" + step);
                    }
                    DynamicShapePlanExecutor executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
                    assertNotNull(executor);
                    // The binding borrows the last caller array and binds its own wrapper for it.
                    try (NativeExecutionBinding binding = executor.captureNativeExecutionBinding()) {
                        INDArray bound = binding.getExternalInputsSnapshot()[binding.findExternalInputIndex("x")];
                        for (int call = 0; call < 4; call++, step++) {
                            try (INDArray values = freshInput(step)) {
                                bound.assign(values);
                            }
                            assertBoundOutput(sd, binding, step, "round=" + round + " bound step=" + step);
                        }
                    }
                    // The binding's wrappers are gone; the next calls bind new caller arrays.
                }
                DspPlanAssertions.assertFrozenExecCountAtLeast(sd, STEPS / 2,
                        "the steady-state path must run");
            }
        } finally {
            for (INDArray x : callerArrays) {
                if (!x.wasClosed()) x.close();
            }
            environment.setTritonGraphCapture(capture);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    @ParameterizedTest
    @MethodSource("cases")
    @Order(3)
    void freshCallerArraysClosedAfterEachCall(Graph graph, boolean compileAll) {
        runFreshCallerArrays(graph, compileAll, true);
    }

    private static void runFreshCallerArrays(Graph graph, boolean compileAll, boolean closeInputs) {
        boolean cuda = Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA;
        assumeTrue(cuda || !compileAll, "Triton compile-all is a CUDA configuration");
        var environment = Nd4j.getEnvironment();
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean capture = environment.tritonGraphCapture();
        boolean previousCompileAll = environment.tritonCompileAll();
        List<INDArray> callerArrays = new ArrayList<>();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            environment.setTritonGraphCapture(true);
            environment.setTritonCompileAll(compileAll);
            try (SameDiff sd = build(graph)) {
                String[] requested = graph == Graph.REQUESTED_ALIAS
                        ? new String[]{"out", "xAlias"}
                        : new String[]{"out"};
                INDArray previousOut = null;
                for (int step = 0; step < STEPS; step++) {
                    INDArray x = previousOut != null ? previousOut : freshInput(step);
                    if (previousOut == null) callerArrays.add(x);
                    Map<String, INDArray> outputs = sd.output(Map.of("x", x), requested);
                    assertOutputs(sd, graph, step, outputs, "step=" + step);
                    if (closeInputs) {
                        // The caller owns its input. Once output() returned, closing it is legal.
                        x.close();
                    }
                    if (graph == Graph.PING_PONG) {
                        previousOut = outputs.get("out");
                        callerArrays.add(previousOut);
                    }
                }
                DspPlanAssertions.assertFrozenExecCountAtLeast(sd, STEPS / 2,
                        "the steady-state path must run");
            }
        } finally {
            for (INDArray x : callerArrays) {
                if (!x.wasClosed()) x.close();
            }
            environment.setTritonCompileAll(previousCompileAll);
            environment.setTritonGraphCapture(capture);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    private static SameDiff build(Graph graph) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, ROWS, COLS);
        SDVariable alias = sd.identity("xAlias", x);
        switch (graph) {
            case MATMUL:
                sd.mmul("projected", alias, sd.var("w", Nd4j.createFromArray(weights()))).add("out", 1.0);
                break;
            case CHAIN:
                sd.identity("xAlias2", alias).add(1.0).mul("out", 2.0);
                break;
            case RESHAPE:
                sd.reshape("out", sd.reshape("xFlat", x, ROWS * COLS).add(1.0).mul(2.0), ROWS, COLS);
                break;
            case VIEW_CHAIN:
                sd.permute("out", sd.permute("xT", alias, 1, 0).add(1.0).mul(2.0), 1, 0);
                break;
            case PING_PONG:
                alias.add("out", 1.0);
                break;
            default:
                alias.add(1.0).mul("out", 2.0);
                break;
        }
        return sd;
    }

    /** Distinct per step and exact in float, so a stale input can never match. */
    private static float input(int step, int row, int col) {
        return (row * COLS + col) * 0.25f + step;
    }

    private static INDArray freshInput(int step) {
        float[][] values = new float[ROWS][COLS];
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                values[row][col] = input(step, row, col);
            }
        }
        return Nd4j.createFromArray(values);
    }

    /** Small integers keep the matmul exact under any accumulation order. */
    private static float[][] weights() {
        float[][] w = new float[COLS][OUT_COLS];
        for (int k = 0; k < COLS; k++) {
            for (int col = 0; col < OUT_COLS; col++) {
                w[k][col] = (k * 3 + col) % 5 - 2;
            }
        }
        return w;
    }

    private static float[][] expectedOut(Graph graph, int step) {
        if (graph == Graph.MATMUL) {
            float[][] w = weights();
            float[][] out = new float[ROWS][OUT_COLS];
            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < OUT_COLS; col++) {
                    float sum = 0;
                    for (int k = 0; k < COLS; k++) {
                        sum += input(step, row, k) * w[k][col];
                    }
                    out[row][col] = sum + 1.0f;
                }
            }
            return out;
        }
        float[][] out = new float[ROWS][COLS];
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                // PING_PONG adds 1 per call, so its input at each step is input(step).
                out[row][col] = graph == Graph.PING_PONG
                        ? input(step, row, col) + 1.0f
                        : (input(step, row, col) + 1.0f) * 2.0f;
            }
        }
        return out;
    }

    private static void assertOutputs(SameDiff sd, Graph graph, int step,
                                      Map<String, INDArray> outputs, String context) {
        assertMatrix(sd, expectedOut(graph, step), outputs.get("out"), context + " out");
        if (graph == Graph.REQUESTED_ALIAS) {
            float[][] alias = new float[ROWS][COLS];
            for (int row = 0; row < ROWS; row++) {
                for (int col = 0; col < COLS; col++) {
                    alias[row][col] = input(step, row, col);
                }
            }
            assertMatrix(sd, alias, outputs.get("xAlias"), context + " xAlias");
        }
    }

    private static void assertMatrix(SameDiff sd, float[][] expected, INDArray actual, String context) {
        assertNotNull(actual, context);
        assertArrayEquals(new long[]{expected.length, expected[0].length}, actual.shape(), context);
        for (int row = 0; row < expected.length; row++) {
            for (int col = 0; col < expected[row].length; col++) {
                int r = row;
                int c = col;
                Supplier<String> message = () -> context + " [" + r + "," + c + "] plan="
                        + DspDebugger.attach(sd).analyzePlan();
                assertEquals(expected[row][col], actual.getFloat(row, col), 0.0f, message);
            }
        }
    }

    /** Executes through the binding and copies its output before releasing native use. */
    private static void assertBoundOutput(SameDiff sd, NativeExecutionBinding binding, int step, String context) {
        NativeOps ops = binding.getBackendOwner().nativeOps();
        Pointer stream = ops.dspGetExecutionStream(binding.getPlanHandle());
        long[] shape = {ROWS, COLS};
        binding.beginNativeUse();
        try {
            int status = ops.executeDynamicShapePlan(binding.getPlanHandle(), binding.getContextHandle(), stream);
            ops.streamSynchronize(stream);
            assertEquals(0, ops.lastErrorCode(), "native execution completion: " + ops.lastErrorMessage());
            assertEquals(0, status, "bound native execution: " + ops.lastErrorMessage());
            OpaqueNDArray output = ops.getOutputArrayNative(binding.getContextHandle(), binding.findOutputIndex("out"));
            assertNotNull(output);
            assertTrue(!output.isNull());
            output.attachOwner(binding.getBackendOwner());
            assertArrayEquals(shape, Shape.shape(output.shapeInfo()));
            long length = output.length();
            Pointer special = ops.getOpaqueNDArraySpecialBuffer(output);
            Pointer primary = special == null || special.isNull() ? ops.getOpaqueNDArrayBuffer(output) : null;
            OpaqueDataBuffer source = ops.dbCreateExternalDataBuffer(length, DataType.FLOAT.toInt(), primary, special);
            assertNotNull(source);
            assertTrue(!source.isNull());
            try (INDArray copy = Nd4j.createUninitialized(DataType.FLOAT, shape,
                    Shape.stride(output.shapeInfo()), Shape.order(output.shapeInfo()))) {
                try {
                    ops.copyBuffer(copy.data().opaqueBuffer(), length, source, 0, 0);
                    Nd4j.getExecutioner().commit();
                    assertMatrix(sd, expectedOut(Graph.ELEMENTWISE, step), copy, context);
                } finally {
                    Nd4j.getExecutioner().commit();
                    ops.deleteDataBuffer(source);
                }
            }
            // output is borrowed from the plan's context; never close that wrapper here.
        } finally {
            ops.streamSynchronize(stream);
            assertEquals(0, ops.lastErrorCode(), "native completion before releasing binding: " + ops.lastErrorMessage());
            Nd4j.getExecutioner().commit();
            binding.completeNativeUse();
        }
    }
}
