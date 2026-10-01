/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import org.bytedeco.javacpp.BytePointer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.Arguments;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptFp8Linear;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Swish;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ModelOptNvfp4Linear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/** Independent numerical and native ABI contracts, shared by CPU and CUDA. */
@NativeTag
public class TestModelOptLinear extends BaseNd4jTestWithBackends {
    // The float types the native ops declare for X (ALL_FLOATS).
    private static final DataType[] ACTIVATION_TYPES = {DataType.FLOAT, DataType.DOUBLE, DataType.HALF, DataType.BFLOAT16};
    private static final float[] E2M1 = {0, .5f, 1, 1.5f, 2, 3, 4, 6};

    @Override
    public char ordering() { return 'c'; }

    public static Stream<Arguments> swiGluStorageConfigs() {
        return configs().flatMap(args -> Stream.of(DataType.HALF, DataType.BFLOAT16)
                .map(dtype -> Arguments.of(args.get()[0], dtype)));
    }

    @ParameterizedTest(name = "SiLU/SwiGLU {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testSwiGluStorageRounding(Nd4jBackend backend, DataType dtype) {
        float[] values = new float[1025];
        for (int i = 0; i < values.length; i++) values[i] = (i - 512) / 64.0f;
        try (INDArray source = Nd4j.createFromArray(values);
             INDArray gate = source.castTo(dtype);
             INDArray up = Nd4j.valueArrayOf(new long[]{values.length}, 1.234375, dtype);
             INDArray wide = gate.castTo(DataType.FLOAT);
             INDArray siluFloat = Nd4j.create(DataType.FLOAT, values.length);
             INDArray actualSilu = Nd4j.create(dtype, values.length);
             INDArray actualFused = Nd4j.create(dtype, values.length);
             INDArray actualHelper = Nd4j.create(dtype, values.length)) {
            Nd4j.exec(new Swish(wide, siluFloat));
            try (INDArray expectedSilu = siluFloat.castTo(dtype);
                 INDArray expected = expectedSilu.mul(up)) {
                Nd4j.exec(new Swish(gate, actualSilu));
                Nd4j.exec(new SwishMul(gate, up, actualFused));
                Nd4j.exec(DynamicCustomOp.builder("silu_and_mul")
                        .addInputs(gate, up).addOutputs(actualHelper).build());
                assertAll(dtype + " SiLU/SwiGLU rounding boundaries",
                        () -> assertArrayEquals(expectedSilu.data().asFloat(), actualSilu.data().asFloat(),
                                "SiLU must compute in FLOAT before its storage rounding"),
                        () -> assertArrayEquals(expected.data().asFloat(), actualFused.data().asFloat(),
                                "Fused SwiGLU must preserve the SiLU storage boundary"),
                        () -> assertArrayEquals(expected.data().asFloat(), actualHelper.data().asFloat(),
                                "silu_and_mul must use the same storage boundary"));
            }
        }
    }

    @ParameterizedTest(name = "SwiGLU views/aliases {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testSwiGluStorageViewsAndAliases(Nd4jBackend backend, DataType dtype) {
        try (INDArray gateParent = Nd4j.valueArrayOf(new long[]{4, 12}, 77, dtype);
             INDArray outputParent = Nd4j.create(dtype, new long[]{4, 12}, 'f').assign(99);
             INDArray gate = gateParent.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 11));
             INDArray output = outputParent.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 2, 12));
             INDArray raw = Nd4j.createFromArray(new float[][]{
                     {-8, -7.75f, -3.015625f, -1.125f, -0.015625f},
                     {0, 0.015625f, 0.333984375f, 1.125f, 7.75f}});
             INDArray up = Nd4j.create(dtype, new long[]{2, 5}, 'f').assign(1.234375)) {
            gate.assign(raw);
            try (INDArray wide = gate.castTo(DataType.FLOAT);
                 INDArray wideSilu = Nd4j.create(DataType.FLOAT, gate.shape())) {
                Nd4j.exec(new Swish(wide, wideSilu));
                try (INDArray expectedSilu = wideSilu.castTo(dtype)) {
                    Nd4j.exec(new Swish(gate, output));
                    assertCastValues(expectedSilu, output, "strided native SiLU");
                }
            }
            try (INDArray expected = swiGluReference(gate, up)) {
                Nd4j.exec(new SwishMul(gate, up, output));
                assertCastValues(expected, output, "offset/stepped inputs and F-order output");
                for (int row = 0; row < 4; row++) {
                    for (int col = 0; col < 12; col++) {
                        if (row >= 1 && row < 3 && col >= 2 && col % 2 == 0) continue;
                        assertEquals(99, outputParent.getDouble(row, col), 0.0, "output sentinel");
                    }
                }
                for (boolean aliasGate : new boolean[]{true, false}) {
                    try (INDArray g = gate.dup('c'); INDArray u = up.dup('f')) {
                        INDArray target = aliasGate ? g : u;
                        Nd4j.exec(new SwishMul(g, u, target));
                        assertCastValues(expected, target, "aliasGate=" + aliasGate);
                    }
                }
            }
            try (INDArray both = gate.dup('c'); INDArray expected = swiGluReference(both, both)) {
                Nd4j.exec(new SwishMul(both, both, both));
                assertCastValues(expected, both, "both operands alias output");
            }
            // Configurable swish_mul keeps x's shape; y may broadcast into it.
            try (INDArray row = Nd4j.valueArrayOf(new long[]{1, 5}, 1.234375, dtype);
                 INDArray expected = swiGluReference(gate, row)) {
                Nd4j.exec(new SwishMul(gate, row, output));
                assertCastValues(expected, output, "row broadcast");
            }
        }
    }

    @ParameterizedTest(name = "SwiGLU mixed multiply {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testSwiGluStorageMixedMultiply(Nd4jBackend backend, DataType dtype) {
        for (DataType upType : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            try (INDArray raw = Nd4j.createFromArray(-7.75f, -3.015625f, -1.125f, 0.333984375f);
                 INDArray gate = raw.castTo(dtype);
                 INDArray wideGate = gate.castTo(DataType.FLOAT);
                 INDArray siluFloat = Nd4j.create(DataType.FLOAT, 4);
                 INDArray up = Nd4j.valueArrayOf(new long[]{4}, 1.23456789, upType);
                 INDArray actual = Nd4j.create(dtype, 4)) {
                Nd4j.exec(new Swish(wideGate, siluFloat));
                try (INDArray siluStored = siluFloat.castTo(dtype);
                     INDArray siluCompute = siluStored.castTo(upType);
                     INDArray product = siluCompute.mul(up);
                     INDArray expected = product.castTo(dtype)) {
                    Nd4j.exec(new SwishMul(gate, up, actual));
                    assertCastValues(expected, actual, dtype + " gate / " + upType + " up");
                }
            }
        }
    }

    @ParameterizedTest(name = "SwiGLU optimized DSP {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testSwiGluStorageOptimizedDsp(Nd4jBackend backend, DataType dtype) {
        float[] values = new float[1025];
        for (int i = 0; i < values.length; i++) values[i] = (i - 512) / 64.0f;
        try (INDArray raw = Nd4j.createFromArray(values);
             INDArray gate = raw.castTo(dtype);
             INDArray up = Nd4j.valueArrayOf(new long[]{values.length}, 1.234375, dtype);
             SameDiff source = SameDiff.create()) {
            SDVariable g = source.placeHolder("gate", dtype, values.length);
            SDVariable u = source.placeHolder("up", dtype, values.length);
            source.nn().swish(g).mul("result", u);
            source.setOutputs("result");
            SameDiff optimized = GraphOptimizer.optimize(source, Collections.singletonList("result"));
            try {
                assertTrue(optimized.getOps().values().stream()
                        .anyMatch(op -> op.getOp() instanceof SwishMul), "must exercise SwiGLU fusion");
                optimized.setDspAutoCompileEnabled(true);
                optimized.setDspNativeAutoCompileEnabled(true);
                Map<String, INDArray> inputs = Map.of("gate", gate, "up", up);
                for (int repeat = 0; repeat < 12; repeat++) {
                    up.assign(repeat % 2 == 0 ? 1.234375 : -1.234375);
                    try (INDArray expected = swiGluReference(gate, up)) {
                        INDArray actual = optimized.outputSingle(inputs, "result");
                        assertCastValues(expected, actual, dtype + " DSP execution " + repeat);
                    }
                }
                DspPlanAssertions.assertPhaseReached(optimized, PlanPhase.SHAPES_FROZEN, "SwiGLU rounding");
                DspPlanAssertions.assertNoPhaseContractViolations(optimized, "SwiGLU rounding");
                DspPlanAssertions.assertNoSegmentFailures(optimized, "SwiGLU rounding");
            } finally {
                if (optimized != source) optimized.close();
            }
        }
    }

    /** Production full-attention gate, without exposing the sigmoid as a graph output. */
    @ParameterizedTest(name = "Attention sigmoid storage {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testAttentionSigmoidStorageOptimizedDsp(Nd4jBackend backend, DataType dtype) {
        attentionSigmoidStorage(dtype, false, false);
    }

    @ParameterizedTest(name = "Attention sigmoid reduction storage {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testAttentionSigmoidStorageReductionDsp(Nd4jBackend backend, DataType dtype) {
        boolean previousCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        try {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            attentionSigmoidStorage(dtype, true, false);
        } finally {
            Nd4j.getEnvironment().setTritonCompileAll(previousCompileAll);
        }
    }

    @ParameterizedTest(name = "Attention sigmoid permuted storage {0}/{1}")
    @MethodSource("swiGluStorageConfigs")
    public void testAttentionSigmoidStoragePermutedDsp(Nd4jBackend backend, DataType dtype) {
        boolean previousCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        boolean previousSectionFusion = Nd4j.getEnvironment().tritonSectionFusion();
        try {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            Nd4j.getEnvironment().setTritonSectionFusion(true);
            attentionSigmoidStorage(dtype, false, true);
        } finally {
            Nd4j.getEnvironment().setTritonSectionFusion(previousSectionFusion);
            Nd4j.getEnvironment().setTritonCompileAll(previousCompileAll);
        }
    }

    private void attentionSigmoidStorage(DataType dtype, boolean reduce, boolean permute) {
        for (int width : new int[]{1, 5}) {
            float[] values = new float[width * 256];
            for (int i = 0; i < values.length; i++) values[i] = ((i % 256) - 128) / 16.0f;
            try (INDArray raw = Nd4j.createFromArray(values);
                 INDArray stored = raw.castTo(dtype);
                 INDArray gate = stored.reshape(1, width, 256);
                 INDArray attention = Nd4j.valueArrayOf(new long[]{1, width, 256}, 1.234375, dtype);
                 INDArray sigmoid = Nd4j.create(dtype, gate.shape());
                 SameDiff source = SameDiff.create()) {
                SDVariable g = source.placeHolder("gate", dtype, 1, width, 256);
                SDVariable a = source.placeHolder("attention", dtype, 1, width, 256);
                SDVariable product = a.mul(source.nn().sigmoid(g));
                if (reduce) product.sum("result", true, 2);
                else if (permute) source.updateVariableNameAndReference(product.permute(0, 2, 1), "result");
                else source.updateVariableNameAndReference(product, "result");
                source.setOutputs("result");
                SameDiff optimized = GraphOptimizer.optimize(source, Collections.singletonList("result"));
                try {
                    optimized.setDspAutoCompileEnabled(true);
                    optimized.setDspNativeAutoCompileEnabled(true);
                    Map<String, INDArray> inputs = Map.of("gate", gate, "attention", attention);
                    for (int repeat = 0; repeat < 12; repeat++) {
                        attention.assign(repeat % 2 == 0 ? 1.234375 : -1.234375);
                        Nd4j.exec(new Sigmoid(gate, sigmoid));
                        try (INDArray materialized = attention.mul(sigmoid);
                             INDArray expected = reduce ? materialized.sum(true, 2)
                                     : permute ? materialized.permute(0, 2, 1) : materialized.dup()) {
                            INDArray actual = optimized.outputSingle(inputs, "result");
                            assertCastValues(expected, actual,
                                    dtype + " sigmoid storage W=" + width + " reduce=" + reduce
                                            + " execution=" + repeat);
                        }
                    }
                    DspPlanAssertions.assertPhaseReached(optimized, PlanPhase.SHAPES_FROZEN,
                            "attention gate rounding");
                    DspPlanAssertions.assertNoPhaseContractViolations(optimized, "attention gate rounding");
                    DspPlanAssertions.assertNoSegmentFailures(optimized, "attention gate rounding");
                } finally {
                    if (optimized != source) optimized.close();
                }
            }
        }
    }

    private static INDArray swiGluReference(INDArray gate, INDArray up) {
        // FLOAT SiLU, one gate-storage cast, then ordinary multiply: the
        // decomposed PyTorch storage boundary, independent of swish_mul.
        try (INDArray wide = gate.castTo(DataType.FLOAT);
             INDArray siluFloat = Nd4j.create(DataType.FLOAT, gate.shape())) {
            Nd4j.exec(new Swish(wide, siluFloat));
            try (INDArray stored = siluFloat.castTo(gate.dataType())) {
                return stored.mul(up);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOptimizerNarrowingRoundTripKeepsRounding(Nd4jBackend backend) {
        for (DataType middle : new DataType[]{DataType.BFLOAT16, DataType.HALF, DataType.INT}) {
            for (boolean exposeMiddle : new boolean[]{false, true}) {
                try (INDArray input = Nd4j.createFromArray(1.0013f, 2.0009f, -1.0005f, 0.3333f);
                     INDArray narrowed = input.castTo(middle);
                     INDArray expected = narrowed.castTo(DataType.FLOAT);
                     SameDiff source = SameDiff.create()) {
                    SDVariable x = source.placeHolder("x", DataType.FLOAT, 4);
                    SDVariable narrow = x.castTo("narrow", middle);
                    narrow.castTo("round_trip", DataType.FLOAT);
                    String[] outputs = exposeMiddle ? new String[]{"narrow", "round_trip"}
                            : new String[]{"round_trip"};
                    source.setOutputs(outputs);
                    SameDiff optimized = GraphOptimizer.optimize(source, Arrays.asList(outputs));
                    try {
                        Map<String, INDArray> result = optimized.output(Collections.singletonMap("x", input), outputs);
                        assertArrayEquals(expected.data().asFloat(), result.get("round_trip").data().asFloat());
                        if (exposeMiddle) {
                            assertEquals(middle, result.get("narrow").dataType());
                            try (INDArray actualMiddle = result.get("narrow").castTo(DataType.FLOAT)) {
                                assertArrayEquals(expected.data().asFloat(), actualMiddle.data().asFloat());
                            }
                        }
                    } finally {
                        if (optimized != source) optimized.close();
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLowPrecisionSigmoidUsesFloatIntermediates(Nd4jBackend backend) {
        float[] values = new float[1025];
        for (int i = 0; i < values.length; i++) values[i] = (i - 512) / 64.0f;
        for (DataType dtype : new DataType[]{DataType.HALF, DataType.BFLOAT16}) {
            try (INDArray source = Nd4j.createFromArray(values);
                 INDArray input = source.castTo(dtype);
                 INDArray wide = input.castTo(DataType.FLOAT);
                 INDArray reference = Nd4j.create(DataType.FLOAT, values.length);
                 INDArray actual = Nd4j.create(dtype, values.length)) {
                Nd4j.exec(new Sigmoid(wide, reference));
                Nd4j.exec(new Sigmoid(input, actual));
                try (INDArray expected = reference.castTo(dtype)) {
                    assertArrayEquals(expected.data().asFloat(), actual.data().asFloat(),
                            dtype + " sigmoid must round only at the final storage boundary");
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspNarrowingRoundTripKeepsRounding(Nd4jBackend backend) {
        for (DataType storage : new DataType[]{DataType.BFLOAT16, DataType.HALF}) {
            try (INDArray input = Nd4j.createFromArray(1.0013f, 2.0009f, -1.0005f, 0.3333f);
                 INDArray narrowed = input.castTo(storage);
                 INDArray expected = narrowed.castTo(DataType.FLOAT);
                 SameDiff sd = SameDiff.create()) {
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4);
                SDVariable roundTrip = x.castTo(storage).castTo(DataType.FLOAT);
                sd.updateVariableNameAndReference(roundTrip, "round_trip");
                for (int repeat = 0; repeat < 6; repeat++) {
                    INDArray actual = sd.outputSingle(Collections.singletonMap("x", input), "round_trip");
                    assertEquals(DataType.FLOAT, actual.dataType());
                    assertArrayEquals(expected.data().asFloat(), actual.data().asFloat(),
                            storage + " rounding lost at execution " + repeat);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspCastNarrowingOverflow(Nd4jBackend backend) {
        try (INDArray input = Nd4j.createFromArray(70000.0f, -70000.0f, 1.0013f)) {
            assertCastChain(input, DataType.HALF, DataType.FLOAT, false);
        }
        try (INDArray input = Nd4j.createFromArray(1.0000000001, -1.0000000001, 1e100)) {
            assertCastChain(input, DataType.FLOAT, DataType.DOUBLE, false);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspCastNonInverseAndCrossPrecision(Nd4jBackend backend) {
        try (INDArray input = Nd4j.createFromArray(1.0013f, -0.3333f, 70000.0f)) {
            assertCastChain(input, DataType.BFLOAT16, DataType.DOUBLE, false);
            assertCastChain(input, DataType.DOUBLE, DataType.HALF, false);
        }
        for (DataType source : new DataType[]{DataType.HALF, DataType.BFLOAT16}) {
            try (INDArray raw = Nd4j.createFromArray(1.00390625f, -0.3333f, 70000.0f);
                 INDArray input = raw.castTo(source)) {
                DataType other = source == DataType.HALF ? DataType.BFLOAT16 : DataType.HALF;
                assertCastChain(input, other, source, false);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspCastIntegerChains(Nd4jBackend backend) {
        try (INDArray input = Nd4j.createFromArray(257, -257, 127)) {
            assertCastChain(input, DataType.BYTE, DataType.INT, false);
            assertCastChain(input, DataType.SHORT, DataType.LONG, false);
        }
        try (INDArray input = Nd4j.createFromArray(16777217L, -16777217L, 1L)) {
            assertCastChain(input, DataType.FLOAT, DataType.LONG, false);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspCastLosslessWideningRoundTrips(Nd4jBackend backend) {
        for (DataType source : new DataType[]{DataType.HALF, DataType.BFLOAT16, DataType.FLOAT}) {
            for (DataType wide : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
                if (source == wide) continue;
                try (INDArray raw = Nd4j.createFromArray(1.0013, -0.3333, 0.0, 8192.0);
                     INDArray input = raw.castTo(source)) {
                    assertCastChain(input, wide, source, false);
                    // Publication is a second consumer: the widened dtype must survive.
                    assertCastChain(input, wide, source, true);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspCastSinkPreservesNarrowingAndPublishedOutput(Nd4jBackend backend) {
        for (DataType source : new DataType[]{DataType.DOUBLE, DataType.HALF}) {
            for (boolean publish : new boolean[]{false, true}) {
                // DOUBLE values lose bits when cast to FLOAT; multiplying by the
                // identity isolates the required conversion from accumulation.
                double magnitude = source == DataType.DOUBLE ? 16777217.0 : 17.0;
                try (INDArray raw = Nd4j.createFromArray(new double[][]{
                             {magnitude, 1.0000000001}, {-magnitude, 2.0}});
                     INDArray input = source == DataType.DOUBLE ? raw.dup() : raw.castTo(source);
                     INDArray expected = input.castTo(DataType.FLOAT);
                     INDArray weights = Nd4j.eye(2).castTo(DataType.FLOAT);
                     SameDiff sd = SameDiff.create()) {
                    SDVariable x = sd.placeHolder("x", source, 2, 2);
                    SDVariable knownSource = x.castTo(source);
                    SDVariable converted = knownSource.castTo("converted", DataType.FLOAT);
                    // An explicit cast producer also supplies dtype proof for the
                    // other operand without consulting external NDArray pointers.
                    SDVariable w = sd.constant("w", weights).castTo(DataType.FLOAT);
                    SDVariable result = converted.mmul(w);
                    sd.updateVariableNameAndReference(result, "result");
                    for (int repeat = 0; repeat < 6; repeat++) {
                        Map<String, INDArray> outputs = sd.output(Collections.singletonMap("x", input),
                                publish ? new String[]{"result", "converted"} : new String[]{"result"});
                        assertCastValues(expected, outputs.get("result"), "sink " + source + "/" + repeat);
                        if (publish) assertCastValues(expected, outputs.get("converted"), "published cast");
                    }
                }
            }
        }
    }

    private static void assertCastChain(INDArray input, DataType intermediate, DataType target,
                                       boolean publishIntermediate) {
        // Exercise both unknown external dtype and explicit producer dtype proof.
        for (boolean producer : new boolean[]{false, true}) {
            try (INDArray narrowed = input.castTo(intermediate);
                 INDArray expected = narrowed.castTo(target);
                 SameDiff sd = SameDiff.create()) {
                SDVariable x = sd.placeHolder("x", input.dataType(), input.shape());
                SDVariable source = producer ? x.castTo(input.dataType()) : x;
                SDVariable middle = source.castTo("middle", intermediate);
                SDVariable result = middle.castTo("result", target);
                for (int repeat = 0; repeat < 6; repeat++) {
                    Map<String, INDArray> outputs = sd.output(Collections.singletonMap("x", input),
                            publishIntermediate ? new String[]{result.name(), middle.name()}
                                    : new String[]{result.name()});
                    String label = input.dataType() + "->" + intermediate + "->" + target
                            + "/producer=" + producer + "/repeat=" + repeat;
                    assertCastValues(expected, outputs.get(result.name()), label);
                    if (publishIntermediate) assertCastValues(narrowed, outputs.get(middle.name()), label);
                }
            }
        }
    }

    private static void assertCastValues(INDArray expected, INDArray actual, String label) {
        assertEquals(expected.dataType(), actual.dataType(), label);
        assertArrayEquals(expected.shape(), actual.shape(), label);
        // Linear indexing follows each array's order. Compare the same logical
        // coordinates even when one operand is C-order and the other is an F view.
        try (INDArray expectedC = expected.dup('c'); INDArray actualC = actual.dup('c')) {
            for (long i = 0; i < expected.length(); i++) {
                if (expected.dataType().isIntType()) {
                    assertEquals(expectedC.getLong(i), actualC.getLong(i), label + "/element=" + i);
                } else {
                    assertEquals(expectedC.getDouble(i), actualC.getDouble(i), 0.0, label + "/element=" + i);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDenseBfloatProjectionAccumulation(Nd4jBackend backend) {
        // Dense auxiliary/MTP projections must not accumulate in BF16 storage precision.
        for (DataType dtype : new DataType[]{DataType.BFLOAT16, DataType.HALF}) {
            try (INDArray x = Nd4j.ones(dtype, 2, 8192);
                 INDArray w = Nd4j.ones(dtype, 8192, 2);
                 INDArray z = x.mmul(w)) {
                assertEquals(dtype, z.dataType());
                for (int row = 0; row < 2; row++) {
                    for (int col = 0; col < 2; col++) assertEquals(8192.0, z.getDouble(row, col), 0.0);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDenseBfloatGemmLayoutsAlphaBeta(Nd4jBackend backend) {
        for (char order : new char[]{'c', 'f'}) {
            for (boolean transA : new boolean[]{false, true}) {
                for (boolean transB : new boolean[]{false, true}) {
                    try (INDArray a = Nd4j.create(DataType.BFLOAT16, new long[]{2, 8192}, order);
                         INDArray b = Nd4j.create(DataType.BFLOAT16, new long[]{8192, 3}, order);
                         INDArray parent = Nd4j.create(DataType.BFLOAT16, new long[]{4, 8}, order)) {
                        a.assign(1);
                        b.assign(1);
                        a.getRow(1).assign(2);
                        b.getColumn(1).assign(-.5);
                        b.getColumn(2).assign(.25);
                        parent.assign(-99);
                        INDArray c = parent.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 7));
                        INDArray inputA = transA ? a.transpose() : a;
                        INDArray inputB = transB ? b.transpose() : b;
                        for (double beta : new double[]{0, .7}) {
                            c.assign(beta == 0 ? Double.NaN : 64);
                            Nd4j.getBlasWrapper().level3().gemm(inputA, inputB, c, transA, transB, 1.003, beta);
                            assertEquals(DataType.BFLOAT16, c.dataType());
                            for (int m = 0; m < 2; m++) {
                                for (int n = 0; n < 3; n++) {
                                    float weight = n == 0 ? 1 : n == 1 ? -.5f : .25f;
                                    float expected = (float) 1.003 * (8192 * (m + 1) * weight) + (float) beta * 64;
                                    assertEquals(round(expected, DataType.BFLOAT16), c.getFloat(m, n), 0,
                                            order + "/" + transA + "/" + transB + "/" + beta);
                                }
                            }
                            for (int m = 0; m < 4; m++)
                                for (int n = 0; n < 8; n++)
                                    if (m == 0 || m == 3 || n % 2 == 0 || n == 7)
                                        assertEquals(-99, parent.getDouble(m, n), 0);
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeDenseLowPrecisionAccumulation(Nd4jBackend backend) {
        // Rank-3 BF16 reaches the generic batched CUDA kernel; rank-2 BF16 uses
        // cuBLAS on SM80+ and the generic GEMM on older devices. CPU exercises
        // both the row accumulator and general F-order strided implementation.
        for (DataType dtype : new DataType[]{DataType.BFLOAT16, DataType.HALF}) {
            for (char order : new char[]{'c', 'f'}) {
                for (int rank : new int[]{2, 3}) {
                    long[] as = rank == 2 ? new long[]{2, 8192} : new long[]{2, 2, 8192};
                    long[] bs = rank == 2 ? new long[]{8192, 3} : new long[]{2, 8192, 3};
                    long[] cs = rank == 2 ? new long[]{2, 3} : new long[]{2, 2, 3};
                    try (INDArray a = Nd4j.create(dtype, as, order);
                         INDArray b = Nd4j.create(dtype, bs, order);
                         INDArray c = Nd4j.create(dtype, cs, order)) {
                        a.assign(1);
                        b.assign(1);
                        for (double beta : new double[]{0, .7}) {
                            c.assign(beta == 0 ? Double.NaN : 64);
                            Nd4j.exec(new Mmul(a, b, c, 1.003, beta, MMulTranspose.allFalse()));
                            assertEquals(dtype, c.dataType());
                            float expected = (float) round((float) 1.003 * 8192 + (float) beta * 64, dtype);
                            for (int i = 0; i < c.length(); i++) assertEquals(expected, c.getFloat(i), 0,
                                    dtype + "/" + order + "/rank=" + rank + "/beta=" + beta);
                        }
                    }
                }
            }
        }
        // Rank five selects the generic batched helper on both backends.
        // The stepped output exercises per-operand stride/offset addressing.
        for (char order : new char[]{'c', 'f'}) {
            try (INDArray a = Nd4j.create(DataType.BFLOAT16, new long[]{1, 1, 2, 2, 8192}, order);
                 INDArray b = Nd4j.create(DataType.BFLOAT16, new long[]{1, 1, 2, 8192, 3}, order);
                 INDArray parent = Nd4j.create(DataType.BFLOAT16, new long[]{1, 1, 2, 4, 8}, order)) {
                a.assign(1);
                b.assign(1);
                parent.assign(-99);
                INDArray c = parent.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 2, 7));
                c.assign(64);
                Nd4j.exec(new Mmul(a, b, c, 1.003, .7, MMulTranspose.allFalse()));
                float expected = (float) round((float) 1.003 * 8192 + .7f * 64, DataType.BFLOAT16);
                for (int batch = 0; batch < 2; batch++)
                    for (int m = 0; m < 4; m++)
                        for (int n = 0; n < 8; n++)
                            assertEquals(m > 0 && m < 3 && n % 2 == 1 && n < 7 ? expected : -99,
                                    parent.getFloat(0, 0, batch, m, n), 0);
            }
        }
        // Integer accumulators must not be converted to FP32/FP64 by the low-type fix.
        long exact = 9007199254740993L;
        try (INDArray a = Nd4j.createFromArray(exact, 0L, 0L, exact).reshape(2, 2);
             INDArray b = Nd4j.createFromArray(1L, 0L, 0L, 1L).reshape(2, 2);
             INDArray c = Nd4j.create(DataType.LONG, 2, 2)) {
            c.assign(10);
            Nd4j.exec(new Mmul(a, b, c, 1, 1, MMulTranspose.allFalse()));
            assertEquals(DataType.LONG, c.dataType());
            assertEquals(exact + 10, c.getLong(0, 0));
            assertEquals(exact + 10, c.getLong(1, 1));
            assertEquals(10, c.getLong(0, 1));
            assertEquals(10, c.getLong(1, 0));
        }
        // Rank-1 right operand selects the native GEMV, not Java level-2 BLAS.
        try (INDArray a = Nd4j.ones(DataType.BFLOAT16, 2, 8192);
             INDArray b = Nd4j.ones(DataType.BFLOAT16, 8192);
             INDArray c = Nd4j.create(DataType.BFLOAT16, 2)) {
            Nd4j.exec(new Mmul(a, b, c, 1, 0, MMulTranspose.allFalse()));
            assertEquals(8192, c.getFloat(0), 0);
            assertEquals(8192, c.getFloat(1), 0);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeDtypesLayoutsAndRepeatedOutputs(Nd4jBackend backend) {
        for (boolean nv : new boolean[]{true, false}) {
            for (DataType dtype : ACTIVATION_TYPES) {
                for (boolean floatOutput : new boolean[]{false, true}) {
                    for (int layout = 0; layout < 4; layout++) {
                        Fixture f = new Fixture(nv, dtype, layout);
                        DynamicCustomOp op = eager(nv, f.x, f.w, f.scale, f.second, floatOutput);
                        String context = nv + "/" + dtype + "/float=" + floatOutput + "/layout=" + layout;
                        INDArray allocated = Nd4j.exec(op)[0];
                        assertResult(f, allocated, floatOutput, context);

                        DataType outType = floatOutput ? DataType.FLOAT : dtype;
                        INDArray parent = Nd4j.createUninitialized(outType, new long[]{6, 8}, layout == 1 ? 'f' : 'c');
                        parent.assign(-99);
                        INDArray supplied = parent.get(NDArrayIndex.all(), NDArrayIndex.interval(1, 2, 7));
                        DynamicCustomOp repeated = eager(nv, f.x, f.w, f.scale, f.second, floatOutput);
                        repeated.addOutputArgument(supplied);
                        for (int pass = 0; pass < 3; pass++) {
                            if (pass > 0) f.x.assign(pass == 1 ? .25 : -.5);
                            Nd4j.exec(repeated);
                            assertResult(f, supplied, floatOutput, context + "/pass=" + pass);
                            // The op must fully overwrite its view, and never touch its neighbours.
                            for (int row = 0; row < 6; row++) {
                                for (int col : new int[]{0, 2, 4, 6, 7})
                                    assertEquals(-99, parent.getDouble(row, col), 0, context);
                            }
                        }
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRankThreeAndVector(Nd4jBackend backend) {
        for (boolean nv : new boolean[]{true, false}) {
            for (DataType dtype : ACTIVATION_TYPES) {
                Fixture f = new Fixture(nv, dtype, 0);
                INDArray rank3 = f.x.reshape(2, 3, 32).permute(1, 0, 2);
                INDArray z = Nd4j.exec(eager(nv, rank3, f.w, f.scale, f.second, true))[0];
                assertArrayEquals(new long[]{3, 2, 3}, z.shape());
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        for (int n = 0; n < 3; n++)
                            assertEquals(reference(f, j * 3 + i, n), z.getDouble(i, j, n), 2e-5);
                INDArray vector = f.x.getRow(2).reshape(32);
                INDArray v = Nd4j.exec(eager(nv, vector, f.w, f.scale, f.second, false))[0];
                assertArrayEquals(new long[]{3}, v.shape());
                assertEquals(dtype, v.dataType());
                for (int n = 0; n < 3; n++)
                    assertEquals(round(reference(f, 2, n), dtype), v.getDouble(n), 2e-5);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEveryNibbleAndBfloatWeightRounding(Nd4jBackend backend) {
        // One-hot activations expose each nibble individually, including negative zero.
        byte[] packed = new byte[8];
        for (int i = 0; i < 8; i++) packed[i] = (byte) ((2 * i) | ((2 * i + 1) << 4));
        for (DataType dtype : ACTIVATION_TYPES) {
            INDArray x = Nd4j.eye(16).castTo(dtype);
            INDArray w = raw(DataType.UBYTE, packed, 1, 8);
            INDArray block = raw(DataType.FLOAT8, new byte[]{0x39}, 1, 1); // 1.125
            INDArray global = scalar(0.1003f);
            INDArray z = Nd4j.exec(new ModelOptNvfp4Linear(x, w, block, global, true))[0];
            for (int k = 0; k < 16; k++) {
                float scale = 1.125f * .1003f;
                float weight = signedNibble(k) * scale;
                assertEquals(round(weight, dtype), z.getFloat(k, 0), 0, dtype + "/nibble=" + k);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFp8RneSubnormalsAndSaturation(Nd4jBackend backend) {
        float midpoint = Math.scalb(1.0f, -10);
        float[] values = {0, Math.nextDown(midpoint), midpoint, Math.nextUp(midpoint),
                -Math.nextDown(midpoint), -midpoint, -Math.nextUp(midpoint),
                3 * midpoint, -3 * midpoint, 1.0625f, 1.1875f, -1.0625f, -1.1875f,
                447, 449, -449, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
        INDArray x = Nd4j.create(values, new long[]{values.length, 1}, DataType.FLOAT);
        INDArray w = raw(DataType.FLOAT8, new byte[]{0x38}, 1, 1);
        INDArray z = Nd4j.exec(new ModelOptFp8Linear(x, w, scalar(1), scalar(1), true))[0];
        for (int i = 0; i < values.length; i++)
            assertEquals(quantizeE4m3(values[i]), z.getFloat(i, 0), 0, "activation=" + values[i]);
        assertEquals(Math.scalb(1.0f, -9), z.getFloat(3, 0), 0);
        assertEquals(-Math.scalb(1.0f, -9), z.getFloat(6, 0), 0);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRawFp8WeightEncodings(Nd4jBackend backend) {
        byte[] bytes = new byte[254];
        for (int i = 0; i < 127; i++) {
            bytes[i] = (byte) i;
            bytes[127 + i] = (byte) (128 | i);
        }
        INDArray w = raw(DataType.FLOAT8, bytes, 254, 1);
        INDArray x = Nd4j.ones(DataType.FLOAT, 1, 1);
        INDArray z = Nd4j.exec(new ModelOptFp8Linear(x, w, scalar(.7f), scalar(.3f), true))[0];
        float activation = quantizeE4m3(1 / .3f) * .3f;
        for (int n = 0; n < bytes.length; n++)
            assertEquals(activation * (decodeE4m3(bytes[n] & 255) * .7f), z.getFloat(0, n), 0,
                    "raw weight encoding=" + (bytes[n] & 255));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFloatAccumulator(Nd4jBackend backend) {
        int k = 8192;
        for (boolean nv : new boolean[]{true, false}) {
            byte[] weights = new byte[nv ? k / 2 : k];
            Arrays.fill(weights, (byte) (nv ? 0x11 : 0x30)); // both encode .5
            byte[] scales = new byte[k / 16];
            Arrays.fill(scales, (byte) 0x38);
            INDArray w = raw(nv ? DataType.UBYTE : DataType.FLOAT8, weights, 1, weights.length);
            INDArray s = nv ? raw(DataType.FLOAT8, scales, 1, scales.length) : scalar(1);
            for (DataType dtype : ACTIVATION_TYPES) {
                INDArray x = Nd4j.ones(dtype, 1, k);
                for (boolean fp32 : new boolean[]{false, true}) {
                    INDArray z = Nd4j.exec(eager(nv, x, w, s, scalar(1), fp32))[0];
                    assertEquals(fp32 ? DataType.FLOAT : dtype, z.dataType());
                    assertEquals(4096, z.getFloat(0), 0, nv + "/" + dtype);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyAndZeroInnerDimension(Nd4jBackend backend) {
        for (boolean nv : new boolean[]{true, false}) {
            for (DataType dtype : ACTIVATION_TYPES) {
                for (boolean floatOutput : new boolean[]{false, true}) {
                    for (long[] shape : new long[][]{{0, 32}, {2, 0}, {2, 32}}) {
                        int k = (int) shape[1];
                        int n = shape[0] == 2 && k == 32 ? 0 : 3;
                        INDArray x = Nd4j.create(dtype, shape);
                        INDArray w = Nd4j.create(nv ? DataType.UBYTE : DataType.FLOAT8, n, nv ? k / 2 : k);
                        byte[] blocks = new byte[n * (k / 16)];
                        Arrays.fill(blocks, (byte) 0x38);
                        INDArray scale = nv ? raw(DataType.FLOAT8, blocks, n, k / 16) : scalar(1);
                        INDArray z = Nd4j.exec(eager(nv, x, w, scale, scalar(1), floatOutput))[0];
                        assertArrayEquals(new long[]{shape[0], n}, z.shape());
                        assertEquals(floatOutput ? DataType.FLOAT : dtype, z.dataType());
                        for (int i = 0; i < z.length(); i++) assertEquals(0, z.getDouble(i), 0);
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNativeMalformedContracts(Nd4jBackend backend) {
        for (boolean nv : new boolean[]{true, false}) {
            Fixture f = new Fixture(nv, DataType.FLOAT, 0);
            INDArray[] good = {f.x, f.w, f.scale, f.second};
            reject(nv, good, new long[]{2}, null);
            reject(nv, good, new long[]{}, null);
            reject(nv, good, new long[]{0, 1}, null);
            reject(nv, Arrays.copyOf(good, 3), new long[]{0}, null);
            reject(nv, good, new long[]{0}, Nd4j.create(DataType.FLOAT, 6, 4));
            reject(nv, good, new long[]{0}, Nd4j.create(DataType.HALF, 6, 3));
            for (int input = 0; input < 4; input++) {
                // X admits every float type; each storage and scale input has one.
                INDArray[] wrong = good.clone();
                wrong[input] = good[input].castTo(input == 0 ? DataType.INT : DataType.DOUBLE);
                reject(nv, wrong, new long[]{0}, null);
            }
            INDArray[] wrongStorage = good.clone();
            wrongStorage[1] = raw(nv ? DataType.BYTE : DataType.FLOAT8_E5M2, f.bytes, 3, nv ? 16 : 32);
            reject(nv, wrongStorage, new long[]{0}, null);
            if (nv) {
                INDArray[] wrongBlocks = good.clone();
                wrongBlocks[2] = raw(DataType.FLOAT8_E5M2, f.blocks, 3, 2);
                reject(true, wrongBlocks, new long[]{0}, null);
            }
            INDArray[] wrong = good.clone();
            wrong[0] = scalar(1);
            reject(nv, wrong, new long[]{0}, null);
            wrong = good.clone();
            wrong[0] = Nd4j.create(DataType.FLOAT, 6, 31);
            reject(nv, wrong, new long[]{0}, null);
            wrong = good.clone();
            wrong[1] = f.w.reshape(f.w.length());
            reject(nv, wrong, new long[]{0}, null);
            wrong = good.clone();
            wrong[2] = nv ? raw(DataType.FLOAT8, new byte[]{0x38}, 1, 1) : scalar(1).reshape(1);
            reject(nv, wrong, new long[]{0}, null);
            wrong = good.clone();
            wrong[3] = scalar(1).reshape(1);
            reject(nv, wrong, new long[]{0}, null);
            // Correct output shape, but backed by X's storage: aliasing is prohibited.
            INDArray alias = f.x.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3));
            reject(nv, good, new long[]{0}, alias);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidHostCurrentScales(Nd4jBackend backend) {
        // Only fresh HOST-current invalid scales: never launch a device assertion.
        for (boolean nv : new boolean[]{true, false}) {
            Fixture f = new Fixture(nv, DataType.FLOAT, 0);
            for (float bad : new float[]{0, -1, Float.NaN, Float.POSITIVE_INFINITY}) {
                reject(nv, new INDArray[]{f.x, f.w, f.scale, scalar(bad)}, new long[]{0}, null);
                if (!nv)
                    reject(false, new INDArray[]{f.x, f.w, scalar(bad), scalar(1)}, new long[]{0}, null);
            }
            if (nv) {
                for (int bad : new int[]{0, 0xb8, 0x7f}) {
                    byte[] blocks = new byte[6];
                    Arrays.fill(blocks, (byte) 0x38);
                    blocks[5] = (byte) bad;
                    reject(true, new INDArray[]{f.x, f.w, raw(DataType.FLOAT8, blocks, 3, 2), scalar(1)},
                            new long[]{0}, null);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSameDiffSerialization(Nd4jBackend backend) throws IOException {
        for (boolean nv : new boolean[]{true, false}) {
            for (DataType dtype : ACTIVATION_TYPES) {
                for (boolean floatOutput : new boolean[]{false, true}) {
                    Fixture f = new Fixture(nv, dtype, 0);
                    SameDiff sd = SameDiff.create();
                    SDVariable x = sd.placeHolder("x", dtype, -1, 32);
                    SDVariable w = sd.constant("w", f.w);
                    SDVariable scale = sd.constant("scale", f.scale);
                    SDVariable second = sd.constant("second", f.second);
                    DynamicCustomOp op = nv ? new ModelOptNvfp4Linear(sd, x, w, scale, second, floatOutput)
                            : new ModelOptFp8Linear(sd, x, w, scale, second, floatOutput);
                    String output = op.outputVariable().name();
                    assertEquals(floatOutput ? DataType.FLOAT : dtype, sd.getVariable(output).dataType());
                    SameDiff restored = SameDiff.fromFlatBuffers(sd.asFlatBuffers(true));
                    DynamicCustomOp restoredOp = (DynamicCustomOp) restored.getVariableOutputOp(output);
                    assertEquals(op.getClass(), restoredOp.getClass());
                    assertArrayEquals(new long[]{floatOutput ? 1 : 0}, restoredOp.iArgs());
                    assertEquals(op.opName(), restoredOp.opName());
                    for (int pass = 0; pass < 2; pass++) {
                        if (pass == 1) f.x.assign(-.25);
                        assertResult(f, sd.outputSingle(Collections.singletonMap("x", f.x), output), floatOutput, "original");
                        assertResult(f, restored.outputSingle(Collections.singletonMap("x", f.x), output), floatOutput, "restored");
                    }
                }
            }
        }
    }

    private static DynamicCustomOp eager(boolean nv, INDArray x, INDArray w, INDArray s, INDArray t, boolean fp32) {
        return nv ? new ModelOptNvfp4Linear(x, w, s, t, fp32) : new ModelOptFp8Linear(x, w, s, t, fp32);
    }

    private static void reject(boolean nv, INDArray[] inputs, long[] args, INDArray output) {
        // Deliberately bypass Java dtype inference to exercise the native validator.
        DynamicCustomOp op = new DynamicCustomOp(nv ? "modelopt_nvfp4_linear" : "modelopt_fp8_linear",
                inputs, output == null ? null : new INDArray[]{output}, Collections.emptyList(), args);
        assertThrows(RuntimeException.class, () -> Nd4j.exec(op));
    }

    /**
     * CUDA tiled-kernel arithmetic contract, bit for bit: every output is 32
     * lane-strided fused multiply-add chains in the activation's aggregate type
     * (FP32, or FP64 for DOUBLE; lane l owns the 8-element words l, l+32, ...,
     * ascending), then the ascending-offset shuffle fold. The kernel processes
     * rows in register passes of 8 and dequantizes each weight word once per
     * pass, so row counts inside, at and across the pass boundary (1, 5, 8, 9,
     * 17) and a prefill-length input (128) must all reproduce the per-row
     * contract exactly, for K spanning one to many words per lane.
     *
     * Every N here is odd, so no case is admitted to a tensor-core path (FP8
     * needs N * sizeof(output) to be a multiple of 16 bytes, NVFP4 needs whole
     * 8-column tiles): all stay on the native tiled kernel. The tensor-core
     * paths are covered by testFp8TensorCorePathMatchesReference and
     * testNvfp4TensorCorePathMatchesReference.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTiledRowPassesMatchLaneContract(Nd4jBackend backend) {
        assumeFalse(Nd4j.getEnvironment().isCPU(), "lane-split fold is the CUDA tiled kernel's contract");
        int[][] shapes = {{1, 2048, 37}, {5, 5120, 33}, {8, 256, 65}, {9, 512, 17}, {17, 1024, 9}, {128, 256, 11}};
        java.util.Random random = new java.util.Random(20260927L);
        for (boolean nv : new boolean[]{true, false}) {
            for (DataType dtype : ACTIVATION_TYPES) {
                DataType acc = aggregate(dtype);
                for (int[] shape : shapes) {
                    int rows = shape[0], k = shape[1], n = shape[2];
                    float[] values = new float[rows * k];
                    for (int i = 0; i < values.length; i++) values[i] = (random.nextInt(4097) - 2048) / 1024.0f;
                    INDArray x = Nd4j.create(values, new long[]{rows, k}, DataType.FLOAT).castTo(dtype);
                    // Reference activations are the dtype-rounded values the kernel reads.
                    double[] xs = x.castTo(DataType.DOUBLE).data().asDouble();
                    byte[] bytes = new byte[n * (nv ? k / 2 : k)];
                    for (int i = 0; i < bytes.length; i++)
                        bytes[i] = nv ? (byte) random.nextInt(256)
                                : (byte) ((random.nextInt(0x60) + 0x10) | (random.nextBoolean() ? 128 : 0));
                    byte[] blocks = new byte[nv ? n * (k / 16) : 0];
                    for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
                    float global = nv ? .1003f : .3f;
                    float weightScale = .7f;
                    INDArray w = raw(nv ? DataType.UBYTE : DataType.FLOAT8, bytes, n, nv ? k / 2 : k);
                    INDArray scale = nv ? raw(DataType.FLOAT8, blocks, n, k / 16) : scalar(weightScale);
                    double[] a = xs;
                    if (!nv) {
                        a = new double[xs.length];
                        for (int i = 0; i < xs.length; i++) a[i] = fp8Activation(acc, xs[i], global);
                    }
                    for (boolean floatOutput : new boolean[]{false, true}) {
                        INDArray z = nv ? Nd4j.exec(new ModelOptNvfp4Linear(x, w, scale, scalar(global), floatOutput))[0]
                                : Nd4j.exec(new ModelOptFp8Linear(x, w, scale, scalar(global), floatOutput))[0];
                        DataType outType = floatOutput ? DataType.FLOAT : dtype;
                        assertEquals(outType, z.dataType());
                        double[] actual = z.castTo(DataType.DOUBLE).data().asDouble();
                        for (int row = 0; row < rows; row++) {
                            for (int col = 0; col < n; col++) {
                                double[] lanes = new double[32];
                                for (int lane = 0; lane < 32; lane++) {
                                    double sum = 0;
                                    for (int word = lane; word < k / 8; word += 32) {
                                        for (int j = 0; j < 8; j++) {
                                            int kk = word * 8 + j;
                                            double b;
                                            if (nv) {
                                                int bits = bytes[col * (k / 2) + kk / 2] & 255;
                                                float blockScale = decodeE4m3(blocks[col * (k / 16) + kk / 16] & 255) * global;
                                                b = round(signedNibble((bits >>> (4 * (kk & 1))) & 15) * blockScale, dtype);
                                            } else {
                                                b = fp8Weight(acc, bytes[col * k + kk] & 255, weightScale);
                                            }
                                            sum = fma(acc, a[row * k + kk], b, sum);
                                        }
                                    }
                                    lanes[lane] = sum;
                                }
                                for (int offset = 1; offset < 32; offset <<= 1) {
                                    double[] next = lanes.clone();
                                    for (int lane = 0; lane + offset < 32; lane++)
                                        next[lane] = add(acc, lanes[lane], lanes[lane + offset]);
                                    lanes = next;
                                }
                                double expected = round(lanes[0], outType);
                                double got = actual[row * n + col];
                                assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(got),
                                        (nv ? "nvfp4" : "fp8") + "/" + dtype + "/floatOutput=" + floatOutput
                                                + "/rows=" + rows + "/k=" + k + "/n=" + n
                                                + " at [" + row + "," + col + "]: expected " + expected + " got " + got);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * FP8 linears run on tensor cores where their shapes allow. Decode-class
     * calls (at most 16 rows) whose shapes the weight-only GEMM admits (K a
     * multiple of 32, whole 8-column tiles) run on mma.m16n8k16 with each
     * activation quantized to E4M3 in registers exactly as the native kernels
     * quantize it. Longer calls, and shapes the weight-only GEMM does not
     * admit, run as one scaled FP8 cuBLASLt GEMM when they meet its alignment
     * rules (K and N * sizeof(output) multiples of 16 bytes). Both multiply the
     * codes exactly and scale each output's FP32 sum once by
     * inputScale * weightScale; only the FP32 accumulation order differs from
     * the native kernels, so the result is checked against an exact double
     * reference with an accumulation-error bound (K * 2^-24 * sum|terms|) plus
     * the output dtype's rounding. Both admit only activations whose aggregate
     * type is FP32 (FLOAT, HALF, BFLOAT16); DOUBLE activations take the native
     * FP64 kernels and must meet the same bound. Covers decode (1, 5), one full
     * MMA tile (16), the first call past the decode class (17), prefill (128),
     * a K shorter than one chunk (96), a K split across blocks (8192 over 64
     * columns) and a K the weight-only GEMM does not admit (16). On CUDA the
     * path is read from the BACKEND diagnostics; no shape here has enough units
     * to stage its weights on any device, so each takes the direct kernel
     * unless -Dnd4j.weightOnly.staging=bulk forces staging.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFp8TensorCorePathMatchesReference(Nd4jBackend backend) {
        int[][] shapes = {{1, 512, 64}, {5, 5120, 48}, {16, 512, 40}, {17, 1024, 24}, {128, 256, 1040},
                {2, 96, 8}, {3, 8192, 64}, {5, 16, 16}};
        java.util.Random random = new java.util.Random(20260928L);
        float inputScale = .3f, weightScale = .7f;
        boolean tensorCores = !Nd4j.getEnvironment().isCPU() && Nd4j.getNativeOps()
                .getDeviceMajor(Nd4j.getAffinityManager().getDeviceForCurrentThread()) >= 8;
        // Set from -Dnd4j.weightOnly.staging; "bulk" stages every shape that fits.
        boolean forcedBulk = "bulk".equals(System.getenv("SD_WEIGHT_ONLY_STAGING"));
        if (tensorCores) DspDiagnostics.setCategories(DspDiagnostics.BACKEND);
        try {
            for (DataType dtype : ACTIVATION_TYPES) {
                for (int[] shape : shapes) {
                    int rows = shape[0], k = shape[1], n = shape[2];
                    float[] values = new float[rows * k];
                    for (int i = 0; i < values.length; i++) values[i] = (random.nextInt(4097) - 2048) / 1024.0f;
                    INDArray x = Nd4j.create(values, new long[]{rows, k}, DataType.FLOAT).castTo(dtype);
                    double[] xs = x.castTo(DataType.DOUBLE).data().asDouble();
                    byte[] bytes = new byte[n * k];
                    for (int i = 0; i < bytes.length; i++)
                        bytes[i] = (byte) ((random.nextInt(0x60) + 0x10) | (random.nextBoolean() ? 128 : 0));
                    INDArray w = raw(DataType.FLOAT8, bytes, n, k);
                    DataType acc = aggregate(dtype);
                    double[] a = new double[xs.length];
                    for (int i = 0; i < xs.length; i++) a[i] = fp8Activation(acc, xs[i], inputScale);
                    boolean weightOnly = tensorCores && dtype != DataType.DOUBLE && rows <= 16 && k % 32 == 0
                            && n % 8 == 0;
                    for (boolean floatOutput : new boolean[]{false, true}) {
                        String context = dtype + "/floatOutput=" + floatOutput + "/rows=" + rows + "/k=" + k
                                + "/n=" + n;
                        DspDiagnostics.clear();
                        INDArray z = Nd4j.exec(new ModelOptFp8Linear(x, w, scalar(weightScale), scalar(inputScale),
                                floatOutput))[0];
                        if (tensorCores) {
                            assertEquals(weightOnly, "weight-only-mma".equals(modelOptPath(context)), context
                                    + ": decode-class calls the weight-only GEMM admits must take it, others not");
                            if (weightOnly && !forcedBulk)
                                assertEquals("direct", weightOnlyKernel(context),
                                        context + ": too few units to stage on any device");
                        }
                        DataType outType = floatOutput ? DataType.FLOAT : dtype;
                        assertEquals(outType, z.dataType());
                        double outputEpsilon =
                                outType == DataType.HALF ? 0x1p-11 : outType == DataType.BFLOAT16 ? 0x1p-8 : 0;
                        double[] actual = z.castTo(DataType.DOUBLE).data().asDouble();
                        for (int row = 0; row < rows; row++) {
                            for (int col = 0; col < n; col++) {
                                double exact = 0, magnitude = 0;
                                for (int kk = 0; kk < k; kk++) {
                                    double term =
                                            a[row * k + kk] * fp8Weight(acc, bytes[col * k + kk] & 255, weightScale);
                                    exact += term;
                                    magnitude += Math.abs(term);
                                }
                                double tolerance = k * 0x1p-24 * magnitude + outputEpsilon * Math.abs(exact) + 1e-30;
                                double got = actual[row * n + col];
                                assertTrue(Math.abs(got - exact) <= tolerance,
                                        dtype + "/floatOutput=" + floatOutput + "/rows=" + rows + "/k=" + k + "/n=" + n
                                                + " at [" + row + "," + col + "]: exact " + exact + " got " + got
                                                + " tolerance " + tolerance);
                            }
                        }
                    }
                }
            }
        } finally {
            if (tensorCores) DspDiagnostics.setCategories(DspDiagnostics.NONE);
        }
    }

    /**
     * NVFP4 linears with BF16/HALF activations whose shapes the weight-only
     * tensor-core GEMM admits (K a multiple of 32, whole 8-column tiles) run on
     * mma.m16n8k16 with the weights dequantized in registers exactly as the
     * native kernels do; every product is exact in FP32 and only the FP32
     * accumulation order differs. Checked against an exact double reference with
     * an accumulation-error bound (K * 2^-24 * sum|terms|) plus the output
     * dtype's rounding. FLOAT and DOUBLE activations take the native kernels and
     * must meet the same bound. Covers decode (1, 5), one full MMA tile (16), a
     * partial second tile (17), prefill (128), and K ranges shorter than, equal
     * to and longer than one split-K chunk per warp.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNvfp4TensorCorePathMatchesReference(Nd4jBackend backend) {
        int[][] shapes = {{1, 32, 8}, {5, 5120, 64}, {16, 512, 40}, {17, 1024, 24}, {128, 256, 136}, {5, 17408, 8}};
        java.util.Random random = new java.util.Random(20260930L);
        float global = .1003f;
        for (DataType dtype : ACTIVATION_TYPES) {
            for (int[] shape : shapes) {
                int rows = shape[0], k = shape[1], n = shape[2];
                float[] values = new float[rows * k];
                for (int i = 0; i < values.length; i++) values[i] = (random.nextInt(4097) - 2048) / 1024.0f;
                INDArray x = Nd4j.create(values, new long[]{rows, k}, DataType.FLOAT).castTo(dtype);
                double[] xs = x.castTo(DataType.DOUBLE).data().asDouble();
                byte[] bytes = new byte[n * k / 2];
                for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) random.nextInt(256);
                byte[] blocks = new byte[n * (k / 16)];
                for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
                INDArray w = raw(DataType.UBYTE, bytes, n, k / 2);
                INDArray scale = raw(DataType.FLOAT8, blocks, n, k / 16);
                for (boolean floatOutput : new boolean[]{false, true}) {
                    INDArray z = Nd4j.exec(new ModelOptNvfp4Linear(x, w, scale, scalar(global), floatOutput))[0];
                    DataType outType = floatOutput ? DataType.FLOAT : dtype;
                    assertEquals(outType, z.dataType());
                    double outputEpsilon = outType == DataType.HALF ? 0x1p-11 : outType == DataType.BFLOAT16 ? 0x1p-8 : 0;
                    double[] actual = z.castTo(DataType.DOUBLE).data().asDouble();
                    for (int row = 0; row < rows; row++) {
                        for (int col = 0; col < n; col++) {
                            double exact = 0, magnitude = 0;
                            for (int kk = 0; kk < k; kk++) {
                                int bits = bytes[col * (k / 2) + kk / 2] & 255;
                                float blockScale = decodeE4m3(blocks[col * (k / 16) + kk / 16] & 255) * global;
                                double weight = round(signedNibble((bits >>> (4 * (kk & 1))) & 15) * blockScale, dtype);
                                double term = xs[row * k + kk] * weight;
                                exact += term;
                                magnitude += Math.abs(term);
                            }
                            double tolerance = k * 0x1p-24 * magnitude + outputEpsilon * Math.abs(exact) + 1e-30;
                            double got = actual[row * n + col];
                            assertTrue(Math.abs(got - exact) <= tolerance,
                                    dtype + "/floatOutput=" + floatOutput + "/rows=" + rows + "/k=" + k + "/n=" + n
                                            + " at [" + row + "," + col + "]: exact " + exact + " got " + got
                                            + " tolerance " + tolerance);
                        }
                    }
                }
            }
        }
    }

    /**
     * Row invariance across decode widths: an output row must be bit-identical
     * whatever number of rows (up to the decode class, 16) the call carries.
     * Speculative decoding computes the same token position in calls of
     * different widths (verify windows, reruns, greedy), and its MTP-vs-greedy
     * losslessness gate requires identical bits there. Prefill-length calls
     * form their own class: both decoding modes prefill with the same width.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRowResultsIndependentOfRowCount(Nd4jBackend backend) {
        java.util.Random random = new java.util.Random(20260929L);
        int k = 5120;
        for (boolean nv : new boolean[]{true, false})
        for (int n : new int[]{64, 6144})
        for (boolean floatOutput : new boolean[]{true, false}) {
            byte[] bytes = new byte[n * (nv ? k / 2 : k)];
            for (int i = 0; i < bytes.length; i++)
                bytes[i] = nv ? (byte) random.nextInt(256)
                        : (byte) ((random.nextInt(0x60) + 0x10) | (random.nextBoolean() ? 128 : 0));
            byte[] blocks = new byte[nv ? n * (k / 16) : 0];
            for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
            INDArray w = raw(nv ? DataType.UBYTE : DataType.FLOAT8, bytes, n, nv ? k / 2 : k);
            INDArray scale = nv ? raw(DataType.FLOAT8, blocks, n, k / 16) : scalar(.7f);
            INDArray global = scalar(nv ? .1003f : .3f);
            int widest = 16;
            float[] values = new float[widest * k];
            for (int i = 0; i < values.length; i++) values[i] = (random.nextInt(4097) - 2048) / 1024.0f;
            INDArray all = Nd4j.create(values, new long[]{widest, k}, DataType.FLOAT).castTo(DataType.BFLOAT16);
            INDArray reference = null;
            for (int rows : new int[]{1, 2, 3, 5, 8, 16}) {
                INDArray x = all.get(NDArrayIndex.interval(0, rows), NDArrayIndex.all()).dup('c');
                INDArray z = nv ? Nd4j.exec(new ModelOptNvfp4Linear(x, w, scale, global, floatOutput))[0]
                        : Nd4j.exec(new ModelOptFp8Linear(x, w, scale, global, floatOutput))[0];
                INDArray firstRow = z.getRow(0).castTo(DataType.FLOAT).dup();
                if (reference == null) {
                    reference = firstRow;
                    continue;
                }
                for (int col = 0; col < n; col++)
                    assertEquals(Float.floatToIntBits(reference.getFloat(col)), Float.floatToIntBits(firstRow.getFloat(col)),
                            (nv ? "nvfp4" : "fp8") + "/n=" + n + "/floatOutput=" + floatOutput + ": row 0 column "
                                    + col + " differs between 1 and " + rows + " rows: " + reference.getFloat(col)
                                    + " vs " + firstRow.getFloat(col));
            }
        }
    }

    /**
     * The weight-only GEMM has two kernels that share their work units, K ranges,
     * reduction and split combine: the direct kernel loads each lane's weights
     * from global memory, and the bulk kernel (sm_90+) stages a unit's weight
     * tile in shared memory with bulk copies. Their results must be identical bit
     * for bit. Block scales 2 bytes into their buffer are admitted (the kernels
     * read scale pairs) but cannot be bulk-copied, so they take the direct
     * kernel. The same scales 16-byte aligned take the bulk kernel where it was
     * measured faster: a unit's weight tile is at least 18 KB (K = 4096 for
     * NVFP4), the shape has at least four waves of units, and the call has at
     * most 8 rows (the MMA's lower half). BULK shapes meet that on every sm_90+
     * device with up to 256 SMs (at least 4096 units, tiles of 24 KB or less);
     * DIRECT shapes fail it on every device; EITHER shapes depend on the SM
     * count and only have to match. The launched kernel is read from the
     * BACKEND diagnostics. Covers the Qwen3.6-27B projections' K ranges at
     * decode widths 1 and 5 (the gate and up projections' K = 5120, at twice
     * their 17408 columns so that every device has four waves; the down
     * projection, K = 17408 over 5120 columns, split across four blocks on
     * GB10), the row bound from both sides (8 and 9 rows at K = 5120, 4096 and
     * 17408), several units per block (65544 columns exceed the 8192-block
     * grid, so blocks restage their tile), a partial last chunk with scale rows
     * that are not 16-byte multiples (K = 5152), an uneven split whose K ranges
     * start mid scale row and end in a partial chunk (K = 17696), a K shorter
     * than one chunk (96), and shapes failing only the tile size (K = 3968, an
     * 18,112-byte tile) or only the wave count (64 columns). With
     * -Dnd4j.weightOnly.staging=direct every run must take the direct kernel;
     * with =bulk the choice is not asserted, and every shape whose tile fits is
     * compared staged against direct.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNvfp4BulkStagingMatchesDirectKernel(Nd4jBackend backend) {
        assumeFalse(Nd4j.getEnvironment().isCPU(), "the weight-only tensor-core GEMM is a CUDA kernel");
        boolean bulkCopies = Nd4j.getNativeOps()
                .getDeviceMajor(Nd4j.getAffinityManager().getDeviceForCurrentThread()) >= 9;
        // Set from -Dnd4j.weightOnly.staging; empty selects the measured rule.
        String staging = System.getenv("SD_WEIGHT_ONLY_STAGING");
        boolean measuredRule = staging == null || staging.isEmpty();
        final int BULK = 1, DIRECT = 0, EITHER = -1;
        // {K, N, then (rows, kernel on sm_90+) pairs}
        int[][] shapes = {
                {5120, 34816, 1, BULK, 5, BULK, 8, BULK, 9, DIRECT, 17, DIRECT},
                {4096, 65544, 3, BULK, 8, BULK, 9, DIRECT},
                {5152, 32768, 2, BULK, 6, BULK},
                {17408, 5120, 1, EITHER, 5, EITHER, 8, EITHER, 9, DIRECT},
                {17696, 5120, 3, EITHER, 7, EITHER},
                {3968, 32768, 1, DIRECT},
                {5120, 64, 1, DIRECT},
                {96, 4096, 3, DIRECT}};
        java.util.Random random = new java.util.Random(20260930L);
        INDArray global = scalar(.1003f);
        DspDiagnostics.setCategories(DspDiagnostics.BACKEND);
        try {
            for (int[] shape : shapes) {
                int k = shape[0], n = shape[1];
                byte[] bytes = new byte[n * (k / 2)];
                random.nextBytes(bytes);
                byte[] blocks = new byte[n * (k / 16)];
                for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
                byte[] padded = new byte[blocks.length + 16];
                System.arraycopy(blocks, 0, padded, 2, blocks.length);
                INDArray w = raw(DataType.UBYTE, bytes, n, k / 2);
                INDArray aligned = raw(DataType.FLOAT8, blocks, n, k / 16);
                INDArray offset = Nd4j.create(raw(DataType.FLOAT8, padded, padded.length).data(),
                        new long[]{n, k / 16}, new long[]{k / 16, 1}, 2, 'c');
                for (int pair = 2; pair < shape.length; pair += 2) {
                    int rows = shape[pair], expected = shape[pair + 1];
                    for (DataType dtype : new DataType[]{DataType.BFLOAT16, DataType.HALF}) {
                        float[] values = new float[rows * k];
                        for (int i = 0; i < values.length; i++)
                            values[i] = (random.nextInt(4097) - 2048) / 1024.0f;
                        INDArray x = Nd4j.create(values, new long[]{rows, k}, DataType.FLOAT).castTo(dtype);
                        for (boolean floatOutput : new boolean[]{false, true}) {
                            String context = dtype + "/floatOutput=" + floatOutput + "/rows=" + rows + "/k=" + k
                                    + "/n=" + n;
                            DspDiagnostics.clear();
                            INDArray direct =
                                    Nd4j.exec(new ModelOptNvfp4Linear(x, w, offset, global, floatOutput))[0];
                            assertEquals("direct", weightOnlyKernel(context),
                                    context + ": scales without 16-byte alignment must take the direct kernel");
                            DspDiagnostics.clear();
                            INDArray staged =
                                    Nd4j.exec(new ModelOptNvfp4Linear(x, w, aligned, global, floatOutput))[0];
                            String kernel = weightOnlyKernel(context);
                            if (!bulkCopies)
                                assertEquals("direct", kernel, context + ": bulk copies need sm_90");
                            else if ("direct".equals(staging))
                                assertEquals("direct", kernel, context + ": staging is forced off");
                            else if (measuredRule && expected == BULK)
                                assertEquals("bulk", kernel, context + ": this shape must be staged");
                            else if (measuredRule && expected == DIRECT)
                                assertEquals("direct", kernel, context + ": staging this shape was measured slower");
                            assertArrayEquals(direct.castTo(DataType.FLOAT).data().asFloat(),
                                    staged.castTo(DataType.FLOAT).data().asFloat(),
                                    context + ": the " + kernel + " kernel differs from the direct kernel");
                        }
                    }
                }
            }
        } finally {
            DspDiagnostics.setCategories(DspDiagnostics.NONE);
        }
    }

    /**
     * The FP8 weight-only GEMM's bulk kernel must give the direct kernel's bits.
     * FP8's weight scale is a scalar, so no call can be forced onto the direct
     * kernel the way misaligned NVFP4 block scales are; each shape is compared
     * across row counts instead. Every call's rows must equal the same rows of a
     * 16-row call, which takes the direct kernel (staging stops at 8 rows), and
     * a row does not depend on the call's width
     * (testRowResultsIndependentOfRowCount). Besides the NVFP4 bounds (an 18 KB
     * tile, four waves of units, at most 8 rows), FP8 stages a unit only while
     * its activation slice, rows by its K range, fits the L1 that the resident
     * bulk blocks leave, because every unit quantizes its slice again. At
     * K = 5120 that is 2 rows on every sm_90+ device (GB10 keeps 28 KB beside two
     * blocks), 3 rows only where more L1 remains, and 7 rows nowhere. BULK,
     * DIRECT and EITHER mean what they mean in
     * testNvfp4BulkStagingMatchesDirectKernel. Covers the tile bound from both
     * sides (K = 2304, an 18,496-byte tile, and K = 2176, 17,472 bytes), the
     * Qwen3.6-27B FP8 output projection (K = 6144 over 5120 columns, split in two
     * on GB10) and a K split in four (17408 over 5120 columns).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFp8BulkStagingMatchesDirectKernel(Nd4jBackend backend) {
        assumeFalse(Nd4j.getEnvironment().isCPU(), "the weight-only tensor-core GEMM is a CUDA kernel");
        boolean bulkCopies = Nd4j.getNativeOps()
                .getDeviceMajor(Nd4j.getAffinityManager().getDeviceForCurrentThread()) >= 9;
        // Set from -Dnd4j.weightOnly.staging; empty selects the measured rule.
        String staging = System.getenv("SD_WEIGHT_ONLY_STAGING");
        boolean measuredRule = staging == null || staging.isEmpty();
        final int BULK = 1, DIRECT = 0, EITHER = -1;
        final int widest = 16;
        // {K, N, then (rows, kernel on sm_90+) pairs}
        int[][] shapes = {
                {5120, 32768, 1, BULK, 2, BULK, 3, EITHER, 7, DIRECT, 9, DIRECT},
                {2304, 32768, 1, BULK, 5, BULK},
                {2176, 32768, 1, DIRECT},
                {6144, 5120, 1, EITHER, 4, EITHER, 5, EITHER, 9, DIRECT},
                {17408, 5120, 1, EITHER, 3, EITHER, 9, DIRECT}};
        java.util.Random random = new java.util.Random(20261001L);
        INDArray weightScale = scalar(.7f), inputScale = scalar(.3f);
        DspDiagnostics.setCategories(DspDiagnostics.BACKEND);
        try {
            for (int[] shape : shapes) {
                int k = shape[0], n = shape[1];
                byte[] bytes = new byte[n * k];
                for (int i = 0; i < bytes.length; i++)
                    bytes[i] = (byte) ((random.nextInt(0x60) + 0x10) | (random.nextBoolean() ? 128 : 0));
                INDArray w = raw(DataType.FLOAT8, bytes, n, k);
                for (DataType dtype : new DataType[]{DataType.BFLOAT16, DataType.HALF}) {
                    float[] values = new float[widest * k];
                    for (int i = 0; i < values.length; i++) values[i] = (random.nextInt(4097) - 2048) / 1024.0f;
                    INDArray all = Nd4j.create(values, new long[]{widest, k}, DataType.FLOAT).castTo(dtype);
                    for (boolean floatOutput : new boolean[]{false, true}) {
                        String context = dtype + "/floatOutput=" + floatOutput + "/k=" + k + "/n=" + n;
                        DspDiagnostics.clear();
                        INDArray reference =
                                Nd4j.exec(new ModelOptFp8Linear(all, w, weightScale, inputScale, floatOutput))[0];
                        if (!"bulk".equals(staging))
                            assertEquals("direct", weightOnlyKernel(context + "/rows=" + widest),
                                    context + ": a 16-row call must take the direct kernel");
                        for (int pair = 2; pair < shape.length; pair += 2) {
                            int rows = shape[pair], expected = shape[pair + 1];
                            String call = context + "/rows=" + rows;
                            INDArray x = all.get(NDArrayIndex.interval(0, rows), NDArrayIndex.all()).dup('c');
                            DspDiagnostics.clear();
                            INDArray z = Nd4j.exec(new ModelOptFp8Linear(x, w, weightScale, inputScale, floatOutput))[0];
                            String kernel = weightOnlyKernel(call);
                            if (!bulkCopies)
                                assertEquals("direct", kernel, call + ": bulk copies need sm_90");
                            else if ("direct".equals(staging))
                                assertEquals("direct", kernel, call + ": staging is forced off");
                            else if (measuredRule && expected == BULK)
                                assertEquals("bulk", kernel, call + ": this shape must be staged");
                            else if (measuredRule && expected == DIRECT)
                                assertEquals("direct", kernel, call + ": staging this shape was measured slower");
                            INDArray rowsOfReference =
                                    reference.get(NDArrayIndex.interval(0, rows), NDArrayIndex.all()).dup('c');
                            assertArrayEquals(rowsOfReference.castTo(DataType.FLOAT).data().asFloat(),
                                    z.castTo(DataType.FLOAT).data().asFloat(),
                                    call + ": the " + kernel + " kernel differs from the 16-row direct call");
                        }
                    }
                }
            }
        } finally {
            DspDiagnostics.setCategories(DspDiagnostics.NONE);
        }
    }

    // The path a ModelOpt linear took since the last DspDiagnostics.clear(),
    // from its BACKEND event.
    private static String modelOptPath(String context) {
        java.util.regex.Matcher path = java.util.regex.Pattern.compile("ModelOpt (?:FP8|NVFP4) linear path=([a-z-]+) ")
                .matcher(DspDiagnostics.getJsonReport());
        assertTrue(path.find(), context + ": no ModelOpt linear path was recorded");
        String taken = path.group(1);
        assertFalse(path.find(), context + ": more than one ModelOpt linear path was recorded");
        return taken;
    }

    // The weight-only GEMM kernel a ModelOpt linear launched since the last
    // DspDiagnostics.clear(), from its BACKEND event.
    private static String weightOnlyKernel(String context) {
        java.util.regex.Matcher launch = java.util.regex.Pattern.compile("WeightOnlyGemm (bulk|direct) ")
                .matcher(DspDiagnostics.getJsonReport());
        assertTrue(launch.find(), context + ": no weight-only GEMM launch was recorded");
        String kernel = launch.group(1);
        assertFalse(launch.find(), context + ": more than one weight-only GEMM launch was recorded");
        return kernel;
    }

    /**
     * Throughput probe for the decode GEMVs of Qwen3.6-27B NVFP4 (shapes taken
     * from the imported graph), run in isolation so kernel work can be measured
     * in seconds instead of a full model run. Reports microseconds per call and
     * effective weight bandwidth for the W=1 and W=5 row counts
     * (-Dmodelopt.gemvBench.rows=r,... overrides them). Opt-in:
     * -Dmodelopt.gemvBench=true (not a correctness test).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named = "modelopt.gemvBench", matches = "true")
    public void benchDecodeGemvShapes(Nd4jBackend backend) {
        // {nvfp4 ? 1 : 0, N, K}
        int[][] shapes = {{1, 17408, 5120}, {1, 5120, 17408}, {0, 6144, 5120}, {0, 10240, 5120}, {0, 5120, 6144}};
        // -Dmodelopt.gemvBench.shapes=nv:N:K,... overrides the shape list (nv = 1 for NVFP4, 0 for FP8).
        String shapeList = System.getProperty("modelopt.gemvBench.shapes");
        if (shapeList != null && !shapeList.isBlank()) {
            String[] entries = shapeList.split(",");
            shapes = new int[entries.length][];
            for (int i = 0; i < entries.length; i++) {
                String[] parts = entries[i].trim().split(":");
                shapes[i] = new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2])};
            }
        }
        int iterations = Integer.getInteger("modelopt.gemvBench.iterations", 50);
        int[] rowCounts = java.util.Arrays.stream(System.getProperty("modelopt.gemvBench.rows", "1,5").split(","))
                .map(String::trim).mapToInt(Integer::parseInt).toArray();
        java.util.Random random = new java.util.Random(7);
        for (int[] shape : shapes) {
            boolean nv = shape[0] == 1;
            int n = shape[1], k = shape[2];
            byte[] bytes = new byte[n * (nv ? k / 2 : k)];
            for (int i = 0; i < bytes.length; i++)
                bytes[i] = nv ? (byte) random.nextInt(256)
                        : (byte) ((random.nextInt(0x60) + 0x10) | (random.nextBoolean() ? 128 : 0));
            byte[] blocks = new byte[nv ? n * (k / 16) : 0];
            for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
            INDArray w = raw(nv ? DataType.UBYTE : DataType.FLOAT8, bytes, n, nv ? k / 2 : k);
            INDArray scale = nv ? raw(DataType.FLOAT8, blocks, n, k / 16) : scalar(.7f);
            INDArray global = scalar(nv ? .1003f : .3f);
            long weightBytes = (long) bytes.length + blocks.length;
            for (int rows : rowCounts) {
                INDArray x = Nd4j.rand(DataType.FLOAT, rows, k).subi(0.5).castTo(DataType.BFLOAT16);
                INDArray z = Nd4j.create(DataType.BFLOAT16, rows, n);
                DynamicCustomOp op = nv ? new ModelOptNvfp4Linear(x, w, scale, global, false)
                        : new ModelOptFp8Linear(x, w, scale, global, false);
                op.addOutputArgument(z);
                for (int i = 0; i < 5; i++) Nd4j.exec(op);
                Nd4j.getExecutioner().commit();
                long start = System.nanoTime();
                for (int i = 0; i < iterations; i++) Nd4j.exec(op);
                Nd4j.getExecutioner().commit();
                double usPerCall = (System.nanoTime() - start) / 1e3 / iterations;
                System.out.printf("GEMV_BENCH %s N=%d K=%d rows=%d: %.1f us/call, %.1f GB/s weights%n",
                        nv ? "nvfp4" : "fp8", n, k, rows, usPerCall, weightBytes / (usPerCall * 1e3));
            }
        }
    }

    /**
     * The Qwen3.6-27B NVFP4 decode MLP in model order — gate and up GEMVs
     * (N=17408, K=5120), swish-mul, down GEMV (N=5120, K=17408) — over
     * -Dmodelopt.mlpBench.layers distinct weight sets, so each kernel runs in the
     * memory context the model gives it instead of back to back with itself.
     * Profile it (nsys) for per-kernel times. Opt-in: -Dmodelopt.mlpBench=true.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named = "modelopt.mlpBench", matches = "true")
    public void benchDecodeMlpSequence(Nd4jBackend backend) {
        int hidden = 5120, intermediate = 17408;
        int layers = Integer.getInteger("modelopt.mlpBench.layers", 8);
        int iterations = Integer.getInteger("modelopt.mlpBench.iterations", 20);
        java.util.Random random = new java.util.Random(11);
        DynamicCustomOp[][] ops = new DynamicCustomOp[layers][];
        INDArray x = Nd4j.rand(DataType.FLOAT, 1, hidden).subi(0.5).castTo(DataType.BFLOAT16);
        for (int layer = 0; layer < layers; layer++) {
            INDArray gate = Nd4j.create(DataType.BFLOAT16, 1, intermediate);
            INDArray up = Nd4j.create(DataType.BFLOAT16, 1, intermediate);
            INDArray hiddenState = Nd4j.create(DataType.BFLOAT16, 1, intermediate);
            INDArray down = Nd4j.create(DataType.BFLOAT16, 1, hidden);
            DynamicCustomOp gateOp = nvfp4Projection(random, x, intermediate, hidden);
            gateOp.addOutputArgument(gate);
            DynamicCustomOp upOp = nvfp4Projection(random, x, intermediate, hidden);
            upOp.addOutputArgument(up);
            DynamicCustomOp downOp = nvfp4Projection(random, hiddenState, hidden, intermediate);
            downOp.addOutputArgument(down);
            ops[layer] = new DynamicCustomOp[]{gateOp, upOp, new SwishMul(gate, up, hiddenState), downOp};
        }
        for (int warmup = 0; warmup < 2; warmup++)
            for (DynamicCustomOp[] layer : ops)
                for (DynamicCustomOp op : layer) Nd4j.exec(op);
        Nd4j.getExecutioner().commit();
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++)
            for (DynamicCustomOp[] layer : ops)
                for (DynamicCustomOp op : layer) Nd4j.exec(op);
        Nd4j.getExecutioner().commit();
        System.out.printf("MLP_BENCH layers=%d: %.1f us/layer%n", layers,
                (System.nanoTime() - start) / 1e3 / iterations / layers);
    }

    /**
     * benchDecodeMlpSequence as an optimized SameDiff graph executed by DSP, so
     * the kernels replay back to back from a CUDA graph with activations in
     * the plan's arena, as in the model. Each layer's down projection feeds
     * the next layer. -Dmodelopt.mlpBench.fp8Interlude=true precedes each MLP
     * with the model's FP8 attention-side projections (N=10240 and N=6144 over
     * K=5120, then N=5120 over K=6144), whose weights stream through L2 at
     * normal priority between the MLPs. =allocate creates the same constants
     * in the same order but leaves the projections out of the computation, so
     * the MLP weights sit where they sit under =true while L2 sees only MLP
     * traffic. Opt-in: -Dmodelopt.mlpDspBench=true.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named = "modelopt.mlpDspBench", matches = "true")
    public void benchDecodeMlpDsp(Nd4jBackend backend) {
        int hidden = 5120, intermediate = 17408, value = 6144;
        int layers = Integer.getInteger("modelopt.mlpBench.layers", 8);
        int iterations = Integer.getInteger("modelopt.mlpBench.iterations", 20);
        String interlude = System.getProperty("modelopt.mlpBench.fp8Interlude", "");
        boolean runInterlude = interlude.equals("true");
        boolean allocateInterlude = runInterlude || interlude.equals("allocate");
        java.util.Random random = new java.util.Random(11);
        try (SameDiff source = SameDiff.create()) {
            SDVariable x = source.placeHolder("x", DataType.BFLOAT16, 1, hidden);
            for (int layer = 0; layer < layers; layer++) {
                if (allocateInterlude) {
                    SDVariable wide = fp8Projection(source, random, "wide" + layer, x, 10240, hidden);
                    SDVariable narrow = fp8Projection(source, random, "narrow" + layer, x, value, hidden);
                    SDVariable mixed = narrow.add(wide.get(SDIndex.all(), SDIndex.interval(0, value)));
                    SDVariable out = fp8Projection(source, random, "out" + layer, mixed, hidden, value);
                    if (runInterlude) x = out;
                }
                SDVariable gate = nvfp4Projection(source, random, "gate" + layer, x, intermediate, hidden);
                SDVariable up = nvfp4Projection(source, random, "up" + layer, x, intermediate, hidden);
                SDVariable hiddenState = source.nn().swish(gate).mul(up);
                x = nvfp4Projection(source, random, "down" + layer, hiddenState, hidden, intermediate);
            }
            String output = x.name();
            source.setOutputs(output);
            SameDiff optimized = GraphOptimizer.optimize(source, Collections.singletonList(output));
            try {
                optimized.setDspAutoCompileEnabled(true);
                optimized.setDspNativeAutoCompileEnabled(true);
                Map<String, INDArray> inputs = Map.of("x",
                        Nd4j.rand(DataType.FLOAT, 1, hidden).subi(0.5).castTo(DataType.BFLOAT16));
                for (int warmup = 0; warmup < 12; warmup++) optimized.outputSingle(inputs, output);
                DspPlanAssertions.assertFullyReplaying(optimized, "the MLP bench must time graph replay");
                long start = System.nanoTime();
                for (int i = 0; i < iterations; i++) optimized.outputSingle(inputs, output);
                System.out.printf("MLP_DSP_BENCH layers=%d: %.1f us/layer%n", layers,
                        (System.nanoTime() - start) / 1e3 / iterations / layers);
            } finally {
                if (optimized != source) optimized.close();
            }
        }
    }

    private static SDVariable nvfp4Projection(SameDiff sd, java.util.Random random, String name, SDVariable x,
                                              int n, int k) {
        DynamicCustomOp op = nvfp4Projection(random, Nd4j.create(DataType.BFLOAT16, 1, k), n, k);
        INDArray[] arguments = op.inputArguments().toArray(new INDArray[0]);
        return new ModelOptNvfp4Linear(sd, x, sd.constant(name + "_w", arguments[1]),
                sd.constant(name + "_scale", arguments[2]), sd.constant(name + "_global", arguments[3]), false)
                .outputVariable();
    }

    // A random FP8 projection [n, k] of x: weight magnitudes 0.5 to 1 (E4M3
    // exponent 6, random sign and mantissa), scaled so outputs keep x's range.
    private static SDVariable fp8Projection(SameDiff sd, java.util.Random random, String name, SDVariable x,
                                            int n, int k) {
        byte[] bytes = new byte[n * k];
        random.nextBytes(bytes);
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) ((bytes[i] & 0x87) | 0x30);
        return new ModelOptFp8Linear(sd, x, sd.constant(name + "_w", raw(DataType.FLOAT8, bytes, n, k)),
                sd.constant(name + "_scale", scalar((float) (1 / (0.75 * Math.sqrt(k))))),
                sd.constant(name + "_input", scalar(1 / 64f)), false).outputVariable();
    }

    // A random NVFP4 projection [n, k] of x, scales in the decoded-value range the model uses.
    private static DynamicCustomOp nvfp4Projection(java.util.Random random, INDArray x, int n, int k) {
        byte[] bytes = new byte[n * (k / 2)];
        random.nextBytes(bytes);
        byte[] blocks = new byte[n * (k / 16)];
        for (int i = 0; i < blocks.length; i++) blocks[i] = (byte) (0x30 + random.nextInt(0x11));
        return new ModelOptNvfp4Linear(x, raw(DataType.UBYTE, bytes, n, k / 2), raw(DataType.FLOAT8, blocks, n, k / 16),
                scalar(.1003f), false);
    }

    private static void assertResult(Fixture f, INDArray z, boolean fp32, String context) {
        assertArrayEquals(new long[]{6, 3}, z.shape(), context);
        DataType dtype = fp32 ? DataType.FLOAT : f.x.dataType();
        assertEquals(dtype, z.dataType(), context);
        for (int row = 0; row < 6; row++)
            for (int n = 0; n < 3; n++)
                assertEquals(round(reference(f, row, n), dtype), z.getDouble(row, n), 2e-5, context);
    }

    private static double reference(Fixture f, int row, int n) {
        DataType acc = aggregate(f.x.dataType());
        double sum = 0;
        for (int k = 0; k < 32; k++) {
            double a = f.x.getDouble(row, k);
            double b;
            if (f.nv) {
                int bits = f.bytes[n * 16 + k / 2] & 255;
                float scale = decodeE4m3(f.blocks[n * 2 + k / 16] & 255) * .1003f;
                b = round(signedNibble((bits >>> (4 * (k & 1))) & 15) * scale, f.x.dataType());
            } else {
                a = fp8Activation(acc, a, .3f);
                b = fp8Weight(acc, f.bytes[n * 32 + k] & 255, .7f);
            }
            sum = fma(acc, a, b, sum);
        }
        return sum;
    }

    // simdOps::AggregateType, the native kernels' accumulator: FP64 for DOUBLE
    // activations, FP32 for every narrower float type.
    private static DataType aggregate(DataType dtype) {
        return dtype == DataType.DOUBLE ? DataType.DOUBLE : DataType.FLOAT;
    }

    // a * b + c with one rounding in the accumulator type.
    private static double fma(DataType acc, double a, double b, double c) {
        return acc == DataType.DOUBLE ? Math.fma(a, b, c) : Math.fma((float) a, (float) b, (float) c);
    }

    private static double add(DataType acc, double a, double b) {
        return acc == DataType.DOUBLE ? a + b : (float) a + (float) b;
    }

    // ModelOpt FP8 activation in the accumulator type: E4M3(x / inputScale) * inputScale.
    // E4M3 conversion from FP64 rounds through FP32, as every narrow float type converts from double.
    private static double fp8Activation(DataType acc, double x, float inputScale) {
        return acc == DataType.DOUBLE ? (double) quantizeE4m3((float) (x / inputScale)) * inputScale
                : quantizeE4m3((float) x / inputScale) * inputScale;
    }

    // ModelOpt FP8 weight in the accumulator type: E4M3 * weightScale.
    private static double fp8Weight(DataType acc, int bits, float weightScale) {
        return acc == DataType.DOUBLE ? (double) decodeE4m3(bits) * weightScale : decodeE4m3(bits) * weightScale;
    }

    private static float signedNibble(int bits) {
        return (bits < 8 ? 1 : -1) * E2M1[bits & 7];
    }

    private static float decodeE4m3(int bits) {
        int magnitude = bits & 127;
        int exponent = magnitude >>> 3;
        int mantissa = magnitude & 7;
        float value = exponent == 0 ? Math.scalb((float) mantissa, -9)
                : Math.scalb(1 + mantissa / 8.0f, exponent - 7);
        return bits < 128 ? value : -value;
    }

    private static float quantizeE4m3(float x) {
        // Exhaustive nearest representable value, ties to the even encoding.
        float magnitude = Math.min(Math.abs(x), 448);
        int best = 0;
        float distance = Float.POSITIVE_INFINITY;
        for (int bits = 0; bits <= 126; bits++) {
            float candidateDistance = Math.abs(decodeE4m3(bits) - magnitude);
            if (candidateDistance < distance || (candidateDistance == distance && (bits & 1) == 0)) {
                distance = candidateDistance;
                best = bits;
            }
        }
        return Math.copySign(decodeE4m3(best), x);
    }

    // x rounded to nearest even in dtype: DOUBLE keeps x, FLOAT rounds once to FP32.
    private static double round(double x, DataType dtype) {
        if (dtype == DataType.DOUBLE || x == 0) return x;
        if (dtype == DataType.FLOAT) return (float) x;
        int fractionBits = dtype == DataType.HALF ? 10 : 7;
        int minExponent = dtype == DataType.HALF ? -14 : -126;
        double step = Math.scalb(1.0, Math.max(Math.getExponent(Math.abs(x)), minExponent) - fractionBits);
        return (float) (Math.rint(x / step) * step);
    }

    private static INDArray raw(DataType dtype, byte[] bytes, long... shape) {
        INDArray array = Nd4j.createUninitialized(dtype, shape, 'c');
        assertEquals(1, dtype.width(), "raw helper accepts only one-byte storage");
        assertEquals(bytes.length, array.length(), "raw payload must exactly fill its typed allocation");
        if (bytes.length > 0) {
            // Borrow typed allocation storage, not numeric byte-to-FP8 conversion.
            // Like SafeTensorsReader, this pointer alias must not be deallocated.
            new BytePointer(array.data().pointer()).capacity(bytes.length).put(bytes);
            Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        }
        return array;
    }

    private static INDArray scalar(float value) {
        INDArray result = Nd4j.scalar(DataType.FLOAT, value);
        Nd4j.getAffinityManager().tagLocation(result, AffinityManager.Location.HOST);
        return result;
    }

    private static INDArray layout(INDArray array, int kind) {
        if (kind == 0) return array;
        if (kind == 1) return array.dup('f');
        if (kind == 2) return array.transpose().dup('c').transpose();
        INDArray parent = Nd4j.create(array.dataType(), array.size(0) + 2, array.size(1) * 2 + 2);
        INDArray view = parent.get(NDArrayIndex.interval(1, array.size(0) + 1),
                NDArrayIndex.interval(1, 2, array.size(1) * 2 + 1));
        view.assign(array);
        return view;
    }

    private static class Fixture {
        final boolean nv;
        final INDArray x, w, scale, second;
        final byte[] bytes, blocks;

        Fixture(boolean nv, DataType dtype, int order) {
            this.nv = nv;
            float[] values = new float[6 * 32];
            for (int i = 0; i < values.length; i++) values[i] = ((i * 7) % 31 - 15) / 8.0f;
            x = layout(Nd4j.create(values, new long[]{6, 32}, dtype), order);
            bytes = new byte[3 * (nv ? 16 : 32)];
            for (int i = 0; i < bytes.length; i++)
                bytes[i] = nv ? (byte) ((i & 15) | (((i + 5) & 15) << 4))
                        : (byte) ((0x28 + i % 32) | (i % 3 == 0 ? 128 : 0));
            blocks = new byte[]{0x38, 0x39, 0x30, 0x40, 0x28, 0x3c};
            w = layout(raw(nv ? DataType.UBYTE : DataType.FLOAT8, bytes, 3, nv ? 16 : 32), order);
            scale = nv ? layout(raw(DataType.FLOAT8, blocks, 3, 2), order) : scalar(.7f);
            second = scalar(nv ? .1003f : .3f);
        }
    }
}
