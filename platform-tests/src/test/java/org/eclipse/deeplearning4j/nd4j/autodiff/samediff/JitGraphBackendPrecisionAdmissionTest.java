/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** FLOAT32 storage admission, not a claim of low-precision JIT lowering. */
@Tag("dsp")
class JitGraphBackendPrecisionAdmissionTest {
    private enum Binding { INPUT, INTERMEDIATE, OUTPUT }

    static Stream<GraphExecutionMode> modes() {
        return Stream.of(GraphExecutionMode.NVRTC_JIT, GraphExecutionMode.PTX_JIT);
    }

    static Stream<Arguments> rejectedBindings() {
        return modes().flatMap(mode -> Stream.of(DataType.HALF, DataType.BFLOAT16,
                DataType.DOUBLE, DataType.BOOL).flatMap(type ->
                Stream.of(Binding.values()).map(binding -> Arguments.of(mode, type, binding))));
    }

    private static String backend(GraphExecutionMode mode) {
        return mode == GraphExecutionMode.NVRTC_JIT ? "NVRTC" : "PTX";
    }

    private static String implementation(GraphExecutionMode mode) {
        return mode == GraphExecutionMode.NVRTC_JIT ? "NvrtcGraphBackend" : "PtxGraphBackend";
    }

    private static SameDiff graph(GraphExecutionMode mode, DataType type, Binding binding) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", binding == Binding.INPUT ? type : DataType.FLOAT, 32);
        SDVariable value = binding == Binding.INPUT && type != DataType.FLOAT
                ? x.castTo("inputFloat", DataType.FLOAT) : x;
        // Enough pointwise slots to exercise compilation rather than a small native-only segment.
        for (int i = 0; i < 12; i++) {
            value = sd.math().neg("neg" + i, value);
            if (i == 5 && binding == Binding.INTERMEDIATE) {
                value = value.castTo("narrowIntermediate", type).castTo("floatIntermediate", DataType.FLOAT);
            }
        }
        if (binding == Binding.OUTPUT) value = value.castTo("typedOutput", type);
        sd.identity("out", value);
        // Deliberately unrelated model tensor: its dtype is not this segment's contract.
        sd.constant("unrelatedBool", Nd4j.zeros(DataType.BOOL, 7));
        sd.setOutputs("out");
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        return sd;
    }

    @ParameterizedTest
    @MethodSource("modes")
    void floatCompilesAndReplaysWithUnrelatedBooleanModelTensor(GraphExecutionMode mode) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA JIT backends");
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(true);
        try (SameDiff sd = graph(mode, DataType.FLOAT, Binding.INPUT);
             INDArray input = Nd4j.ones(DataType.FLOAT, 32)) {
            for (int step = 0; step < 8; step++) {
                input.assign(step + 1);
                INDArray actual = sd.output(Map.of("x", input), "out").get("out");
                assertEquals(DataType.FLOAT, actual.dataType());
                assertTrue(input.equalsWithEps(actual, 1e-6), "Twelve negations must preserve the input");
            }
            DspHandle handle = sd.dsp();
            assertTrue(handle.isCompiled());
            assertTrue(handle.numSegments() > 0);
            boolean compiled = false;
            for (int i = 0; i < handle.numSegments(); i++) {
                if (handle.segmentCompiledBackend(i).contains(backend(mode))) {
                    String audit = handle.segmentCompilationAudit(i);
                    JsonObject segment = new JsonParser().parse(audit).getAsJsonObject();
                    assertFalse(segment.get("compilationFailed").getAsBoolean(), audit);
                    assertTrue(segment.get("compiledByBackend").getAsString().contains(backend(mode)), audit);
                    assertTrue(segment.get("capturable").getAsBoolean(), audit);
                    assertTrue(segment.get("executionCount").getAsInt() >= 1, audit);
                    compiled = true;
                }
            }
            assertTrue(compiled, "FLOAT success must be JIT compilation, not host fallback");
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    @ParameterizedTest(name = "{0}-{1}-{2}")
    @MethodSource("rejectedBindings")
    void unsupportedStorageFailsAtCompilation(GraphExecutionMode mode, DataType type, Binding binding) {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA JIT backends");
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        int mask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
        int level = Nd4j.getNativeOps().dspDiagGetLevel();
        InferenceSession.setDynamicShapePlanEnabled(true);
        DspDiagnostics.setCategories(DspDiagnostics.COMPILE);
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_FULL);
        DspDiagnostics.clear();
        try (SameDiff sd = graph(mode, type, binding);
             INDArray input = Nd4j.ones(binding == Binding.INPUT ? type : DataType.FLOAT, 32)) {
            assertThrows(RuntimeException.class, () -> {
                for (int step = 0; step < 8; step++) sd.output(Map.of("x", input), "out");
            }, "Explicit JIT mode must reject unsupported storage, not execute a fallback");
            // Only COMPILE events are enabled: a launch-time failure or generic exception
            // cannot satisfy this assertion. The backend's concrete dtype gate must fire.
            String report = DspDiagnostics.getJsonReport();
            assertTrue(report.contains(implementation(mode) + ": FLOAT32-only JIT tensor contract rejected"), report);
            assertTrue(report.contains("dtype=" + type.toInt() + "; required FLOAT32"), report);
        } finally {
            DspDiagnostics.clear();
            DspDiagnostics.setCategories(mask);
            DspDiagnostics.setLevel(level);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    @Test
    @Tag("source-lint")
    void concreteAdmissionPrecedesCacheGenerationAndLaunch() throws Exception {
        Path gpu = Path.of("..").toAbsolutePath().normalize().resolve("libnd4j/include/graph/gpu");
        for (String name : new String[]{"NvrtcGraphBackend", "PtxGraphBackend"}) {
            String source = Files.readString(gpu.resolve(name + ".cu"));
            String compile = source.substring(source.indexOf("bool " + name + "::compileSegment("));
            int admission = compile.indexOf("jitValidateFloatTensorBindings(");
            assertTrue(admission >= 0 && admission < compile.indexOf("cache_.find(key)"));
            assertTrue(admission < compile.indexOf(name.equals("NvrtcGraphBackend")
                    ? "generateCudaSource(" : "generatePtx("));
            assertTrue(admission < compile.indexOf("GpuKernelLauncher::loadPtxModule("));
            assertTrue(compile.substring(admission, compile.indexOf("cache_.find(key)")).contains("return false;"));
        }
        String common = Files.readString(gpu.resolve("JitGraphBackendCommon.cu"));
        String admission = common.substring(common.indexOf("bool jitValidateFloatTensorBindings("),
                common.indexOf("Status jitExecuteSegment("));
        assertTrue(admission.contains("array && array->dataType() == DataType::FLOAT"));
        assertTrue(admission.contains("std::string(\"missing\")"));
        assertTrue(admission.contains("wiring.inputSourceIndices[i]"));
        assertTrue(admission.contains("wiring.outputSlotIndices[o]"));
        assertTrue(admission.contains("externalInputs && index >= 0 && index < numExternalInputs"));
        assertTrue(admission.contains("outputSlots && index >= 0 && index < totalOutputSlots"));
        assertFalse(admission.contains("< numExternalInputs;"), "Do not scan unrelated model externals");
        String execute = common.substring(common.indexOf("Status jitExecuteSegment("));
        assertTrue(execute.indexOf("jitValidateFloatTensorBindings(") < execute.indexOf("cache.find(key)"));
        assertTrue(execute.indexOf("jitValidateFloatTensorBindings(") < execute.indexOf("GpuKernelLauncher::launchKernel("));
        assertTrue(execute.indexOf("arr->dataType() != DataType::FLOAT") < execute.indexOf("arr->specialBuffer()"));
    }
}
