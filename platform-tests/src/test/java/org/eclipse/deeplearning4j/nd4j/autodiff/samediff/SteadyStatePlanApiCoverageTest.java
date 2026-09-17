/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueContext;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Source coverage for every preset, plus declaration checks without loading native libraries. */
class SteadyStatePlanApiCoverageTest {
    private static final String BACKENDS = "nd4j/nd4j-backends/nd4j-backend-impls/";
    private static final String STEADY_STATE_DECLARATION =
            "@Override public native int executeSteadyStatePlan(@Cast(\"sd::Pointer\") Pointer planHandle, "
                    + "org.nd4j.nativeblas.OpaqueContext opContext, @Cast(\"sd::Pointer\") Pointer stream);";
    private static final String[][] BACKEND_ROWS = {
        {"native", "cpu", "Cpu", "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu", "cpu/NativeOps_dsp.cpp"},
        {"cuda", "cuda", "Cuda", "org.nd4j.linalg.jcublas.bindings.Nd4jCuda", "cuda/NativeOps_dsp.cu"},
        {"minimizer", "minimal", "Minimal", "org.nd4j.linalg.minimal.bindings.Nd4jMinimal", "cpu/NativeOps_dsp.cpp"},
        {"tpu", "tpu", "Tpu", "org.nd4j.linalg.jtpu.bindings.Nd4jTpu", "cpu/NativeOps_dsp.cpp"},
        {"vulkan", "vulkan", "Vulkan", "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan", "vulkan/NativeOps_dsp_plan.cpp"},
        {"metal", "metal", "Metal", "org.nd4j.linalg.metal.bindings.Nd4jMetal", "cpu/NativeOps_dsp.cpp"},
        {"hexagon", "hexagon", "Hexagon", "org.nd4j.linalg.hexagon.bindings.Nd4jHexagon", "cpu/NativeOps_dsp.cpp"}
    };

    static Stream<Arguments> backends() {
        return Stream.of(BACKEND_ROWS).map(row -> Arguments.of((Object[]) row));
    }

    private static Path root() {
        for (Path path = Path.of("").toAbsolutePath(); path != null; path = path.getParent()) {
            if (Files.isDirectory(path.resolve("libnd4j/include/legacy"))) return path;
        }
        throw new AssertionError("Run source coverage from the deeplearning4j checkout/platform-tests");
    }

    private static String source(String path) throws IOException {
        return Files.readString(root().resolve(path));
    }

    private static String presetPath(String module, String pkg, String name) {
        return BACKENDS + "nd4j-" + module + "-preset/src/main/java/org/nd4j/presets/"
                + pkg + "/Nd4j" + name + "Presets.java";
    }

    @ParameterizedTest(name = "{2}: preset and native owner")
    @MethodSource("backends")
    void everyNativeOpsPresetExposesSteadyState(String module, String pkg, String name,
                                               String binding, String owner) throws IOException {
        String preset = source(presetPath(module, pkg, name));
        assertTrue(preset.contains("\"" + binding + "\""), name);
        assertTrue(preset.contains("\"dsp/NativeOpsDsp.h\""), name);
        assertFalse(preset.contains("new Info(\"executeSteadyStatePlan\").skip()"), name);
        String mappings = preset;
        if (name.equals("Minimal")) {
            assertTrue(preset.contains("extends Nd4jCpuPresets"));
            assertTrue(preset.contains("super.map(infoMap)"));
            mappings = source(presetPath("native", "cpu", "Cpu"));
        }
        assertTrue(mappings.contains("new Info(\"executeSteadyStatePlan\").javaText("), name);
        assertFalse(mappings.contains("new Info(\"executeSteadyStatePlan\").objectify().annotations("), name);
        assertTrue(mappings.contains(STEADY_STATE_DECLARATION.replace("\"", "\\\"")), name);
        assertTrue(mappings.contains("new Info(\"OpaqueContext\").pointerTypes(\"org.nd4j.nativeblas.OpaqueContext\")"), name);
        assertTrue(mappings.contains("new Info(\"sd::Pointer\").cast().valueTypes(\"Pointer\")"), name);
        String helper = source(BACKENDS + "nd4j-" + module
                + "-preset/src/main/java/org/nd4j/presets/" + pkg + "/Nd4j" + name + "Helper.java");
        assertTrue(helper.contains("implements NativeOps"), name);
        String nativeSource = source("libnd4j/include/legacy/" + owner);
        assertTrue(nativeSource.contains("int executeSteadyStatePlan("), owner);
        assertTrue(nativeSource.contains("return executePlanContext(planHandle, opContext, stream, true);"), owner);
        assertTrue(nativeSource.contains("? plan->executeSteadyState("), owner);
        assertTrue(nativeSource.contains("resolveExecutionOutputCount(boundOutputCount)"), owner);
        assertTrue(nativeSource.contains("opContext->setOutputArray(i,"), owner);
    }

    /**
     * Run after JavaCPP parsing, before native builds/runtime tests. This checks only artifacts
     * whose generated source exists; the preset matrix covers absent artifacts, not hardware.
     * A documentation-only occurrence must never count as a generated method.
     */
    @Test
    void availableRegeneratedSourcesDeclareNativeInstanceOverride() throws IOException {
        int inspected = 0;
        for (String[] row : BACKEND_ROWS) {
            String module = row[0].equals("cuda") ? "cuda-backend-common" : row[0];
            Path generated = root().resolve(BACKENDS + "nd4j-" + module + "/src/main/java/"
                    + row[3].replace('.', '/') + ".java");
            if (!Files.exists(generated)) continue;
            String declarations = Files.readString(generated)
                    .replaceAll("(?s)/\\*.*?\\*/|//[^\\r\\n]*", " ")
                    .replaceAll("\\s+", " ");
            assertTrue(declarations.contains(STEADY_STATE_DECLARATION),
                    generated + ": regenerate bindings; documentation is not a native declaration");
            assertEquals(1, declarations.split("executeSteadyStatePlan\\s*\\(", -1).length - 1,
                    generated + ": expected exactly one native instance override");
            inspected++;
        }
        assertTrue(inspected > 0, "Generate bindings before running the source declaration guard");
        Path sdx = root().resolve(BACKENDS
                + "nd4j-sdx/src/main/java/org/nd4j/dsp/runtime/bindings/SdxNative.java");
        if (Files.exists(sdx)) {
            String declarations = Files.readString(sdx)
                    .replaceAll("(?s)/\\*.*?\\*/|//[^\\r\\n]*", " ")
                    .replaceAll("\\s+", " ");
            for (String suffix : new String[]{"", "Allocating"}) {
                // SDX has its own C ABI, not the NativeOps Pointer/OpaqueContext signature.
                String ordinary = "sdxRun" + suffix;
                Matcher declaration = Pattern.compile("public (?:static )?native [^;{}]*\\b"
                        + ordinary + "\\([^;{}]*\\);").matcher(declarations);
                assertTrue(declaration.find(), sdx + ": missing reference declaration " + ordinary);
                assertTrue(declarations.contains(declaration.group()
                                .replace(ordinary + "(", "sdxRunSteadyState" + suffix + "(")),
                        sdx + ": regenerate matching SDX steady-state C ABI declaration");
            }
        }
    }

    /**
     * Requires full CPU + bindings regeneration/install, then CUDA + bindings regeneration/install
     * when both artifacts are installed. Installing preset JARs alone cannot update this API.
     * Keep getDeclaredMethod strict: inherited defaults must not hide stale installed bindings.
     */
    @Test
    void installedGeneratedBindingsDeclareNativeInstanceOverride() throws Exception {
        int inspected = 0;
        for (String[] row : BACKEND_ROWS) {
            Class<?> binding;
            try {
                // initialize=false: declaration inspection is NOT hardware/runtime validation.
                binding = Class.forName(row[3], false, getClass().getClassLoader());
            } catch (ClassNotFoundException absentArtifact) {
                // All absent artifacts are still covered by the source matrix above.
                continue;
            }
            Method method = binding.getDeclaredMethod("executeSteadyStatePlan",
                    Pointer.class, OpaqueContext.class, Pointer.class);
            assertTrue(NativeOps.class.isAssignableFrom(binding), row[3]);
            assertEquals(int.class, method.getReturnType(), row[3]);
            assertTrue(Modifier.isNative(method.getModifiers()), row[3]);
            assertTrue(Modifier.isPublic(method.getModifiers()), row[3]);
            assertFalse(Modifier.isStatic(method.getModifiers()), row[3]);
            inspected++;
        }
        assertTrue(inspected > 0, "The selected platform-tests backend must supply regenerated bindings");
    }

    @Test
    void nativeOwnersAndSdxTransportRemainDistinct() throws IOException {
        String cmake = source("libnd4j/cmake/MainBuildFlow.cmake");
        assertTrue(cmake.contains("./include/legacy/cpu/*.cpp"));
        assertTrue(cmake.contains("./include/legacy/vulkan/*.cpp"));
        assertTrue(cmake.contains("./include/legacy/cuda/*.cu"));
        String vulkan = source("libnd4j/include/legacy/vulkan/NativeOps_dsp_plan.cpp");
        assertTrue(vulkan.contains("VulkanExecutionStreamGuard streamGuard(executionStream)"));
        assertTrue(vulkan.contains("executionStream->synchronize()"));
        String header = source("libnd4j/include/dsp/runtime/dsp_runtime_c.h");
        String runtime = source("libnd4j/include/legacy/impl/DspRuntimeC.cpp");
        for (String name : new String[]{"sdxRunSteadyState", "sdxRunSteadyStateAllocating"}) {
            assertTrue(header.contains("SDX_API sdx_status_t " + name + "("));
            assertTrue(runtime.contains("SDX_API sdx_status_t " + name + "("));
        }
        assertTrue(runtime.contains("? executeSteadyStatePlan(context->plan_handle, context->graph_context, execStream)"));
        assertTrue(runtime.contains("options, true, nullptr, true)"));
        assertTrue(runtime.contains("options, false, nullptr, true)"));
        String sdx = source(BACKENDS + "nd4j-sdx-preset/src/main/java/org/nd4j/dsp/runtime/presets/SdxRuntimePresets.java");
        assertTrue(sdx.contains("include = \"dsp/runtime/dsp_runtime_c.h\""));
        assertFalse(sdx.contains("NativeOpsDsp.h"));
        String liteRt = source(BACKENDS + "nd4j-sdx-preset/src/main/java/org/nd4j/dsp/runtime/presets/LiteRtLmPresets.java");
        assertTrue(liteRt.contains("include = \"c/engine.h\""));
        assertFalse(liteRt.contains("NativeOpsDsp.h"));
        String tokenizers = source("nd4j/nd4j-tokenizers/tokenizers-native-preset/src/main/java/org/eclipse/deeplearning4j/tokenizers/presets/TokenizersPresets.java");
        assertTrue(tokenizers.contains("\"tokenizers_c.h\""));
        assertFalse(tokenizers.contains("NativeOpsDsp.h"));
    }
}
