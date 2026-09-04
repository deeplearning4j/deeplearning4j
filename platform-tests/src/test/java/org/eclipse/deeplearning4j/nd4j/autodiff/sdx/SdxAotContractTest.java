/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.sdx;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Pins the source contracts shared by GraalVM AOT Java, the SDX LLM C header,
 * the DSP C header, and the Kompile Android consumer. These boundaries are
 * necessarily cross-language; source-level contract tests make drift fail in
 * the producer build instead of during a phone import.
 */
class SdxAotContractTest {

    private static final List<String> PREPARED_RESULT_KEYS = List.of(
            "schema",
            "targetProfile",
            "cacheHit",
            "sourceSha256",
            "sourceBytes",
            "canonicalSdzLogicalSha256",
            "canonicalSdzLogicalBytes",
            "canonicalSdzPath",
            "canonicalSdzBytes",
            "modelPath",
            "tokenizerPath",
            "compileKey",
            "targetSoc",
            "contextLength",
            "maxPrefillLength",
            "executionProvider",
            "conversionProfileSha256",
            "diagnosticMode",
            "optimizedSourcePath",
            "optimizedSourceBytes",
            "importResourcesReleased"
    );

    private static final Pattern ENTRY_POINT = Pattern.compile(
            "@CEntryPoint\\(name\\s*=\\s*\"([^\"]+)\"");
    private static final Pattern C_FUNCTION = Pattern.compile(
            "\\b(sdx(?:Llm|Vlm|Audio)[A-Za-z0-9_]*)\\s*\\(");
    private static final Path PROJECT_ROOT = locateProjectRoot();
    private static final Path SDX_AOT_ROOT = PROJECT_ROOT.resolve("nd4j/sdx-aot");
    private static final Path AOT_API = SDX_AOT_ROOT.resolve(
            "src/main/java/org/eclipse/deeplearning4j/sdx/aot/SdxLlmCApi.java");
    private static final Path PREPARER = SDX_AOT_ROOT.resolve(
            "src/main/java/org/eclipse/deeplearning4j/sdx/aot/SdxGgufModelPreparer.java");
    private static final Path LLM_HEADER = SDX_AOT_ROOT.resolve("include/sdx_llm_c.h");
    private static final Path DSP_HEADER = PROJECT_ROOT.resolve(
            "libnd4j/include/dsp/runtime/dsp_runtime_c.h");

    @Test
    void llmCHeaderMatchesGraalEntryPointsAbiAndDspStatusCodes() throws IOException {
        String javaApi = Files.readString(AOT_API);
        String llmHeader = Files.readString(LLM_HEADER);
        String dspHeader = Files.readString(DSP_HEADER);
        Set<String> javaNames = annotatedEntryPoints(javaApi);
        Set<String> headerNames = declaredHeaderFunctions(Files.readAllLines(LLM_HEADER));

        assertEquals(21, javaNames.size(), "Unexpected Java C-entry-point count");
        assertEquals(javaNames, headerNames,
                "sdx_llm_c.h and SdxLlmCApi @CEntryPoint declarations diverged");
        assertEquals(javaIntegerValue(javaApi, "ABI_VERSION"),
                integerValue(llmHeader, "SDX_LLM_ABI_VERSION"));

        Map<String, Integer> javaStatuses = new LinkedHashMap<>();
        javaStatuses.put("OK", javaIntegerValue(javaApi, "OK"));
        javaStatuses.put("INVALID_ARGUMENT", javaIntegerValue(javaApi, "INVALID_ARGUMENT"));
        javaStatuses.put("MODEL_LOAD_FAILED", javaIntegerValue(javaApi, "MODEL_LOAD_FAILED"));
        javaStatuses.put("EXECUTION_FAILED", javaIntegerValue(javaApi, "EXECUTION_FAILED"));
        javaStatuses.put("IO_ERROR", javaIntegerValue(javaApi, "IO_ERROR"));
        for (Map.Entry<String, Integer> status : javaStatuses.entrySet()) {
            int llmValue = integerValue(llmHeader, "SDX_LLM_STATUS_" + status.getKey());
            int dspValue = integerValue(dspHeader, "SDX_STATUS_" + status.getKey());
            assertEquals(status.getValue().intValue(), llmValue,
                    () -> "Graal entry point and LLM header diverged: " + status.getKey());
            assertEquals(dspValue, llmValue,
                    () -> "LLM and DSP C status codes diverged: " + status.getKey());
        }
    }

    @Test
    void androidPreparationAbiMatchesAotProducer() throws IOException {
        Path options = locateAndroidMainSource(
                "ai/kompile/chat/local/android/model/ModelPreparationOptions.kt");
        Path nativeApi = locateAndroidMainSource(
                "ai/kompile/chat/local/android/model/SdxAndroidLlmNative.java");
        assumeTrue(options != null && nativeApi != null,
                "Kompile Android sibling checkout is not present");

        String optionSource = Files.readString(options);
        String nativeSource = Files.readString(nativeApi);
        String preparerSource = Files.readString(PREPARER);
        assertEquals(stringValue(preparerSource, "GRAPH_IMPORT_ABI"),
                stringValue(optionSource, "MODEL_PREPARATION_GRAPH_IMPORT_ABI"));
        assertEquals(integerValue(Files.readString(LLM_HEADER), "SDX_LLM_ABI_VERSION"),
                javaIntegerValue(nativeSource, "SDX_LLM_ABI_VERSION"));
    }

    @Test
    void preparedResultV6MatchesAndroidConsumer() throws IOException {
        Path contractPath = locateAndroidMainSource(
                "ai/kompile/chat/local/android/model/SdxRawGgufContract.kt");
        assumeTrue(contractPath != null, "Kompile Android sibling checkout is not present");

        String contract = Files.readString(contractPath);
        String llmHeader = Files.readString(LLM_HEADER);
        String preparedMethod = sourceBetween(
                Files.readString(PREPARER),
                "private static String preparedJson(",
                "private static Path readCanonicalPointer(");
        Set<String> emitted = resultPutKeys(preparedMethod);
        assertEquals("sdx-prepared-text-model-v6",
                stringValue(Files.readString(PREPARER), "PREPARED_SCHEMA"));
        assertEquals("sdx-prepared-text-model-v6",
                stringValue(contract, "PREPARED_SCHEMA"));
        assertTrue(llmHeader.contains("return sdx-prepared-text-model-v6 JSON"),
                "Public sdx_llm_c.h documents a stale prepared-result schema");

        List<String> missingAndroidKeys = new ArrayList<>();
        List<String> missingProducerKeys = new ArrayList<>();
        for (String key : PREPARED_RESULT_KEYS) {
            Pattern declaration = Pattern.compile(
                    "const val [A-Z_0-9]+_FIELD\\s*=\\s*\"" + Pattern.quote(key) + "\"");
            if (!declaration.matcher(contract).find()) {
                missingAndroidKeys.add(key);
            }
            if (!emitted.contains(key)) {
                missingProducerKeys.add(key);
            }
        }
        assertTrue(missingAndroidKeys.isEmpty(),
                () -> "Android result contract is missing: " + missingAndroidKeys);
        assertTrue(missingProducerKeys.isEmpty(),
                () -> "AOT preparedJson no longer emits: " + missingProducerKeys);
    }

    @Test
    void resolvedBundleEnvelopeKeepsItsDocumentedFields() throws IOException {
        String resolveMethod = sourceBetween(
                Files.readString(PREPARER),
                "static String resolve(",
                "private static Path configureTemporaryDirectory(");
        Set<String> emitted = resultPutKeys(resolveMethod);
        assertTrue(emitted.containsAll(List.of(
                "schema",
                "targetProfile",
                "modelPath",
                "tokenizerPath",
                "tokenizerConfigPath",
                "textGenerationConfigPath",
                "compileKey",
                "compilerId",
                "executionProvider"
        )), () -> "Resolved-bundle envelope is incomplete: " + emitted);
    }

    private static Path locateProjectRoot() {
        Path cwd = Path.of("").toAbsolutePath();
        for (Path cursor = cwd; cursor != null; cursor = cursor.getParent()) {
            if (Files.isRegularFile(cursor.resolve("nd4j/sdx-aot/include/sdx_llm_c.h"))
                    && Files.isRegularFile(cursor.resolve(
                    "libnd4j/include/dsp/runtime/dsp_runtime_c.h"))) {
                return cursor;
            }
        }
        throw new IllegalStateException("Could not locate the deeplearning4j project root");
    }

    private static Path locateAndroidMainSource(String relativePath) {
        for (Path cursor = PROJECT_ROOT; cursor != null; cursor = cursor.getParent()) {
            for (Path repository : new Path[]{cursor, cursor.resolve("kompile")}) {
                Path candidate = repository.resolve(
                        "kompile-chat-local/mobile/android/app/src/main/java").resolve(relativePath);
                if (Files.isRegularFile(candidate)) {
                    return candidate;
                }
            }
        }
        return null;
    }

    private static Set<String> annotatedEntryPoints(String source) {
        Set<String> names = new LinkedHashSet<>();
        Matcher matcher = ENTRY_POINT.matcher(source);
        while (matcher.find()) {
            String name = matcher.group(1);
            assertTrue(names.add(name), () -> "Duplicate @CEntryPoint name: " + name);
        }
        return names;
    }

    private static Set<String> declaredHeaderFunctions(List<String> lines) {
        Set<String> names = new LinkedHashSet<>();
        StringBuilder declaration = null;
        for (String line : lines) {
            if (declaration == null && line.stripLeading().startsWith("SDX_LLM_API ")) {
                declaration = new StringBuilder();
            }
            if (declaration == null) {
                continue;
            }
            declaration.append(' ').append(line.trim());
            if (!line.contains(";")) {
                continue;
            }
            String declarationText = declaration.toString();
            Matcher matcher = C_FUNCTION.matcher(declarationText);
            assertTrue(matcher.find(),
                    () -> "Malformed SDX_LLM_API declaration: " + declarationText);
            String name = matcher.group(1);
            assertTrue(names.add(name), () -> "Duplicate C header function: " + name);
            declaration = null;
        }
        assertTrue(declaration == null, "Unterminated SDX_LLM_API declaration");
        return names;
    }

    private static Set<String> resultPutKeys(String source) {
        Set<String> keys = new LinkedHashSet<>();
        Matcher matcher = Pattern.compile("result\\.put\\(\"([^\"]+)\"").matcher(source);
        while (matcher.find()) {
            keys.add(matcher.group(1));
        }
        return keys;
    }

    private static String sourceBetween(String source, String startMarker, String endMarker) {
        int start = source.indexOf(startMarker);
        int end = start < 0 ? -1 : source.indexOf(endMarker, start);
        assertTrue(start >= 0, () -> "Missing source marker: " + startMarker);
        assertTrue(end > start, () -> "Missing source boundary: " + endMarker);
        return source.substring(start, end);
    }

    private static int javaIntegerValue(String source, String name) {
        Matcher matcher = Pattern.compile(
                "(?:public\\s+)?static\\s+final\\s+int\\s+" + Pattern.quote(name)
                        + "\\s*=\\s*([0-9]+)").matcher(source);
        assertTrue(matcher.find(), () -> "Missing Java integer contract value: " + name);
        return Integer.parseInt(matcher.group(1));
    }

    private static int integerValue(String source, String name) {
        Matcher matcher = Pattern.compile(
                "\\b" + Pattern.quote(name) + "\\s*=\\s*([0-9]+)").matcher(source);
        assertTrue(matcher.find(), () -> "Missing integer contract value: " + name);
        return Integer.parseInt(matcher.group(1));
    }

    private static String stringValue(String source, String name) {
        Matcher matcher = Pattern.compile(
                "\\b" + Pattern.quote(name) + "\\s*=\\s*\"([^\"]+)\"").matcher(source);
        assertTrue(matcher.find(), () -> "Missing string contract value: " + name);
        return matcher.group(1);
    }
}
