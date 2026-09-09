/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Executes the production hash/guard with a fixture file inventory, never invoking Git. */
@Tag("source-lint")
class AndroidSourceManifestDiagnosticTest {
    @TempDir Path temp;

    private String source() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().getParent();
        return Files.readString(root.resolve("libnd4j/tools/mobile/build-android-accelerator.sh"));
    }

    private String functions() throws Exception {
        String text = source();
        Path consumer = Path.of("").toAbsolutePath().normalize().getParent().getParent()
                .resolve("kompile/kompile-chat-local/mobile/android/tools/build-offline-accelerators.sh");
        String packager = Files.readString(consumer);
        return text.substring(text.indexOf("source_tree_manifest_sha256() {"),
                text.indexOf("native_source_manifest_sha256() {"))
                + packager.substring(packager.indexOf("dl4j_source_manifest_sha256() {"),
                        packager.indexOf("dl4j_aot_source_manifest_sha256() {"));
    }

    private Path file(String name, String content) throws Exception {
        Path path = temp.resolve(name);
        Files.createDirectories(path.getParent());
        Files.writeString(path, content);
        Files.setPosixFilePermissions(path, PosixFilePermissions.fromString("rw-r--r--"));
        return path;
    }

    private Result run(String body, String... arguments) throws Exception {
        String script = "set -euo pipefail\nREPO_ROOT=$1; shift\n"
                + "sha256_file() { sha256sum -- \"$1\" | cut -d ' ' -f 1; }\n"
                + "git() { printf '%s\\0' \"${FILES[@]}\"; }\n"
                + functions() + body;
        List<String> command = new ArrayList<>(List.of("bash", "-c", script, "fixture", temp.toString()));
        command.addAll(List.of(arguments));
        Path output = Files.createTempFile(temp, "output-", ".txt");
        Process process = new ProcessBuilder(command).redirectErrorStream(true)
                .redirectOutput(output.toFile()).start();
        if (!process.waitFor(20, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            fail("Manifest fixture exceeded 20 seconds");
        }
        return new Result(process.exitValue(), Files.readString(output));
    }

    private String hash(Path diagnostic, String... names) throws Exception {
        List<String> arguments = new ArrayList<>();
        arguments.add(diagnostic == null ? "" : diagnostic.toString());
        arguments.addAll(List.of(names));
        Result result = run("diagnostic=$1; shift; FILES=(\"$@\")\n"
                + "source_tree_manifest_sha256 \"$diagnostic\" libnd4j\n"
                + "DL4J_ROOT=$REPO_ROOT; dl4j_source_manifest_sha256\n",
                arguments.toArray(String[]::new));
        assertEquals(0, result.code(), result.output());
        String[] hashes = result.output().trim().split("\\R");
        assertEquals(2, hashes.length, result.output());
        assertEquals(hashes[0], hashes[1], "Producer and APK verifier must use the same source manifest");
        return hashes[0];
    }

    @Test
    void diagnosticsPreserveExistingNulDelimitedDigestAndEscapePaths() throws Exception {
        String[] names = {"libnd4j/a space.txt", "libnd4j/b\nline.txt"};
        MessageDigest expected = MessageDigest.getInstance("SHA-256");
        for (String name : names) {
            byte[] bytes = Files.readAllBytes(file(name, "same"));
            String digest = hex(MessageDigest.getInstance("SHA-256").digest(bytes));
            // GNU sha256sum prefixes escaped filename output with a backslash.
            // Preserve this existing sha256_file/cut behavior in the reference.
            if (name.contains("\n") || name.contains("\\")) {
                digest = "\\" + digest;
            }
            expected.update((name + "\0" + "644\0" + digest + "\0").getBytes(StandardCharsets.UTF_8));
        }
        Path diagnostic = temp.resolve("before.txt");
        String actual = hash(diagnostic, names);
        assertEquals(hex(expected.digest()), actual);
        assertEquals(actual, hash(null, names));
        assertEquals(2, Files.readAllLines(diagnostic).size(), "Newlines in paths must be escaped");
        assertTrue(Files.readString(diagnostic).contains("a\\ space.txt"));
        assertTrue(Files.readString(diagnostic).contains("b\\nline.txt"));
        hash(diagnostic, names);
        assertEquals(2, Files.readAllLines(diagnostic).size(), "Snapshots must replace, not append");
    }

    @Test
    void contentModeAdditionAndDeletionRemainReceiptInvalidating() throws Exception {
        String name = "libnd4j/source.cpp";
        Path input = file(name, "original");
        String before = hash(temp.resolve("before.txt"), name);
        Files.writeString(input, "changed");
        assertNotEquals(before, hash(temp.resolve("content.txt"), name));
        Files.writeString(input, "original");
        Files.setPosixFilePermissions(input, PosixFilePermissions.fromString("rwxr-xr-x"));
        assertNotEquals(before, hash(temp.resolve("mode.txt"), name));
        Files.setPosixFilePermissions(input, PosixFilePermissions.fromString("rw-r--r--"));
        file("libnd4j/added.cpp", "added");
        assertNotEquals(before, hash(temp.resolve("added.txt"), name, "libnd4j/added.cpp"));
        Files.delete(input);
        assertNotEquals(before, hash(temp.resolve("deleted.txt"), name));
        assertEquals("", Files.readString(temp.resolve("deleted.txt")));
    }

    @Test
    void cudaOnlyTranslationUnitsDoNotInvalidateAndroidReceipts() throws Exception {
        String cpu = "libnd4j/include/graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp";
        String graph = "libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cu";
        String helper = "libnd4j/include/ops/declarable/helpers/cuda/ggml_qmatmul.cu";
        file(cpu, "android input");
        Path graphFile = file(graph, "cuda original");
        file(helper, "cuda original");
        String before = hash(null, cpu);
        Path diagnostic = temp.resolve("android.txt");
        assertEquals(before, hash(diagnostic, cpu, graph, helper));
        assertEquals(1, Files.readAllLines(diagnostic).size());
        Files.writeString(graphFile, "cuda changed during Android build");
        assertEquals(before, hash(diagnostic, cpu, graph, helper));
        Files.delete(graphFile);
        assertEquals(before, hash(null, cpu, graph, helper));
    }

    @Test
    void sharedHeadersCpuSourcesAndBuildInputsStillInvalidateAndroidReceipts() throws Exception {
        String[] names = {
                "libnd4j/include/graph/impl/NativeDynamicShapePlan.h",
                "libnd4j/include/ops/declarable/helpers/cuda/shared.cuh",
                "libnd4j/include/ops/declarable/helpers/cuda/generated.cu.in",
                "libnd4j/include/ops/declarable/helpers/cpu/ggml_qmatmul.cpp",
                "libnd4j/include/graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp",
                "libnd4j/include/helpers/shape.h",
                "libnd4j/cmake/MainBuildFlow.cmake",
                "libnd4j/tools/mobile/build-android-accelerator.sh"
        };
        for (String name : names) {
            Path input = file(name, "original");
            String before = hash(null, name);
            Files.writeString(input, "changed");
            assertNotEquals(before, hash(null, name), name);
        }
    }

    @Test
    void sourceGuardStillFailsClosedAndReportsExactChangedPath() throws Exception {
        String name = "libnd4j/changed.cpp";
        Path input = file(name, "original");
        Path beforeFile = temp.resolve("before.txt");
        String before = hash(beforeFile, name);
        Files.writeString(input, "changed");
        Path afterFile = temp.resolve("after.txt");
        String after = hash(afterFile, name);
        String text = source();
        String guard = text.substring(text.indexOf("[[ \"$CURRENT_SOURCE_MANIFEST_SHA256\""),
                text.indexOf("FRESH_JAVA_BUILDS=\"$FINAL_AAR.fresh-java-builds\""));
        String bindings = "SOURCE_MANIFEST_SHA256=$1; CURRENT_SOURCE_MANIFEST_SHA256=$2; "
                + "SOURCE_MANIFEST_BEFORE=$3; SOURCE_MANIFEST_AFTER=$4\n";
        Result changed = run(bindings + guard + "printf 'RECEIPT_ALLOWED\\n'\n",
                before, after, beforeFile.toString(), afterFile.toString());
        assertEquals(1, changed.code(), changed.output());
        assertTrue(changed.output().contains("Source tree changed during Android accelerator build"));
        assertTrue(changed.output().contains(name), changed.output());
        assertFalse(changed.output().contains("RECEIPT_ALLOWED"));
        Result stable = run(bindings + guard + "printf 'RECEIPT_ALLOWED\\n'\n",
                before, before, beforeFile.toString(), beforeFile.toString());
        assertEquals(0, stable.code(), stable.output());
        assertTrue(stable.output().contains("RECEIPT_ALLOWED"));
    }

    @Test
    void aotGuardRetainsExactInventoriesAndFailsClosedOnDrift() throws Exception {
        String before = "src/a space.java\0" + "100644\0old\0"
                + "src/deleted.java\0" + "100644\0deleted\0";
        String after = "src/a space.java\0" + "100755\0new\0"
                + "src/added\nline.java\0" + "100644\0added\0";
        Result result = runAotGuard(before, after);
        assertEquals(3, result.code(), result.output());
        assertTrue(result.output().contains("DL4J AOT source tree changed during the build"));
        assertTrue(result.output().contains("src/a\\ space.java"), result.output());
        assertTrue(result.output().contains("src/added\\nline.java"), result.output());
        assertTrue(result.output().contains("src/deleted.java"), result.output());
        assertFalse(result.output().contains("RECEIPT_ALLOWED"));
        try (var paths = Files.list(temp)) {
            Path diagnostics = paths.filter(p -> p.getFileName().toString().startsWith("source-drift."))
                    .findFirst().orElseThrow();
            assertArrayEquals(before.getBytes(StandardCharsets.UTF_8), Files.readAllBytes(diagnostics.resolve("before.nul")));
            assertArrayEquals(after.getBytes(StandardCharsets.UTF_8), Files.readAllBytes(diagnostics.resolve("after.nul")));
            assertEquals(2, Files.readAllLines(diagnostics.resolve("after.txt")).size());
        }
    }

    @Test
    void aotGuardPreservesDigestAndAcceptsUnchangedInventory() throws Exception {
        String inventory = "src/input.java\0" + "100644\0abc\0";
        Result result = runAotGuard(inventory, inventory);
        assertEquals(0, result.code(), result.output());
        assertTrue(result.output().contains("RECEIPT_ALLOWED"), result.output());
        try (var paths = Files.list(temp)) {
            assertFalse(paths.anyMatch(p -> p.getFileName().toString().startsWith("source-drift.")));
        }
    }

    private Result runAotGuard(String before, String after) throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().getParent();
        String builder = Files.readString(root.resolve("nd4j/sdx-aot/src/main/android/build-android-aot-sdk.sh"));
        String capture = builder.substring(builder.indexOf("SOURCE_MANIFEST_BEFORE=\"$BUILD_ROOT/source-before.nul\""),
                builder.indexOf("BASE_SDK_RECEIPT_SHA256=not-applicable"));
        String guard = builder.substring(builder.indexOf("SOURCE_MANIFEST_AFTER=\"$BUILD_ROOT/source-after.nul\""),
                builder.indexOf("if [[ -f \"$BASE_SDK_INPUT\" ]]; then", builder.indexOf("SOURCE_MANIFEST_AFTER=")));
        Path beforeFile = file("input-before.nul", before);
        Path afterFile = file("input-after.nul", after);
        String expected = hex(MessageDigest.getInstance("SHA-256").digest(before.getBytes(StandardCharsets.UTF_8)));
        return run("WORK_DIR=$REPO_ROOT; BUILD_ROOT=$WORK_DIR/generation; mkdir \"$BUILD_ROOT\"\n"
                + "DL4J_ROOT=$REPO_ROOT; DL4J_AOT_SOURCE_ROOTS=(src); inventory=$1\n"
                + "sdx_git_source_manifest() { cat -- \"$inventory\"; }\n"
                + "fail() { printf '%s\\n' \"$*\" >&2; exit 3; }\n"
                + capture + "[[ $SOURCE_MANIFEST_SHA256 == $3 ]]\ninventory=$2\n"
                + guard + "printf 'RECEIPT_ALLOWED\\n'\n", beforeFile.toString(), afterFile.toString(), expected);
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(Character.forDigit((value & 0xff) >>> 4, 16));
            result.append(Character.forDigit(value & 0x0f, 16));
        }
        return result.toString();
    }

    private static final class Result {
        private final int code;
        private final String output;

        private Result(int code, String output) {
            this.code = code;
            this.output = output;
        }

        int code() {
            return code;
        }

        String output() {
            return output;
        }
    }
}
