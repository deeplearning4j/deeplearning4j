/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HexagonAotCliTest {
    @TempDir
    Path temporary;

    @Test
    void plansDeterministicSortedRequestWithUnsignedShapeKey() throws Exception {
        Path segments = temporary.resolve("segments.json");
        Path first = temporary.resolve("first.json");
        Path second = temporary.resolve("second.json");
        Files.writeString(segments, "["
                + "{\"shapeKeyStatus\":\"STABLE\",\"shapeKey\":7,"
                + "\"numOps\":1,\"isCapturable\":false,\"index\":2,"
                + "\"endSlot\":9,\"compilationFailed\":false,\"startSlot\":9},"
                + "{\"ops\":{\"matmul\":1,\"add\":1},"
                + "\"shapeKeyStatus\":\"STABLE\",\"shapeKey\":-1,"
                + "\"numOps\":2,\"isCapturable\":true,\"index\":1,"
                + "\"endSlot\":3,\"compilationFailed\":false,\"startSlot\":2}"
                + "]", StandardCharsets.UTF_8);

        assertEquals(0, run("hexagon-plan", "--segments-json", segments.toString(),
                "--soc", "SM8650", "--model-id", "local-chat",
                "--output", first.toString()));
        assertEquals(0, run("hexagon-plan", "--segments-json", segments.toString(),
                "--soc", "SM8650", "--model-id", "local-chat",
                "--output", second.toString()));

        byte[] request = Files.readAllBytes(first);
        assertEquals(new String(request, StandardCharsets.UTF_8),
                Files.readString(second, StandardCharsets.UTF_8));
        String json = new String(request, StandardCharsets.UTF_8);
        assertTrue(json.endsWith("\n"));
        assertTrue(json.indexOf("\"adapterAbi\"") < json.indexOf("\"cacheAbi\""));
        assertTrue(json.contains("\"shapeKey\": 18446744073709551615"));
        assertTrue(json.contains("\"shapeKeyHex\": \"ffffffffffffffff\""));
        assertTrue(json.contains("\"artifact\": \"hexagon_2_3_ffffffffffffffff.bin\""));
        assertTrue(json.contains("\"reasons\": [\n        \"not-capturable\""));
        assertTrue(json.indexOf("\"add\"") < json.indexOf("\"matmul\""));
    }

    @Test
    void finalizesAndVerifiesExactSidecarsAndRawRequestHash() throws Exception {
        Path request = planSingleSegment();
        Path kernels = temporary.resolve("kernels");
        Files.createDirectories(kernels);
        Path artifact = kernels.resolve("hexagon_2_3_ffffffffffffffff.bin");
        byte[] payload = "vendor-aot-test".getBytes(StandardCharsets.UTF_8);
        Files.write(artifact, payload);

        assertEquals(0, run("hexagon-finalize", "--request", request.toString(),
                "--kernel-dir", kernels.toString()));
        assertEquals(0, run("hexagon-verify", "--request", request.toString(),
                "--kernel-dir", kernels.toString()));

        String metadata = Files.readString(
                kernels.resolve("hexagon_2_3_ffffffffffffffff.meta"),
                StandardCharsets.UTF_8);
        assertEquals("cacheAbi=sdx-hexagon-aot-v1\n"
                + "adapterAbi=1\n"
                + "soc=SM8650\n"
                + "rangeSemantics=inclusive\n"
                + "startSlot=2\n"
                + "endSlot=3\n"
                + "shapeKey=ffffffffffffffff\n"
                + "byteSize=15\n"
                + "sha256=" + sha256(payload) + "\n", metadata);

        String manifest = Files.readString(
                kernels.resolve("hexagon-aot-manifest.json"),
                StandardCharsets.UTF_8);
        assertTrue(manifest.contains("\"requestSha256\": \""
                + sha256(Files.readAllBytes(request)) + "\""));
        assertTrue(manifest.contains("\"sha256\": \"" + sha256(payload) + "\""));
    }

    @Test
    void verificationRejectsTamperingAndUnexpectedKernels() throws Exception {
        Path request = planSingleSegment();
        Path kernels = temporary.resolve("kernels");
        Files.createDirectories(kernels);
        Path artifact = kernels.resolve("hexagon_2_3_ffffffffffffffff.bin");
        Files.writeString(artifact, "vendor-aot-test", StandardCharsets.UTF_8);
        HexagonAot.finalizeArtifacts(request, kernels);

        Files.writeString(artifact, "tampered", StandardCharsets.UTF_8);
        IOException tampered = assertThrows(
                IOException.class, () -> HexagonAot.verify(request, kernels));
        assertTrue(tampered.getMessage().contains("metadata mismatch"));

        Files.writeString(artifact, "vendor-aot-test", StandardCharsets.UTF_8);
        Files.writeString(kernels.resolve("hexagon_4_4_0000000000000007.bin"),
                "unexpected", StandardCharsets.UTF_8);
        IOException unexpected = assertThrows(
                IOException.class, () -> HexagonAot.verify(request, kernels));
        assertTrue(unexpected.getMessage().contains("unexpected Hexagon AOT files"));
        Files.delete(kernels.resolve("hexagon_4_4_0000000000000007.bin"));

        Files.writeString(kernels.resolve("hexagon_4_4_0000000000000007.meta"),
                "unexpected", StandardCharsets.UTF_8);
        IOException unexpectedMetadata = assertThrows(
                IOException.class, () -> HexagonAot.verify(request, kernels));
        assertTrue(unexpectedMetadata.getMessage().contains("unexpected Hexagon AOT files"));
        Files.delete(kernels.resolve("hexagon_4_4_0000000000000007.meta"));

        Files.createDirectory(kernels.resolve("unexpected-directory"));
        IOException nonRegular = assertThrows(
                IOException.class, () -> HexagonAot.verify(request, kernels));
        assertTrue(nonRegular.getMessage().contains("not a regular file"));
        Files.delete(kernels.resolve("unexpected-directory"));

        Path manifest = kernels.resolve("hexagon-aot-manifest.json");
        Files.writeString(
                manifest,
                Files.readString(manifest, StandardCharsets.UTF_8)
                        .replace(sha256(Files.readAllBytes(request)), "0".repeat(64)),
                StandardCharsets.UTF_8);
        IOException staleManifest = assertThrows(
                IOException.class, () -> HexagonAot.verify(request, kernels));
        assertTrue(staleManifest.getMessage().contains("manifest does not match"));
    }

    @Test
    void rejectsArtifactAndMetadataPathTraversalBeforeFileAccess() throws Exception {
        Path request = planSingleSegment();
        Path kernels = temporary.resolve("kernels");
        Files.createDirectories(kernels);
        Files.writeString(
                temporary.resolve("outside.bin"), "outside", StandardCharsets.UTF_8);

        String validRequest = Files.readString(request, StandardCharsets.UTF_8);
        Path artifactTraversal = temporary.resolve("artifact-traversal.json");
        Files.writeString(
                artifactTraversal,
                validRequest.replace(
                        "\"artifact\": \"hexagon_2_3_ffffffffffffffff.bin\"",
                        "\"artifact\": \"../outside.bin\""),
                StandardCharsets.UTF_8);
        IOException artifactFailure = assertThrows(
                IOException.class,
                () -> HexagonAot.finalizeArtifacts(artifactTraversal, kernels));
        assertTrue(artifactFailure.getMessage().contains("artifact and metadata names"));

        Path validArtifact = kernels.resolve("hexagon_2_3_ffffffffffffffff.bin");
        Files.writeString(validArtifact, "vendor-aot-test", StandardCharsets.UTF_8);
        Path metadataTraversal = temporary.resolve("metadata-traversal.json");
        Files.writeString(
                metadataTraversal,
                validRequest.replace(
                        "\"metadata\": \"hexagon_2_3_ffffffffffffffff.meta\"",
                        "\"metadata\": \"../outside.meta\""),
                StandardCharsets.UTF_8);
        IOException metadataFailure = assertThrows(
                IOException.class,
                () -> HexagonAot.finalizeArtifacts(metadataTraversal, kernels));
        assertTrue(metadataFailure.getMessage().contains("artifact and metadata names"));
        assertFalse(Files.exists(temporary.resolve("outside.meta")));
    }

    @Test
    void rejectsInvalidRangesDuplicateArtifactsAndMalformedRequests() throws Exception {
        Path invalidRange = temporary.resolve("invalid-range.json");
        Files.writeString(invalidRange,
                "[{\"index\":0,\"startSlot\":2,\"endSlot\":3,\"numOps\":1,"
                        + "\"isCapturable\":true,\"shapeKey\":1,"
                        + "\"shapeKeyStatus\":\"STABLE\"}]",
                StandardCharsets.UTF_8);
        IOException range = assertThrows(IOException.class, () -> HexagonAot.plan(
                invalidRange, "SM8650", null, temporary.resolve("request.json"),
                false, false, false));
        assertTrue(range.getMessage().contains("expected 2"));

        Path duplicate = temporary.resolve("duplicate.json");
        Files.writeString(duplicate,
                "[{\"index\":0,\"startSlot\":2,\"endSlot\":2,\"numOps\":1,"
                        + "\"isCapturable\":true,\"shapeKey\":1,"
                        + "\"shapeKeyStatus\":\"STABLE\"},"
                        + "{\"index\":1,\"startSlot\":2,\"endSlot\":2,\"numOps\":1,"
                        + "\"isCapturable\":true,\"shapeKey\":1,"
                        + "\"shapeKeyStatus\":\"STABLE\"}]",
                StandardCharsets.UTF_8);
        IOException collision = assertThrows(IOException.class, () -> HexagonAot.plan(
                duplicate, "SM8650", null, temporary.resolve("request.json"),
                false, false, false));
        assertTrue(collision.getMessage().contains("duplicate range/shape"));

        Path malformed = temporary.resolve("malformed-request.json");
        Files.writeString(malformed,
                "{\"formatVersion\":2,\"cacheAbi\":\"sdx-hexagon-aot-v1\","
                        + "\"adapterAbi\":1,\"rangeSemantics\":\"inclusive\","
                        + "\"soc\":\"SM8650\",\"segments\":[]}",
                StandardCharsets.UTF_8);
        IOException version = assertThrows(IOException.class,
                () -> HexagonAot.verify(malformed, temporary));
        assertTrue(version.getMessage().contains("formatVersion"));
    }

    @Test
    void rejectsUnknownInapplicableAndRepeatedOptions() {
        IllegalArgumentException typo = assertThrows(
                IllegalArgumentException.class,
                () -> run("hexagon-plan", "--allow-unstabel"));
        assertTrue(typo.getMessage().contains("Unknown option for hexagon-plan"));

        IllegalArgumentException inapplicable = assertThrows(
                IllegalArgumentException.class,
                () -> run("hexagon-plan", "--cache", temporary.toString()));
        assertTrue(inapplicable.getMessage().contains("Unknown option for hexagon-plan"));

        IllegalArgumentException repeated = assertThrows(
                IllegalArgumentException.class,
                () -> run("hexagon-verify",
                        "--request", "first.json",
                        "--request", "second.json",
                        "--kernel-dir", "kernels"));
        assertTrue(repeated.getMessage().contains("only once"));
    }

    private Path planSingleSegment() throws Exception {
        Path segments = temporary.resolve("single-segments.json");
        Path request = temporary.resolve("single-request.json");
        Files.writeString(segments,
                "[{\"index\":0,\"startSlot\":2,\"endSlot\":3,\"numOps\":2,"
                        + "\"isCapturable\":true,\"compilationFailed\":false,"
                        + "\"shapeKey\":-1,\"shapeKeyStatus\":\"STABLE\","
                        + "\"ops\":{\"add\":1,\"matmul\":1}}]",
                StandardCharsets.UTF_8);
        HexagonAot.plan(segments, "SM8650", "test-model", request,
                false, false, false);
        return request;
    }

    private static int run(String... args) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ByteArrayOutputStream error = new ByteArrayOutputStream();
        return SdxCompilerCli.run(
                args,
                new PrintStream(output, true, StandardCharsets.UTF_8.name()),
                new PrintStream(error, true, StandardCharsets.UTF_8.name()));
    }

    private static String sha256(byte[] value) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(value);
        StringBuilder result = new StringBuilder();
        for (byte item : digest) {
            result.append(String.format("%02x", item & 0xff));
        }
        return result.toString();
    }
}
