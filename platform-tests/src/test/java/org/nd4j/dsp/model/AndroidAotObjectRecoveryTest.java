/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Runs the production cache selection/validation with tiny objects, not Graal. */
@Tag("source-lint")
class AndroidAotObjectRecoveryTest {
    @TempDir Path temp;

    private String run(String setup, String checks) throws Exception {
        return run(setup, checks, 0);
    }

    private String run(String setup, String checks, int expectedExit) throws Exception {
        Path android = Path.of("").toAbsolutePath().normalize().getParent()
                .resolve("nd4j/sdx-aot/src/main/android");
        String builder = Files.readString(android.resolve("build-android-aot-sdk.sh"));
        String functions = builder.substring(builder.indexOf("validate_native_image_object() {"),
                builder.indexOf("OBJECT_REUSED=0\n"));
        String selection = builder.substring(builder.indexOf("OBJECT_REUSED=0\n"),
                builder.indexOf("if [[ \"$OBJECT_REUSED\" == 0 ]]; then"));
        String script = "set -euo pipefail\ncd \"$1\"\ntrap 'chmod -R u+w \"$PWD\"' EXIT\n"
                + "fail() { printf '%s\\n' \"$*\" >&2; exit 3; }\n"
                + "sha256_file() { sha256sum -- \"$1\" | cut -d ' ' -f 1; }\n"
                + "receipt_has() { grep -Fxq -- \"$2\" \"$1\"; }\n"
                + "sdx_native_image_object_identity_lines() { printf 'format=test\\n'; }\n"
                + Files.readString(android.resolve("native-image-cache.sh"))
                + "\nSDX_NATIVE_CACHE=1; SDX_NATIVE_FORCE_REBUILD=0; SDX_NATIVE_CACHE_DIR=$PWD/shared; sdx_native_cache_configure\n"
                + "OBJECT_STAGE_INPUTS_SHA256=$(printf key | sha256sum | cut -d ' ' -f 1)\n"
                + "OBJECT_STAGES_DIR=$PWD/stages; OBJECT_STAGE_DIR=$OBJECT_STAGES_DIR/$OBJECT_STAGE_INPUTS_SHA256\n"
                + "BUILD_ROOT=$PWD/build; OBJECT=$BUILD_ROOT/libsdx_llm.o\n"
                + "OBJECT_CACHE_TARGET=android-sdx-aot-object-v5; OBJECT_CACHE_ARTIFACT=libsdx_llm.o\n"
                + "mkdir -p \"$OBJECT_STAGES_DIR\" \"$BUILD_ROOT\"\n"
                // The cache selection is real; only the external ELF decoder is a fixture.
                + "printf '#!/bin/bash\\nprintf \"Machine: AArch64\\\\n\"\\n' > readelf-fixture\n"
                + "chmod +x readelf-fixture; LLVM_READELF=$PWD/readelf-fixture\n"
                + "printf original > original.o\nGOOD_SHA=$(sha256_file original.o)\n"
                + "make_local() { mkdir -p \"$OBJECT_STAGE_DIR\"; cp original.o \"$OBJECT_STAGE_DIR/libsdx_llm.o\"; "
                + "printf 'format=test\\nobject_stage_inputs_sha256=%s\\nobject_sha256=%s\\n' "
                + "\"$OBJECT_STAGE_INPUTS_SHA256\" \"$GOOD_SHA\" > \"$OBJECT_STAGE_DIR/build-receipt\"; }\n"
                + functions + setup + "\n" + selection + checks + "\nprintf 'CHECKS_PASSED\\n'\n";
        Path output = temp.resolve("output.txt");
        Process p = new ProcessBuilder("bash", "-c", script, "fixture", temp.toString())
                .redirectErrorStream(true).redirectOutput(output.toFile()).start();
        if (!p.waitFor(20, TimeUnit.SECONDS)) {
            p.destroyForcibly();
            fail("Cache fixture exceeded 20 seconds");
        }
        String result = Files.readString(output);
        assertEquals(expectedExit, p.exitValue(), result);
        if (expectedExit == 0) {
            assertTrue(result.contains("CHECKS_PASSED"), result);
        }
        return result;
    }

    @Test
    void corruptLocalFallsBackToVerifiedSharedAndPreservesEvidence() throws Exception {
        String output = run("make_local\nprintf damaged > \"$OBJECT_STAGE_DIR/libsdx_llm.o\"\n"
                        + "chmod -R a-w \"$OBJECT_STAGE_DIR\"\nLOCAL_MODE=$(stat -c '%a' \"$OBJECT_STAGE_DIR\")\n"
                        + "sdx_native_cache_publish \"$OBJECT_CACHE_TARGET\" \"$OBJECT_STAGE_INPUTS_SHA256\" libsdx_llm.o original.o\n",
                "[[ $OBJECT_REUSED == 1 && ! -e $OBJECT_STAGE_DIR ]]\n"
                        + "[[ $(sha256_file \"$OBJECT\") == $GOOD_SHA ]]\n"
                        + "bad=(\"$OBJECT_STAGES_DIR.invalid\"/*/stage)\n"
                        + "[[ ${#bad[@]} == 1 ]]\n"
                        + "[[ $(stat -c '%a' \"${bad[0]}\") == $LOCAL_MODE ]]\n"
                        + "[[ $(stat -c '%a' \"${bad[0]}/libsdx_llm.o\") == 444 ]]\n"
                        + "[[ $(sha256_file \"${bad[0]}/libsdx_llm.o\") != $GOOD_SHA ]]\n"
                        + "receipt_has \"${bad[0]}/build-receipt\" \"object_sha256=$GOOD_SHA\"\n");
        assertTrue(output.contains("invalid local Native Image object stage"));
        assertTrue(output.contains("CACHE HIT: restored native artifact"));
    }

    @Test
    void validLocalRemainsUntouched() throws Exception {
        String output = run("make_local\nchmod -R a-w \"$OBJECT_STAGE_DIR\"\n",
                "[[ $OBJECT_REUSED == 1 && -d $OBJECT_STAGE_DIR ]]\n"
                        + "[[ ! -e $OBJECT_STAGES_DIR.invalid ]]\n"
                        + "[[ $(sha256_file \"$OBJECT\") == $GOOD_SHA ]]\n");
        assertTrue(output.contains("reusing validated local"));
    }

    @Test
    void corruptBothCachesRequireRebuildNotAcceptance() throws Exception {
        run("make_local\nprintf damaged > \"$OBJECT_STAGE_DIR/libsdx_llm.o\"\n"
                        + "chmod -R a-w \"$OBJECT_STAGE_DIR\"\nLOCAL_MODE=$(stat -c '%a' \"$OBJECT_STAGE_DIR\")\n"
                        + "sdx_native_cache_publish \"$OBJECT_CACHE_TARGET\" \"$OBJECT_STAGE_INPUTS_SHA256\" libsdx_llm.o original.o\n"
                        + "shared=$(sdx_native_cache_artifact_path \"$OBJECT_CACHE_TARGET\" \"$OBJECT_STAGE_INPUTS_SHA256\" libsdx_llm.o)\n"
                        + "chmod u+w \"$shared\"; printf damaged > \"$shared\"\n",
                "[[ $OBJECT_REUSED == 0 && ! -e $OBJECT ]]\n");
    }

    @Test
    void absentCachesRequireRebuildWithoutQuarantine() throws Exception {
        run("", "[[ $OBJECT_REUSED == 0 && ! -e $OBJECT && ! -e $OBJECT_STAGES_DIR.invalid ]]\n");
    }

    @Test
    void symlinkedQuarantineFailsWithoutMovingEvidence() throws Exception {
        String output = run("make_local\nprintf damaged > \"$OBJECT_STAGE_DIR/libsdx_llm.o\"\n"
                        + "mkdir elsewhere; ln -s \"$PWD/elsewhere\" \"$OBJECT_STAGES_DIR.invalid\"\n",
                "", 3);
        assertTrue(output.contains("quarantine must not be a symlink"), output);
        try (var entries = Files.list(temp.resolve("elsewhere"))) {
            assertEquals(0, entries.count());
        }
        try (var entries = Files.list(temp.resolve("stages"))) {
            assertEquals(1, entries.count(), "Invalid entry must remain preserved in place");
        }
    }

    @Test
    void danglingStageLinkIsQuarantinedWithoutFollowingIt() throws Exception {
        run("ln -s \"$PWD/nonexistent\" \"$OBJECT_STAGE_DIR\"\n",
                "[[ $OBJECT_REUSED == 0 && ! -L $OBJECT_STAGE_DIR ]]\n"
                        + "bad=(\"$OBJECT_STAGES_DIR.invalid\"/*/stage)\n"
                        + "[[ -L ${bad[0]} && ! -e $PWD/nonexistent ]]\n");
    }
}
