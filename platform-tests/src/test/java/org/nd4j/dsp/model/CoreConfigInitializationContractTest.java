/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Source guard for initialization order; device profiling validates the effective wait policy. */
@Tag("source-lint")
class CoreConfigInitializationContractTest {

    @Test
    void inferenceDefaultsPrecedeTheFirstOpenMpRuntimeCall() throws Exception {
        String source = initializationSource();
        Matcher firstRuntimeCall = Pattern.compile("\\b(?:omp|kmp)_[A-Za-z0-9_]+\\s*\\(")
                .matcher(source);
        assertTrue(firstRuntimeCall.find(), "Expected an OpenMP runtime initialization call");
        for (String name : new String[] {"KMP_BLOCKTIME", "KMP_AFFINITY", "GOMP_SPINCOUNT"}) {
            int setting = source.indexOf("sd_setenv(\"" + name + "\"");
            assertTrue(setting >= 0 && setting < firstRuntimeCall.start(),
                    name + " must be set before OpenMP can initialize and cache its environment");
        }
    }

    @Test
    void inferenceDefaultsPreserveExplicitEnvironmentAndConfiguredThreadCount() throws Exception {
        String source = initializationSource();
        for (String name : new String[] {"KMP_BLOCKTIME", "KMP_AFFINITY", "GOMP_SPINCOUNT"}) {
            String quotedName = Pattern.quote("\"" + name + "\"");
            Pattern guardedDefault = Pattern.compile("if\\s*\\(!std::getenv\\(" + quotedName
                    + "\\)\\)\\s*\\{\\s*sd_setenv\\(" + quotedName
                    + ",\\s*\"[^\"\\r\\n]*\",\\s*0\\);\\s*}");
            assertTrue(guardedDefault.matcher(source).find(),
                    name + " must remain a non-overwriting default guarded by getenv");
        }
        assertTrue(source.contains("omp_set_num_threads(_maxThreads.load());"),
                "Initialization must retain the configured thread count rather than force a new cap");
    }

    private static String initializationSource() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String source = Files.readString(root.resolve(
                "libnd4j/include/system/config/impl/CoreConfig.cpp"));
        int start = source.indexOf("void CoreConfig::initFromEnvironment() {");
        assertTrue(start >= 0, "CoreConfig environment initializer must exist");
        return source.substring(start).replaceAll("(?s)/\\*.*?\\*/|//[^\\r\\n]*", "");
    }
}
