/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.autodiff.samediff.serde;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.config.ND4JSystemProperties;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class SDZSerializerTempDirectoryTest {

    @TempDir
    Path tempDirectory;

    @Test
    void usesRelatedModelDirectoryBeforeConfiguredFallback() throws Exception {
        Path modelDirectory = tempDirectory.resolve("app-private").resolve("model-cache").resolve("tmp");
        Files.createDirectories(modelDirectory);
        Path configuredFallback = tempDirectory.resolve("configured-fallback");
        String previous = System.getProperty(ND4JSystemProperties.ND4J_TEMP_DIR_PROPERTY);
        Path workingDirectory = null;
        try {
            System.setProperty(ND4JSystemProperties.ND4J_TEMP_DIR_PROPERTY,
                    configuredFallback.toString());
            workingDirectory = SDZSerializer.createWorkingDirectory(
                    modelDirectory.resolve("generated.sdz").toFile(), "sdz-test-");

            assertEquals(modelDirectory.toRealPath(), workingDirectory.getParent().toRealPath());
            assertFalse(Files.exists(configuredFallback));
        } finally {
            if (workingDirectory != null) {
                Files.deleteIfExists(workingDirectory);
            }
            if (previous == null) {
                System.clearProperty(ND4JSystemProperties.ND4J_TEMP_DIR_PROPERTY);
            } else {
                System.setProperty(ND4JSystemProperties.ND4J_TEMP_DIR_PROPERTY, previous);
            }
        }
    }
}
