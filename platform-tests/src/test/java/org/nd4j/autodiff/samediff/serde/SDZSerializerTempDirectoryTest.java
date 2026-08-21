/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.autodiff.samediff.serde;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    @Test
    void overlappingModelLoadsRestoreExecutionStateOnlyAfterLastScopeCloses() {
        boolean previousDsp = InferenceSession.isDynamicShapePlanEnabled();
        String previousCudaGraphs =
                System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
        SDZSerializer.ModelLoadExecutionScope first = null;
        SDZSerializer.ModelLoadExecutionScope second = null;
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "true");

            first = SDZSerializer.suppressDspDuringModelLoad();
            second = SDZSerializer.suppressDspDuringModelLoad();
            assertFalse(InferenceSession.isDynamicShapePlanEnabled());
            assertEquals("false",
                    System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED));

            // The first loader may finish while a later parallel loader is still active.
            first.close();
            first = null;
            assertFalse(InferenceSession.isDynamicShapePlanEnabled());
            assertEquals("false",
                    System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED));

            second.close();
            second = null;
            assertTrue(InferenceSession.isDynamicShapePlanEnabled());
            assertEquals("true",
                    System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED));
        } finally {
            if (first != null) {
                first.close();
            }
            if (second != null) {
                second.close();
            }
            InferenceSession.setDynamicShapePlanEnabled(previousDsp);
            if (previousCudaGraphs == null) {
                System.clearProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
            } else {
                System.setProperty(
                        ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, previousCudaGraphs);
            }
        }
    }
}
