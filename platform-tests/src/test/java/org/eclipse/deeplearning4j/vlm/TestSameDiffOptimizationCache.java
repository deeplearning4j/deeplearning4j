/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.vlm;

import org.eclipse.deeplearning4j.vlm.model.loading.SameDiffOptimizationCache;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TestSameDiffOptimizationCache {

    private static final String OPTIMIZER_ENABLED_PROPERTY = SameDiffOptimizationCache.OPTIMIZER_ENABLED_PROPERTY;
    private static final String TEST_FINGERPRINT = "sha256:test-cache-fingerprint";

    private String previousOptimizerEnabled;
    private String previousFingerprint;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() throws Exception {
        previousOptimizerEnabled = System.getProperty(OPTIMIZER_ENABLED_PROPERTY);
        System.clearProperty(OPTIMIZER_ENABLED_PROPERTY);
        Field field = optimizerFingerprintField();
        previousFingerprint = (String) field.get(null);
        field.set(null, TEST_FINGERPRINT);
    }

    @AfterEach
    void tearDown() throws Exception {
        if (previousOptimizerEnabled == null) {
            System.clearProperty(OPTIMIZER_ENABLED_PROPERTY);
        } else {
            System.setProperty(OPTIMIZER_ENABLED_PROPERTY, previousOptimizerEnabled);
        }
        optimizerFingerprintField().set(null, previousFingerprint);
    }

    @Test
    void hashedSidecarMarksOptimizedCacheValid() throws Exception {
        CacheFiles files = cacheFiles();
        SameDiffOptimizationCache.writeBuildFingerprint(files.optSdz);

        String sidecar = Files.readString(metaFile(files.optSdz).toPath());
        assertTrue(sidecar.equals("optimizerFingerprint=" + TEST_FINGERPRINT + "\n"));
        assertTrue(SameDiffOptimizationCache.hasValidOptimizedCache(files.onnx, files.baseSdz, false));
    }

    @Test
    void legacyMultilineBuildInfoSidecarIsNotValidOptimizedCache() throws Exception {
        CacheFiles files = cacheFiles();
        Files.writeString(metaFile(files.optSdz).toPath(),
                "optimizerFingerprint=Build Info:\nGCC: example\nBuildStamp: old\n");

        assertFalse(SameDiffOptimizationCache.hasValidOptimizedCache(files.onnx, files.baseSdz, false));
    }

    private CacheFiles cacheFiles() throws Exception {
        File onnx = tempDir.resolve("model.onnx").toFile();
        File baseSdz = tempDir.resolve("model.sdz").toFile();
        File optSdz = tempDir.resolve("model.opt.sdz").toFile();
        Files.writeString(onnx.toPath(), "onnx");
        Files.writeString(baseSdz.toPath(), "base");
        Files.writeString(optSdz.toPath(), "opt");

        long now = System.currentTimeMillis();
        assertTrue(onnx.setLastModified(now - 2000));
        assertTrue(baseSdz.setLastModified(now - 1000));
        assertTrue(optSdz.setLastModified(now));
        return new CacheFiles(onnx, baseSdz, optSdz);
    }

    private static File metaFile(File cacheFile) {
        return new File(cacheFile.getParentFile(), cacheFile.getName() + ".meta");
    }

    private static Field optimizerFingerprintField() throws Exception {
        Field field = SameDiffOptimizationCache.class.getDeclaredField("OPTIMIZER_FINGERPRINT");
        field.setAccessible(true);
        return field;
    }

    private static class CacheFiles {
        final File onnx;
        final File baseSdz;
        final File optSdz;

        CacheFiles(File onnx, File baseSdz, File optSdz) {
            this.onnx = onnx;
            this.baseSdz = baseSdz;
            this.optSdz = optSdz;
        }
    }
}
