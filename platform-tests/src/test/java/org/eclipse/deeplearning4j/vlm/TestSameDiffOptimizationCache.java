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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TestSameDiffOptimizationCache {

    private static final String OPTIMIZER_ENABLED_PROPERTY = SameDiffOptimizationCache.OPTIMIZER_ENABLED_PROPERTY;
    private static final String TEST_FINGERPRINT = "sha256:test-cache-fingerprint";

    private String previousOptimizerEnabled;
    private String previousWeightDtype;
    private String previousFp16;
    private String previousBf16;
    private String previousFingerprint;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() throws Exception {
        previousOptimizerEnabled = System.getProperty(OPTIMIZER_ENABLED_PROPERTY);
        previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        previousFp16 = System.getProperty("nd4j.optimizer.fp16");
        previousBf16 = System.getProperty("nd4j.optimizer.bf16");
        System.clearProperty(OPTIMIZER_ENABLED_PROPERTY);
        System.clearProperty("nd4j.optimizer.weightDtype");
        System.clearProperty("nd4j.optimizer.fp16");
        System.clearProperty("nd4j.optimizer.bf16");
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
        restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
        restoreProperty("nd4j.optimizer.fp16", previousFp16);
        restoreProperty("nd4j.optimizer.bf16", previousBf16);
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

    @Test
    void cacheFileNameIncludesResolvedWeightDataType() {
        File source = tempDir.resolve("model.onnx").toFile();

        assertEquals("model.opt.sdz",
                SameDiffOptimizationCache.getOptimizedSdzCacheFile(source).getName());

        System.setProperty("nd4j.optimizer.weightDtype", "fp32");
        assertEquals("model.nofp16.opt.sdz",
                SameDiffOptimizationCache.getOptimizedSdzCacheFile(source).getName());

        System.setProperty("nd4j.optimizer.weightDtype", "bf16");
        assertEquals("model.bf16.opt.sdz",
                SameDiffOptimizationCache.getOptimizedSdzCacheFile(source).getName());

        System.setProperty("nd4j.optimizer.weightDtype", "fp8_e5m2");
        assertEquals("model.fp8_e5m2.opt.sdz",
                SameDiffOptimizationCache.getOptimizedSdzCacheFile(source).getName());

        System.setProperty("nd4j.optimizer.weightDtype", "int4");
        assertEquals("model.int4.opt.sdz",
                SameDiffOptimizationCache.getOptimizedSdzCacheFile(source).getName());
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

    private static void restoreProperty(String name, String previousValue) {
        if (previousValue == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previousValue);
        }
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
