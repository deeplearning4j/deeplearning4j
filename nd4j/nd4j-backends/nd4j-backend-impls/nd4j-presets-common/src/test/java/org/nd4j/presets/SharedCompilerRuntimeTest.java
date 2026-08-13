/*
 * ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ******************************************************************************
 */

package org.nd4j.presets;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.Loader;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SharedCompilerRuntimeTest {

    private static final String RESOURCE_ROOT =
            "org/nd4j/presets/runtime-fixture/";

    @TempDir
    Path cacheDirectory;

    @Test
    void materializesCompatibilityAliasesWithoutPreloadingThem() throws Exception {
        Assumptions.assumeTrue(
                System.getProperty("os.name", "").toLowerCase().contains("linux"));
        Assumptions.assumeTrue(
                System.getProperty("os.arch", "").matches("amd64|x86_64"));

        System.setProperty(
                "org.bytedeco.javacpp.cachedir", cacheDirectory.toString());
        Properties loaderProperties = Loader.loadProperties();
        assertEquals("linux-x86_64", loaderProperties.getProperty("platform"));

        ClassLoader classLoader = SharedCompilerRuntimeTest.class.getClassLoader();
        String classifierRoot = RESOURCE_ROOT + "linux-x86_64/";
        URL canonicalResource = classLoader.getResource(
                classifierRoot + "libnvcuda.so");
        assertTrue(canonicalResource != null);

        File cachedCanonical = Loader.cacheResource(canonicalResource);
        assertTrue(cachedCanonical != null && cachedCanonical.isFile());
        Path canonicalPath = cachedCanonical.toPath();
        Path cudaAlias = canonicalPath.resolveSibling("libcuda.so");
        Path cudaSonameAlias = canonicalPath.resolveSibling("libcuda.so.1");

        // Model a stale JavaCPP cache produced when aliases were extracted as
        // independent regular files. Configuration must replace both with the
        // canonical runtime identity before any native library is loaded.
        Files.write(
                cudaAlias,
                "stale-independent-runtime".getBytes(StandardCharsets.UTF_8));
        Files.write(
                cudaSonameAlias,
                "stale-independent-runtime".getBytes(StandardCharsets.UTF_8));

        ClassProperties properties = new ClassProperties(loaderProperties);
        int added = SharedCompilerRuntime.configure(
                properties, SharedCompilerRuntimeTest.class, RESOURCE_ROOT);

        assertEquals(1, added);
        assertTrue(Files.isSameFile(cudaAlias, canonicalPath));
        assertTrue(Files.isSameFile(cudaSonameAlias, canonicalPath));

        List<String> preloads = properties.get("platform.preload");
        assertEquals(1, preloads.stream()
                .filter(value -> value.contains("libnvcuda.so"))
                .count());
        assertFalse(preloads.stream()
                .anyMatch(value -> value.contains(":libcuda.so")));
    }
}
