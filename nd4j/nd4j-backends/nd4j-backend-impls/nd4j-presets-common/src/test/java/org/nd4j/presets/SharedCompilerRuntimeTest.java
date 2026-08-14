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
    void usesValidatedExternalRuntimeClosureWhenLibraryPathProvidesIt() throws Exception {
        Path runtimeDirectory = Files.createDirectory(cacheDirectory.resolve("runtime"));
        Files.write(runtimeDirectory.resolve(SharedCompilerRuntime.MANIFEST_NAME), List.of(
                "# nd4j-shared-runtime-manifest-v1",
                "# runtime-count=2",
                "libLLVM.so.22.0git",
                "libMLIR.so.22.0git"), StandardCharsets.UTF_8);
        Files.write(runtimeDirectory.resolve("libLLVM.so.22.0git"), new byte[]{1});
        Files.write(runtimeDirectory.resolve("libMLIR.so.22.0git"), new byte[]{2});

        String previousLibraryPath =
                System.getProperty("org.bytedeco.javacpp.library.path");
        try {
            System.setProperty(
                    "org.bytedeco.javacpp.library.path",
                    runtimeDirectory.toString());
            ClassProperties properties =
                    new ClassProperties(Loader.loadProperties());

            int added = SharedCompilerRuntime.configure(
                    properties, SharedCompilerRuntimeTest.class, RESOURCE_ROOT);

            assertEquals(2, added);
            List<String> preloads = properties.get("platform.preload");
            assertTrue(preloads.stream()
                    .anyMatch(value -> value.endsWith(
                            ":libLLVM.so.22.0git#libLLVM.so.22.0git")));
            assertTrue(preloads.stream()
                    .anyMatch(value -> value.endsWith(
                            ":libMLIR.so.22.0git#libMLIR.so.22.0git")));
        } finally {
            if (previousLibraryPath == null) {
                System.clearProperty("org.bytedeco.javacpp.library.path");
            } else {
                System.setProperty(
                        "org.bytedeco.javacpp.library.path",
                        previousLibraryPath);
            }
        }
    }

    @Test
    void usesValidatedExternalRuntimeClosureFromNd4jSubprocessPath() throws Exception {
        Path runtimeDirectory = Files.createDirectory(cacheDirectory.resolve("subprocess-runtime"));
        Files.write(runtimeDirectory.resolve(SharedCompilerRuntime.MANIFEST_NAME), List.of(
                "# nd4j-shared-runtime-manifest-v1",
                "# runtime-count=1",
                "libLLVM.so.22.0git"), StandardCharsets.UTF_8);
        Files.write(runtimeDirectory.resolve("libLLVM.so.22.0git"), new byte[]{1});

        String previousNd4jPath =
                System.getProperty("org.nd4j.presets.sharedRuntimePath");
        String previousJavaCppPath =
                System.getProperty("org.bytedeco.javacpp.library.path");
        try {
            System.setProperty(
                    "org.nd4j.presets.sharedRuntimePath",
                    runtimeDirectory.toString());
            System.clearProperty("org.bytedeco.javacpp.library.path");
            ClassProperties properties =
                    new ClassProperties(Loader.loadProperties());

            int added = SharedCompilerRuntime.configure(
                    properties, SharedCompilerRuntimeTest.class, RESOURCE_ROOT);

            assertEquals(1, added);
            assertTrue(properties.get("platform.preload").stream()
                    .anyMatch(value -> value.endsWith(
                            ":libLLVM.so.22.0git#libLLVM.so.22.0git")));
        } finally {
            restoreProperty("org.nd4j.presets.sharedRuntimePath", previousNd4jPath);
            restoreProperty("org.bytedeco.javacpp.library.path", previousJavaCppPath);
        }
    }

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

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
