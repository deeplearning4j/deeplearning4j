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
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SharedCompilerRuntimeJarTest {

    private static final String RESOURCE_ROOT =
            "org/nd4j/presets/runtime-jar-fixture/";

    @TempDir
    Path temporaryDirectory;

    @Test
    void materializesAliasesBesideCanonicalRuntimeFromJarResources()
            throws Exception {
        Assumptions.assumeTrue(
                System.getProperty("os.name", "").toLowerCase().contains("linux"));
        Assumptions.assumeTrue(
                System.getProperty("os.arch", "").matches("amd64|x86_64"));

        String classifierRoot = RESOURCE_ROOT + "linux-x86_64/";
        Path fixtureJar = temporaryDirectory.resolve("runtime-fixture.jar");
        try (JarOutputStream output = new JarOutputStream(
                Files.newOutputStream(fixtureJar))) {
            addEntry(output, classifierRoot + SharedCompilerRuntime.MANIFEST_NAME,
                    String.join("\n",
                            "# nd4j-shared-runtime-manifest-v1",
                            "# runtime-count=1",
                            "# runtime-alias-count=2",
                            "# runtime-alias=libcuda.so->libnvcuda.so",
                            "# runtime-alias=libcuda.so.1->libnvcuda.so",
                            "libnvcuda.so",
                            ""));
            addEntry(output, classifierRoot + "libnvcuda.so", "canonical-runtime");
            addEntry(output, classifierRoot + "libcuda.so", "packaged-alias");
            addEntry(output, classifierRoot + "libcuda.so.1", "packaged-alias");
        }

        String cacheProperty = "org.bytedeco.javacpp.cachedir";
        String previousCache = System.getProperty(cacheProperty);
        Path cacheDirectory = temporaryDirectory.resolve("javacpp-cache");
        System.setProperty(cacheProperty, cacheDirectory.toString());
        try (URLClassLoader resourceLoader = new URLClassLoader(
                new URL[]{fixtureJar.toUri().toURL()}, null)) {
            Class<?> presetClass = Proxy.newProxyInstance(
                    resourceLoader,
                    new Class<?>[]{Runnable.class},
                    (proxy, method, arguments) -> null).getClass();
            Properties loaderProperties = Loader.loadProperties();
            assertEquals("linux-x86_64", loaderProperties.getProperty("platform"));

            ClassProperties properties = new ClassProperties(loaderProperties);
            assertEquals(1, SharedCompilerRuntime.configure(
                    properties, presetClass, RESOURCE_ROOT));

            URL canonicalResource = resourceLoader.getResource(
                    classifierRoot + "libnvcuda.so");
            assertTrue(canonicalResource != null);
            File cachedCanonical = Loader.cacheResource(canonicalResource);
            assertTrue(cachedCanonical != null && cachedCanonical.isFile());
            Path canonicalPath = cachedCanonical.toPath();
            assertTrue(Files.isSameFile(
                    canonicalPath.resolveSibling("libcuda.so"), canonicalPath));
            assertTrue(Files.isSameFile(
                    canonicalPath.resolveSibling("libcuda.so.1"), canonicalPath));
        } finally {
            if (previousCache == null) {
                System.clearProperty(cacheProperty);
            } else {
                System.setProperty(cacheProperty, previousCache);
            }
        }
    }

    private static void addEntry(
            JarOutputStream output, String name, String contents) throws IOException {
        output.putNextEntry(new JarEntry(name));
        output.write(contents.getBytes(StandardCharsets.UTF_8));
        output.closeEntry();
    }
}
