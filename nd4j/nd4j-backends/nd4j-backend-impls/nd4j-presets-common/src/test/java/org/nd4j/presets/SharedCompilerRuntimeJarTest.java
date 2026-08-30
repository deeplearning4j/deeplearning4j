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
                            "# resource-count=2",
                            "# resource=rocblas/library/TensileLibrary.dat",
                            "# resource=.kpack/blas_lib_gfx1103.kpack",
                            "# runtime-alias=libcuda.so->libnvcuda.so",
                            "# runtime-alias=libcuda.so.1->libnvcuda.so",
                            "libnvcuda.so",
                            ""));
            addEntry(output, classifierRoot + "libnvcuda.so", "canonical-runtime");
            addEntry(output, classifierRoot + "libcuda.so", "packaged-alias");
            addEntry(output, classifierRoot + "libcuda.so.1", "packaged-alias");
            addEntry(output, classifierRoot
                    + "rocblas/library/TensileLibrary.dat", "tensile-data");
            addEntry(output, classifierRoot
                    + ".kpack/blas_lib_gfx1103.kpack", "blas-kpack");
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
            Path materializedTensile = canonicalPath.getParent().resolve(
                    "rocblas/library/TensileLibrary.dat");
            assertTrue(Files.isRegularFile(materializedTensile));
            assertEquals("tensile-data", Files.readString(materializedTensile));
            URL tensileResource = resourceLoader.getResource(
                    classifierRoot + "rocblas/library/TensileLibrary.dat");
            assertTrue(tensileResource != null);
            File cachedTensile = Loader.cacheResource(tensileResource);
            assertTrue(cachedTensile != null && cachedTensile.isFile());
            assertEquals(
                    "blas-kpack",
                    Files.readString(canonicalPath.getParent().resolve(
                            ".kpack/blas_lib_gfx1103.kpack")));
        } finally {
            if (previousCache == null) {
                System.clearProperty(cacheProperty);
            } else {
                System.setProperty(cacheProperty, previousCache);
            }
        }
    }

    @Test
    void honorsResolvedExtensionWhenMultipleClassifierManifestsExist()
            throws Exception {
        Assumptions.assumeTrue(
                System.getProperty("os.name", "").toLowerCase().contains("linux"));
        Assumptions.assumeTrue(
                System.getProperty("os.arch", "").matches("amd64|x86_64"));

        String defaultExtension = "-zluda-rocm-7.2.4";
        String rocm10Extension = "-zluda-rocm-10.0.0";
        String defaultRoot = RESOURCE_ROOT + "linux-x86_64" + defaultExtension + "/";
        String rocm10Root = RESOURCE_ROOT + "linux-x86_64" + rocm10Extension + "/";
        Path fixtureJar = temporaryDirectory.resolve("multi-classifier-fixture.jar");
        try (JarOutputStream output = new JarOutputStream(
                Files.newOutputStream(fixtureJar))) {
            addEntry(output, defaultRoot + SharedCompilerRuntime.MANIFEST_NAME,
                    "# nd4j-shared-runtime-manifest-v1\n"
                            + "# runtime-count=1\nlibdefault.so\n");
            addEntry(output, defaultRoot + "libdefault.so", "default");
            addEntry(output, rocm10Root + SharedCompilerRuntime.MANIFEST_NAME,
                    "# nd4j-shared-runtime-manifest-v1\n"
                            + "# runtime-count=1\nlibrocm10.so\n");
            addEntry(output, rocm10Root + "librocm10.so", "rocm10");
        }

        String cacheProperty = "org.bytedeco.javacpp.cachedir";
        String previousCache = System.getProperty(cacheProperty);
        System.setProperty(
                cacheProperty,
                temporaryDirectory.resolve("multi-classifier-cache").toString());
        try (URLClassLoader resourceLoader = new URLClassLoader(
                new URL[]{fixtureJar.toUri().toURL()}, null)) {
            Class<?> presetClass = Proxy.newProxyInstance(
                    resourceLoader,
                    new Class<?>[]{Runnable.class},
                    (proxy, method, arguments) -> null).getClass();
            Properties loaderProperties = Loader.loadProperties();
            Assumptions.assumeTrue(loaderProperties.getProperty(
                    "platform.extension", "").isEmpty());
            ClassProperties properties = new ClassProperties(loaderProperties);
            properties.get("platform.extension").clear();
            properties.get("platform.extension").add(defaultExtension);
            properties.get("platform.extension").add(rocm10Extension);

            assertEquals(1, SharedCompilerRuntime.configure(
                    properties,
                    presetClass,
                    RESOURCE_ROOT,
                    defaultExtension));
            assertTrue(properties.get("platform.preload").stream()
                    .anyMatch(value -> value.contains("libdefault.so")));
            assertTrue(properties.get("platform.preload").stream()
                    .noneMatch(value -> value.contains("librocm10.so")));
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
