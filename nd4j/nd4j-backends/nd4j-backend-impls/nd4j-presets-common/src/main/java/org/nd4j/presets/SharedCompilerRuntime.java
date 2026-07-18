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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

/**
 * Aligns JavaCPP's dependent-library list with the project-managed shared
 * compiler runtimes selected by CMake's {@code StageSharedRuntime.cmake}.
 */
public final class SharedCompilerRuntime {

    public static final String MANIFEST_NAME = "shared-runtime-manifest.txt";
    private static final String MANIFEST_FORMAT =
            "# nd4j-shared-runtime-manifest-v1";
    private static final String RUNTIME_COUNT_PREFIX = "# runtime-count=";
    private static final String BUILD_TOOLCHAIN_NAME =
            "javacpp-build-toolchain.properties";

    private SharedCompilerRuntime() {
    }

    /**
     * Adds the exact CMake-selected compiler runtimes to JavaCPP's preload list.
     *
     * @param properties JavaCPP platform properties
     * @param presetClass preset class whose class loader owns the classifier
     * @param resourceRoot package-relative classifier root, ending in {@code /}
     * @return number of compiler runtimes added to the preload list
     */
    public static int configure(
            ClassProperties properties,
            Class<?> presetClass,
            String resourceRoot) {
        String platform = properties.getProperty("platform");
        if (platform == null || platform.isEmpty()) {
            throw new IllegalStateException(
                    "JavaCPP ClassProperties has no platform");
        }
        Set<String> runtimeNames;
        if (Loader.isLoadLibraries()) {
            Properties configuredProperties = Loader.loadProperties();
            String configuredPlatform =
                    configuredProperties.getProperty("platform");
            if (!platform.equals(configuredPlatform)) {
                throw new IllegalStateException(
                        "JavaCPP platform mismatch: ClassProperties platform='"
                                + platform + "', configured platform='"
                                + configuredPlatform + "'");
            }
            String configuredExtension =
                    configuredProperties.getProperty("platform.extension");
            runtimeNames = readBundledManifest(
                    presetClass,
                    resourceRoot,
                    platform,
                    configuredExtension,
                    properties.get("platform.extension"));
        } else {
            Path manifest = findBuildManifest(properties);
            if (manifest == null) {
                throw new IllegalStateException(
                        "The native build did not produce " + MANIFEST_NAME
                                + " in any JavaCPP platform.linkpath");
            }
            runtimeNames = readBuildManifest(manifest);
            configureBuildToolchain(properties, manifest.getParent());
            configureBuildLinking(properties, manifest.getParent(), platform);
        }

        List<String> preloads = properties.get("platform.preload");
        int insertionIndex = 0;
        int added = 0;
        for (String runtimeName : runtimeNames) {
            String runtimeSpec = runtimeSpec(runtimeName);
            if (!preloads.contains(runtimeSpec)) {
                preloads.add(insertionIndex++, runtimeSpec);
                added++;
            }
        }
        return added;
    }

    private static Path findBuildManifest(ClassProperties properties) {
        for (String linkPath : properties.get("platform.linkpath")) {
            Path manifest = Paths.get(linkPath, MANIFEST_NAME);
            if (Files.isRegularFile(manifest)) {
                return manifest.toAbsolutePath().normalize();
            }
        }
        return null;
    }

    private static Set<String> readBuildManifest(Path manifest) {
        try (BufferedReader reader = Files.newBufferedReader(
                manifest, StandardCharsets.UTF_8)) {
            return readRuntimeNames(reader, manifest.toString());
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Cannot read compiler runtime manifest " + manifest, e);
        }
    }

    private static void configureBuildToolchain(
            ClassProperties properties, Path runtimeDirectory) {
        Path toolchain = runtimeDirectory.resolve(BUILD_TOOLCHAIN_NAME);
        if (!Files.isRegularFile(toolchain)) {
            throw new IllegalStateException(
                    "CMake did not produce JavaCPP build toolchain metadata "
                            + toolchain);
        }

        String compiler = null;
        try (BufferedReader reader = Files.newBufferedReader(
                toolchain, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String prefix = "platform.compiler=";
                if (!line.startsWith(prefix) || compiler != null) {
                    throw new IllegalStateException(
                            "Invalid JavaCPP build toolchain metadata " + toolchain);
                }
                compiler = line.substring(prefix.length());
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Cannot read JavaCPP build toolchain metadata " + toolchain, e);
        }

        if (compiler == null || compiler.isEmpty()) {
            throw new IllegalStateException(
                    "JavaCPP build toolchain metadata has no platform.compiler: "
                            + toolchain);
        }
        Path compilerPath = Paths.get(compiler);
        if (!compilerPath.isAbsolute() || !Files.isRegularFile(compilerPath)) {
            throw new IllegalStateException(
                    "CMake C++ compiler is not an absolute executable file: "
                            + compiler);
        }
        properties.setProperty("platform.compiler", compilerPath.toString());
    }

    private static void configureBuildLinking(
            ClassProperties properties, Path runtimeDirectory, String platform) {
        // The JNI wrapper directly links only the backend DSO. CMake has already
        // selected and staged that backend's project-managed compiler runtimes.
        List<String> linkPaths = properties.get("platform.linkpath");
        linkPaths.clear();
        linkPaths.add(runtimeDirectory.toString());
        properties.get("platform.linkresource").clear();

        // JavaCPP normally emits one absolute rpath for every build-time link path.
        // Keep link discovery (-L) but let the relocatable loader path own runtime
        // discovery, matching the backend DSO's CMake configuration.
        properties.remove("platform.linkpath.prefix2");

        String loaderToken = null;
        String loaderFlag = null;
        if (platform.startsWith("linux")) {
            loaderToken = "$ORIGIN";
            loaderFlag = "-Wl,-rpath,$ORIGIN/";
        } else if (platform.startsWith("macosx")) {
            loaderToken = "@loader_path";
            loaderFlag = "-Wl,-rpath,@loader_path";
        }
        if (loaderFlag != null) {
            String output = properties.getProperty("platform.compiler.output");
            if (output == null || output.isEmpty()) {
                throw new IllegalStateException(
                        "JavaCPP platform has no shared-library compiler output flags: "
                                + platform);
            }
            if (!output.contains(loaderToken)) {
                properties.setProperty(
                        "platform.compiler.output", loaderFlag + " " + output);
            }
        }
    }

    private static Set<String> readBundledManifest(
            Class<?> presetClass,
            String resourceRoot,
            String platform,
            String configuredExtension,
            List<String> candidateExtensions) {
        Set<String> classifiers = new LinkedHashSet<>();
        if (configuredExtension != null && !configuredExtension.isEmpty()) {
            // An explicitly configured extension is exact. Falling back to another
            // classifier would silently mix native libraries from different artifacts.
            classifiers.add(platform + configuredExtension);
        } else {
            // JavaCPP searches annotation-provided extensions in reverse order so later
            // properties override earlier ones, then falls back to the base platform.
            for (int i = candidateExtensions.size() - 1; i >= 0; i--) {
                classifiers.add(platform + candidateExtensions.get(i));
            }
            classifiers.add(platform);
        }

        ClassLoader classLoader = presetClass.getClassLoader();
        Set<String> attemptedResources = new LinkedHashSet<>();
        for (String classifier : classifiers) {
            String resource = resourceRoot + classifier + "/" + MANIFEST_NAME;
            attemptedResources.add(resource);
            InputStream input = classLoader.getResourceAsStream(resource);
            if (input == null) {
                continue;
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                    input, StandardCharsets.UTF_8))) {
                Set<String> runtimeNames = readRuntimeNames(reader, resource);
                for (String runtimeName : runtimeNames) {
                    if (classLoader.getResource(resourceRoot + classifier
                            + "/" + runtimeName) == null) {
                        throw new IllegalStateException(
                                "Classifier '" + classifier
                                        + "' is missing runtime " + runtimeName
                                        + " declared by " + resource);
                    }
                }
                return runtimeNames;
            } catch (IOException e) {
                throw new IllegalStateException(
                        "Cannot read compiler runtime manifest " + resource, e);
            }
        }

        String configured = configuredExtension == null
                || configuredExtension.isEmpty()
                ? "<none>" : "'" + configuredExtension + "'";
        throw new IllegalStateException(
                "No compiler runtime manifest matches JavaCPP platform configuration: "
                        + "platform='" + platform + "', configuredExtension="
                        + configured + ", candidateExtensions="
                        + candidateExtensions + ", attemptedResources="
                        + attemptedResources);
    }

    private static Set<String> readRuntimeNames(
            BufferedReader reader, String source) throws IOException {
        Set<String> runtimeNames = new LinkedHashSet<>();
        boolean formatSeen = false;
        Integer declaredCount = null;
        String line;
        while ((line = reader.readLine()) != null) {
            String runtimeName = line.trim();
            if (runtimeName.isEmpty()) {
                continue;
            }
            if (runtimeName.equals(MANIFEST_FORMAT)) {
                if (formatSeen) {
                    throw new IllegalStateException(
                            "Duplicate format marker in compiler runtime manifest "
                                    + source);
                }
                formatSeen = true;
                continue;
            }
            if (runtimeName.startsWith(RUNTIME_COUNT_PREFIX)) {
                if (declaredCount != null) {
                    throw new IllegalStateException(
                            "Duplicate runtime count in compiler runtime manifest "
                                    + source);
                }
                String count = runtimeName.substring(RUNTIME_COUNT_PREFIX.length());
                try {
                    declaredCount = Integer.valueOf(count);
                } catch (NumberFormatException e) {
                    throw new IllegalStateException(
                            "Invalid runtime count '" + count
                                    + "' in compiler runtime manifest " + source,
                            e);
                }
                if (declaredCount < 0) {
                    throw new IllegalStateException(
                            "Negative runtime count in compiler runtime manifest "
                                    + source);
                }
                continue;
            }
            if (runtimeName.startsWith("#")) {
                continue;
            }
            if (!runtimeName.matches("[A-Za-z0-9][A-Za-z0-9._+@-]*")) {
                throw new IllegalStateException(
                        "Invalid shared-library loader name '" + runtimeName
                                + "' in compiler runtime manifest " + source);
            }
            runtimeNames.add(runtimeName);
        }
        if (!formatSeen || declaredCount == null) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source
                            + " has no supported format/count header");
        }
        if (declaredCount != runtimeNames.size()) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source + " declares "
                            + declaredCount + " runtimes but contains "
                            + runtimeNames.size());
        }
        return runtimeNames;
    }

    private static String runtimeSpec(String runtimeName) {
        String alias = runtimeName.replaceAll("[^A-Za-z0-9]", "_");
        return "nd4j_compiler_runtime_" + alias + ":"
                + runtimeName + "#" + runtimeName;
    }
}
