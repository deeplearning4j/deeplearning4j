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
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Aligns JavaCPP's dependent-library list with the project-managed shared
 * compiler runtimes selected by CMake's {@code StageSharedRuntime.cmake}.
 */
public final class SharedCompilerRuntime {

    public static final String MANIFEST_NAME = "shared-runtime-manifest.txt";
    private static final String MANIFEST_FORMAT =
            "# nd4j-shared-runtime-manifest-v1";
    private static final String RUNTIME_COUNT_PREFIX = "# runtime-count=";
    private static final String RUNTIME_ALIAS_COUNT_PREFIX =
            "# runtime-alias-count=";
    private static final String RUNTIME_ALIAS_PREFIX = "# runtime-alias=";
    private static final String RESOURCE_COUNT_PREFIX = "# resource-count=";
    private static final String RESOURCE_PREFIX = "# resource=";
    private static final String RUNTIME_ALIAS_SEPARATOR = "->";
    private static final String BUILD_TOOLCHAIN_NAME =
            "javacpp-build-toolchain.properties";
    private static final String ND4J_SHARED_RUNTIME_PATH_PROPERTY =
            "org.nd4j.presets.sharedRuntimePath";
    private static final String JAVACPP_LIBRARY_PATH_PROPERTY =
            "org.bytedeco.javacpp.library.path";

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
        RuntimeManifest manifest;
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
            Path externalManifest = findExternalRuntimeManifest();
            if (externalManifest != null) {
                manifest = readExternalRuntimeManifest(externalManifest);
            } else {
                String configuredExtension =
                        configuredProperties.getProperty("platform.extension");
                manifest = readBundledManifest(
                        presetClass,
                        resourceRoot,
                        platform,
                        configuredExtension,
                        properties.get("platform.extension"));
            }
        } else {
            Path manifestPath = findBuildManifest(properties);
            if (manifestPath == null) {
                throw new IllegalStateException(
                        "The native build did not produce " + MANIFEST_NAME
                                + " in any JavaCPP platform.linkpath");
            }
            manifest = readBuildManifest(manifestPath);
            configureBuildToolchain(properties, manifestPath.getParent());
            configureBuildLinking(properties, manifestPath.getParent(), platform);
        }

        List<String> preloads = properties.get("platform.preload");
        int insertionIndex = 0;
        int added = 0;
        for (String runtimeName : manifest.runtimeNames) {
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

    private static RuntimeManifest readBuildManifest(Path manifest) {
        try (BufferedReader reader = Files.newBufferedReader(
                manifest, StandardCharsets.UTF_8)) {
            return readRuntimeManifest(reader, manifest.toString());
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Cannot read compiler runtime manifest " + manifest, e);
        }
    }

    private static Path findExternalRuntimeManifest() {
        Set<Path> manifests = new LinkedHashSet<>();
        collectExternalRuntimeManifests(
                System.getProperty(ND4J_SHARED_RUNTIME_PATH_PROPERTY), manifests);
        collectExternalRuntimeManifests(
                System.getProperty(JAVACPP_LIBRARY_PATH_PROPERTY), manifests);
        if (manifests.size() > 1) {
            throw new IllegalStateException(
                    "Multiple external compiler runtime manifests found in "
                            + ND4J_SHARED_RUNTIME_PATH_PROPERTY + " or "
                            + JAVACPP_LIBRARY_PATH_PROPERTY + ": " + manifests);
        }
        return manifests.isEmpty() ? null : manifests.iterator().next();
    }

    private static void collectExternalRuntimeManifests(
            String configuredPath, Set<Path> manifests) {
        if (configuredPath == null || configuredPath.isBlank()) {
            return;
        }
        for (String entry : configuredPath.split(
                Pattern.quote(File.pathSeparator), -1)) {
            if (entry.isBlank()) {
                continue;
            }
            Path manifest = Paths.get(entry).toAbsolutePath().normalize()
                    .resolve(MANIFEST_NAME);
            if (Files.isRegularFile(manifest)) {
                manifests.add(manifest);
            }
        }
    }

    private static RuntimeManifest readExternalRuntimeManifest(Path manifestPath) {
        RuntimeManifest manifest = readBuildManifest(manifestPath);
        Path runtimeDirectory = manifestPath.getParent();
        for (String runtimeName : manifest.runtimeNames) {
            Path runtime = runtimeDirectory.resolve(runtimeName);
            if (!Files.isRegularFile(runtime)) {
                throw new IllegalStateException(
                        "External compiler runtime manifest " + manifestPath
                                + " declares missing runtime " + runtimeName);
            }
        }
        for (Map.Entry<String, String> alias : manifest.runtimeAliases.entrySet()) {
            Path aliasPath = runtimeDirectory.resolve(alias.getKey());
            Path targetPath = runtimeDirectory.resolve(alias.getValue());
            try {
                if (!Files.isRegularFile(aliasPath)
                        || !Files.isSameFile(aliasPath, targetPath)) {
                    throw new IllegalStateException(
                            "External compiler runtime alias '" + alias.getKey()
                                    + "' does not resolve to canonical runtime '"
                                    + alias.getValue() + "' in " + runtimeDirectory);
                }
            } catch (IOException e) {
                throw new IllegalStateException(
                        "Cannot validate external compiler runtime alias '"
                                + alias.getKey() + "' in " + runtimeDirectory, e);
            }
        }
        for (String resourceName : manifest.resourceNames) {
            Path resource = runtimeDirectory.resolve(resourceName).normalize();
            if (!resource.startsWith(runtimeDirectory)
                    || !Files.isRegularFile(resource)) {
                throw new IllegalStateException(
                        "External compiler runtime manifest " + manifestPath
                                + " declares missing resource " + resourceName);
            }
            try {
                Loader.cacheResource(resource.toUri().toURL());
            } catch (IOException e) {
                throw new IllegalStateException(
                        "Cannot extract external runtime resource " + resourceName
                                + " from " + manifestPath, e);
            }
        }
        return manifest;
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
        String requestedCompiler = System.getProperty("javacpp.platform.compiler");
        if (properties.getProperty("platform").startsWith("android")
                && requestedCompiler != null
                && !requestedCompiler.isEmpty()) {
            // Android JavaCPP compilation is cross-compilation. The CMake manifest
            // describes the host compiler used to build the shared runtime, while
            // JavaCPP must use the API-specific NDK target wrapper selected by the
            // release shard.
            compiler = requestedCompiler;
        } else {
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

    private static RuntimeManifest readBundledManifest(
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
                RuntimeManifest manifest = readRuntimeManifest(reader, resource);
                String runtimeResourceRoot = resourceRoot + classifier + "/";
                for (String runtimeName : manifest.runtimeNames) {
                    if (classLoader.getResource(runtimeResourceRoot
                            + runtimeName) == null) {
                        throw new IllegalStateException(
                                "Classifier '" + classifier
                                        + "' is missing runtime " + runtimeName
                                        + " declared by " + resource);
                    }
                }
                for (String runtimeAlias : manifest.runtimeAliases.keySet()) {
                    if (classLoader.getResource(runtimeResourceRoot
                            + runtimeAlias) == null) {
                        throw new IllegalStateException(
                                "Classifier '" + classifier
                                        + "' is missing compatibility alias "
                                        + runtimeAlias + " declared by " + resource);
                    }
                }
                materializeRuntimeAliases(
                        classLoader, runtimeResourceRoot, manifest, resource);
                materializeRuntimeResources(
                        classLoader, runtimeResourceRoot, manifest, resource);
                return manifest;
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

    private static RuntimeManifest readRuntimeManifest(
            BufferedReader reader, String source) throws IOException {
        Set<String> runtimeNames = new LinkedHashSet<>();
        Map<String, String> runtimeAliases = new LinkedHashMap<>();
        Set<String> resourceNames = new LinkedHashSet<>();
        boolean formatSeen = false;
        Integer declaredCount = null;
        Integer declaredAliasCount = null;
        Integer declaredResourceCount = null;
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
            if (runtimeName.startsWith(RUNTIME_ALIAS_COUNT_PREFIX)) {
                if (declaredAliasCount != null) {
                    throw new IllegalStateException(
                            "Duplicate runtime alias count in compiler runtime manifest "
                                    + source);
                }
                String count = runtimeName.substring(
                        RUNTIME_ALIAS_COUNT_PREFIX.length());
                try {
                    declaredAliasCount = Integer.valueOf(count);
                } catch (NumberFormatException e) {
                    throw new IllegalStateException(
                            "Invalid runtime alias count '" + count
                                    + "' in compiler runtime manifest " + source,
                            e);
                }
                if (declaredAliasCount < 0) {
                    throw new IllegalStateException(
                            "Negative runtime alias count in compiler runtime manifest "
                                    + source);
                }
                continue;
            }
            if (runtimeName.startsWith(RESOURCE_COUNT_PREFIX)) {
                if (declaredResourceCount != null) {
                    throw new IllegalStateException(
                            "Duplicate resource count in compiler runtime manifest "
                                    + source);
                }
                String count = runtimeName.substring(RESOURCE_COUNT_PREFIX.length());
                try {
                    declaredResourceCount = Integer.valueOf(count);
                } catch (NumberFormatException e) {
                    throw new IllegalStateException(
                            "Invalid resource count '" + count
                                    + "' in compiler runtime manifest " + source,
                            e);
                }
                if (declaredResourceCount < 0) {
                    throw new IllegalStateException(
                            "Negative resource count in compiler runtime manifest "
                                    + source);
                }
                continue;
            }
            if (runtimeName.startsWith(RESOURCE_PREFIX)) {
                String resourceName = runtimeName.substring(RESOURCE_PREFIX.length());
                validateResourceName(resourceName, source);
                if (!resourceNames.add(resourceName)) {
                    throw new IllegalStateException(
                            "Duplicate resource '" + resourceName
                                    + "' in compiler runtime manifest " + source);
                }
                continue;
            }
            if (runtimeName.startsWith(RUNTIME_ALIAS_PREFIX)) {
                String mapping = runtimeName.substring(RUNTIME_ALIAS_PREFIX.length());
                int separator = mapping.indexOf(RUNTIME_ALIAS_SEPARATOR);
                if (separator <= 0 || separator + RUNTIME_ALIAS_SEPARATOR.length()
                        >= mapping.length()
                        || mapping.indexOf(RUNTIME_ALIAS_SEPARATOR,
                        separator + RUNTIME_ALIAS_SEPARATOR.length()) >= 0) {
                    throw new IllegalStateException(
                            "Invalid runtime alias mapping '" + mapping
                                    + "' in compiler runtime manifest " + source);
                }
                String alias = mapping.substring(0, separator);
                String target = mapping.substring(
                        separator + RUNTIME_ALIAS_SEPARATOR.length());
                validateRuntimeName(alias, source);
                validateRuntimeName(target, source);
                if (runtimeAliases.put(alias, target) != null) {
                    throw new IllegalStateException(
                            "Duplicate runtime alias '" + alias
                                    + "' in compiler runtime manifest " + source);
                }
                continue;
            }
            if (runtimeName.startsWith("#")) {
                continue;
            }
            validateRuntimeName(runtimeName, source);
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
        if (!runtimeAliases.isEmpty() && declaredAliasCount == null) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source
                            + " declares runtime aliases without an alias count");
        }
        if (declaredAliasCount != null
                && declaredAliasCount != runtimeAliases.size()) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source + " declares "
                            + declaredAliasCount + " runtime aliases but contains "
                            + runtimeAliases.size());
        }
        if (!resourceNames.isEmpty() && declaredResourceCount == null) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source
                            + " declares resources without a resource count");
        }
        if (declaredResourceCount != null
                && declaredResourceCount != resourceNames.size()) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source + " declares "
                            + declaredResourceCount + " resources but contains "
                            + resourceNames.size());
        }
        for (Map.Entry<String, String> alias : runtimeAliases.entrySet()){
            if (runtimeNames.contains(alias.getKey())) {
                throw new IllegalStateException(
                        "Compiler runtime manifest " + source
                                + " lists compatibility alias '" + alias.getKey()
                                + "' as a preload runtime");
            }
            if (!runtimeNames.contains(alias.getValue())) {
                throw new IllegalStateException(
                        "Compiler runtime manifest " + source
                                + " maps compatibility alias '" + alias.getKey()
                                + "' to missing canonical runtime '"
                                + alias.getValue() + "'");
            }
        }
        return new RuntimeManifest(runtimeNames, runtimeAliases, resourceNames);
    }

    private static void validateResourceName(String resourceName, String source) {
        if (resourceName.isEmpty()
                || resourceName.startsWith("/")
                || resourceName.startsWith("\\")
                || resourceName.contains("\\")
                || resourceName.contains("../")
                || resourceName.equals("..")
                || !resourceName.startsWith("rocblas/library/")) {
            throw new IllegalStateException(
                    "Invalid runtime resource path '" + resourceName
                            + "' in compiler runtime manifest " + source);
        }
    }

    private static void validateRuntimeName(String runtimeName, String source) {
        if (!runtimeName.matches("[A-Za-z0-9][A-Za-z0-9._+@-]*")) {
            throw new IllegalStateException(
                    "Invalid shared-library loader name '" + runtimeName
                            + "' in compiler runtime manifest " + source);
        }
    }

    private static synchronized void materializeRuntimeResources(
            ClassLoader classLoader,
            String runtimeResourceRoot,
            RuntimeManifest manifest,
            String source) throws IOException {
        if (manifest.resourceNames.isEmpty()) {
            return;
        }
        if (manifest.runtimeNames.isEmpty()) {
            throw new IllegalStateException(
                    "Compiler runtime manifest " + source
                            + " declares resources without a runtime directory");
        }

        String anchorName = manifest.runtimeNames.iterator().next();
        URL anchorResource = classLoader.getResource(runtimeResourceRoot + anchorName);
        File cachedAnchor = anchorResource == null
                ? null : Loader.cacheResource(anchorResource);
        if (cachedAnchor == null || !cachedAnchor.isFile()) {
            throw new IllegalStateException(
                    "Cannot extract runtime anchor '" + anchorName
                            + "' declared by " + source);
        }
        Path cacheDirectory = cachedAnchor.toPath().toAbsolutePath()
                .normalize().getParent();
        Path lockPath = cacheDirectory.resolve(".nd4j-shared-runtime-resources.lock");
        try (FileChannel channel = FileChannel.open(
                lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
             FileLock ignored = channel.lock()) {
            for (String resourceName : manifest.resourceNames) {
                URL resource = classLoader.getResource(
                        runtimeResourceRoot + resourceName);
                if (resource == null) {
                    throw new IllegalStateException(
                            "Classifier resource '" + resourceName
                                    + "' declared by " + source + " is missing");
                }
                Path destination = cacheDirectory.resolve(resourceName)
                        .toAbsolutePath().normalize();
                if (!destination.startsWith(cacheDirectory)) {
                    throw new IllegalStateException(
                            "Classifier resource escapes JavaCPP cache directory: "
                                    + resourceName);
                }
                Files.createDirectories(destination.getParent());
                if (!Files.isRegularFile(destination)) {
                    File extracted = Loader.extractResource(
                            resource, destination.toFile(), null, null);
                    if (extracted == null
                            || !destination.equals(extracted.toPath()
                            .toAbsolutePath().normalize())) {
                        throw new IllegalStateException(
                                "Cannot extract classifier resource '" + resourceName
                                        + "' to its runtime-relative path declared by "
                                        + source);
                    }
                }
                if (!Files.isRegularFile(destination)) {
                    throw new IllegalStateException(
                            "Cannot materialize classifier resource '" + resourceName
                                    + "' beside runtimes declared by " + source);
                }
            }
        }
    }

    private static synchronized void materializeRuntimeAliases(
            ClassLoader classLoader,
            String runtimeResourceRoot,
            RuntimeManifest manifest,
            String source) throws IOException {
        if (manifest.runtimeAliases.isEmpty()) {
            return;
        }

        Map<String, Path> canonicalPaths = new LinkedHashMap<>();
        Path cacheDirectory = null;
        for (String target : new LinkedHashSet<>(manifest.runtimeAliases.values())) {
            URL targetResource = classLoader.getResource(runtimeResourceRoot + target);
            File cachedTarget = targetResource == null
                    ? null : Loader.cacheResource(targetResource);
            if (cachedTarget == null || !cachedTarget.isFile()) {
                throw new IllegalStateException(
                        "Cannot extract canonical runtime '" + target
                                + "' declared by " + source);
            }
            Path targetPath = cachedTarget.toPath().toAbsolutePath().normalize();
            if (cacheDirectory == null) {
                cacheDirectory = targetPath.getParent();
            } else if (!cacheDirectory.equals(targetPath.getParent())) {
                throw new IllegalStateException(
                        "Canonical runtimes declared by " + source
                                + " were extracted into different directories");
            }
            canonicalPaths.put(target, targetPath);
        }

        Path lockPath = cacheDirectory.resolve(".nd4j-shared-runtime-aliases.lock");
        try (FileChannel channel = FileChannel.open(
                lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
             FileLock ignored = channel.lock()) {
            for (Map.Entry<String, String> mapping
                    : manifest.runtimeAliases.entrySet()) {
                Path targetPath = canonicalPaths.get(mapping.getValue());
                Path aliasPath = cacheDirectory.resolve(mapping.getKey()).normalize();
                if (!cacheDirectory.equals(aliasPath.getParent())) {
                    throw new IllegalStateException(
                            "Runtime alias escapes JavaCPP cache directory: "
                                    + mapping.getKey());
                }
                if (Files.exists(aliasPath, LinkOption.NOFOLLOW_LINKS)
                        && Files.isSameFile(aliasPath, targetPath)) {
                    continue;
                }
                Files.deleteIfExists(aliasPath);
                try {
                    Files.createSymbolicLink(
                            aliasPath, Paths.get(targetPath.getFileName().toString()));
                } catch (IOException | UnsupportedOperationException
                         | SecurityException symbolicLinkError) {
                    Files.deleteIfExists(aliasPath);
                    try {
                        Files.createLink(aliasPath, targetPath);
                    } catch (FileAlreadyExistsException race) {
                        if (!Files.isSameFile(aliasPath, targetPath)) {
                            throw race;
                        }
                    } catch (IOException | UnsupportedOperationException
                             | SecurityException hardLinkError) {
                        hardLinkError.addSuppressed(symbolicLinkError);
                        throw new IllegalStateException(
                                "Cannot materialize runtime alias '"
                                        + mapping.getKey() + "' -> '"
                                        + mapping.getValue() + "' declared by "
                                        + source,
                                hardLinkError);
                    }
                }
                if (!Files.exists(aliasPath, LinkOption.NOFOLLOW_LINKS)
                        || !Files.isSameFile(aliasPath, targetPath)) {
                    throw new IllegalStateException(
                            "Runtime alias '" + mapping.getKey()
                                    + "' does not resolve to canonical runtime '"
                                    + mapping.getValue() + "'");
                }
            }
        }
    }

    private static final class RuntimeManifest {
        private final Set<String> runtimeNames;
        private final Map<String, String> runtimeAliases;
        private final Set<String> resourceNames;

        private RuntimeManifest(
                Set<String> runtimeNames,
                Map<String, String> runtimeAliases,
                Set<String> resourceNames) {
            this.runtimeNames = runtimeNames;
            this.runtimeAliases = runtimeAliases;
            this.resourceNames = resourceNames;
        }
    }

    private static String runtimeSpec(String runtimeName) {
        String alias = runtimeName.replaceAll("[^A-Za-z0-9]", "_");
        return "nd4j_compiler_runtime_" + alias + ":"
                + runtimeName + "#" + runtimeName;
    }
}
