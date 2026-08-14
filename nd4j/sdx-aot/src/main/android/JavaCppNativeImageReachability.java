/*
 * Copyright (c) Eclipse Foundation.
 *
 * This program is made available under the terms of the Apache License 2.0.
 *
 * Build-time generator for JavaCPP classes loaded reflectively or through JNI.
 * It intentionally depends only on the JDK so it can run in source-file mode
 * against the exact, already-resolved target classpath.
 */

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * Resolves the complete ND4J-owned JavaCPP binding closure and the classes
 * JavaCPP Loader.load() reaches, then emits identical GraalVM reflection and
 * JNI class catalogs.
 *
 * Usage:
 *   java -Dorg.bytedeco.javacpp.platform=android-arm64
 *     JavaCppNativeImageReachability.java
 *     CLASSPATH_FILE REFLECT_CONFIG JNI_CONFIG INITIALIZATION_CONFIG OUTPUT_MANIFEST
 */
public final class JavaCppNativeImageReachability {
    private static final String EXPECTED_PLATFORM = "android-arm64";
    private static final String SCANNER_CLASS =
            "org.nd4j.nativeimage.Nd4jJavaCppClassScanner";

    private JavaCppNativeImageReachability() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            throw new IllegalArgumentException(
                    "Usage: java -Dorg.bytedeco.javacpp.platform=" + EXPECTED_PLATFORM
                            + " JavaCppNativeImageReachability.java "
                            + "CLASSPATH_FILE REFLECT_CONFIG JNI_CONFIG "
                            + "INITIALIZATION_CONFIG OUTPUT_MANIFEST");
        }

        Path classpathFile = Path.of(args[0]).toAbsolutePath().normalize();
        Path reflectConfig = Path.of(args[1]).toAbsolutePath().normalize();
        Path jniConfig = Path.of(args[2]).toAbsolutePath().normalize();
        Path initializationConfig = Path.of(args[3]).toAbsolutePath().normalize();
        Path outputManifest = Path.of(args[4]).toAbsolutePath().normalize();
        List<Path> classpath = readClasspath(classpathFile);
        List<URL> classpathUrls = new ArrayList<>();
        for (Path entry : classpath) {
            classpathUrls.add(entry.toUri().toURL());
        }

        ClassLoader previousContextLoader = Thread.currentThread().getContextClassLoader();
        try (URLClassLoader targetLoader =
                     new URLClassLoader(classpathUrls.toArray(URL[]::new), ClassLoader.getPlatformClassLoader())) {
            Thread.currentThread().setContextClassLoader(targetLoader);
            generate(
                    classpath,
                    targetLoader,
                    reflectConfig,
                    jniConfig,
                    initializationConfig,
                    outputManifest);
        } finally {
            Thread.currentThread().setContextClassLoader(previousContextLoader);
        }
    }

    private static List<Path> readClasspath(Path classpathFile) throws Exception {
        String rawClasspath = Files.readString(classpathFile, StandardCharsets.UTF_8).trim();
        if (rawClasspath.isEmpty()) {
            throw new IllegalStateException("Target classpath is empty: " + classpathFile);
        }

        List<Path> paths = new ArrayList<>();
        for (String entry : rawClasspath.split(Pattern.quote(File.pathSeparator), -1)) {
            if (entry.isEmpty()) {
                throw new IllegalStateException("Target classpath contains an empty entry: " + classpathFile);
            }
            Path path = Path.of(entry).toAbsolutePath().normalize();
            if (!Files.exists(path)) {
                throw new IllegalStateException("Target classpath entry does not exist: " + path);
            }
            paths.add(path);
        }
        return Collections.unmodifiableList(paths);
    }

    @SuppressWarnings("unchecked")
    private static void generate(
            List<Path> classpath,
            ClassLoader targetLoader,
            Path reflectConfig,
            Path jniConfig,
            Path initializationConfig,
            Path outputManifest) throws Exception {
        Class<?> scanner = Class.forName(SCANNER_CLASS, false, targetLoader);
        Method discoverBindings =
                scanner.getMethod("discoverBindingClasses", List.class, ClassLoader.class);
        Method topLevelClasses = scanner.getMethod("topLevelClasses", Set.class);

        Set<Class<?>> bindingClasses =
                (Set<Class<?>>) discoverBindings.invoke(null, classpath, targetLoader);
        Set<Class<?>> bindingRoots =
                (Set<Class<?>>) topLevelClasses.invoke(null, bindingClasses);
        if (bindingClasses.isEmpty() || bindingRoots.isEmpty()) {
            throw new IllegalStateException("ND4J JavaCPP binding closure is empty");
        }

        Class<?> javaCppLoader = Class.forName("org.bytedeco.javacpp.Loader", false, targetLoader);
        Method loadDefaultProperties = javaCppLoader.getMethod("loadProperties");
        Method loadClassProperties =
                javaCppLoader.getMethod("loadProperties", Class.class, Properties.class, boolean.class);

        Properties platformProperties = (Properties) loadDefaultProperties.invoke(null);
        String resolvedPlatform = platformProperties.getProperty("platform");
        if (!EXPECTED_PLATFORM.equals(resolvedPlatform)) {
            throw new IllegalStateException(
                    "Expected JavaCPP platform " + EXPECTED_PLATFORM + ", resolved " + resolvedPlatform);
        }

        Map<String, Set<String>> targetsByRoot = new LinkedHashMap<>();
        Set<String> reachableClasses = new TreeSet<>();
        for (Class<?> bindingClass : bindingClasses) {
            reachableClasses.add(bindingClass.getName());
        }

        for (Class<?> rootClass : bindingRoots) {
            Object classProperties =
                    loadClassProperties.invoke(null, rootClass, platformProperties, true);
            Set<String> targets = javaCppLoaderTargets(classProperties, rootClass);
            for (String target : targets) {
                Class.forName(target, false, targetLoader);
            }
            targetsByRoot.put(rootClass.getName(), targets);
            reachableClasses.addAll(targets);
        }

        String classMetadata = classMetadataJson(reachableClasses);
        writeAtomically(reflectConfig, classMetadata);
        writeAtomically(jniConfig, classMetadata);
        writeAtomically(initializationConfig, initializationProperties(reachableClasses));
        writeAtomically(
                outputManifest,
                reachabilityManifest(targetsByRoot, reachableClasses));
    }

    @SuppressWarnings("unchecked")
    private static Set<String> javaCppLoaderTargets(Object classProperties, Class<?> rootClass)
            throws Exception {
        Class<?> classPropertiesClass = classProperties.getClass();
        Method get = classPropertiesClass.getMethod("get", String.class);
        Method getInheritedClasses = classPropertiesClass.getMethod("getInheritedClasses");

        LinkedHashSet<String> targets = new LinkedHashSet<>();
        List<String> globals = (List<String>) get.invoke(classProperties, "global");
        if (globals != null) {
            for (String global : globals) {
                addClassName(targets, global);
            }
        }
        if (targets.isEmpty()) {
            Class<?>[] inheritedClasses = (Class<?>[]) getInheritedClasses.invoke(classProperties);
            if (inheritedClasses != null) {
                for (Class<?> inheritedClass : inheritedClasses) {
                    addClassName(targets, inheritedClass.getName());
                }
            }
        }
        // Loader.load(Class, ...) may resolve a separate generated global class,
        // but the binding root itself remains part of the JNI/reflection contract.
        addClassName(targets, rootClass.getName());
        return Collections.unmodifiableSet(new TreeSet<>(targets));
    }

    private static void addClassName(Set<String> targets, String className) {
        if (className != null) {
            String trimmed = className.trim();
            if (!trimmed.isEmpty()) {
                targets.add(trimmed);
            }
        }
    }

    private static String classMetadataJson(Set<String> classes) {
        StringBuilder json = new StringBuilder();
        json.append("[\n");
        int index = 0;
        for (String className : classes) {
            if (index++ > 0) {
                json.append(",\n");
            }
            json.append("  {\n")
                    .append("    \"name\": \"")
                    .append(jsonEscape(className))
                    .append("\",\n")
                    .append("    \"allDeclaredMethods\": true,\n")
                    .append("    \"allDeclaredConstructors\": true,\n")
                    .append("    \"allDeclaredFields\": true\n")
                    .append("  }");
        }
        json.append("\n]\n");
        return json.toString();
    }

    private static String initializationProperties(Set<String> classes) {
        return "Args = --initialize-at-run-time=" + String.join(",", classes) + "\n";
    }

    private static String reachabilityManifest(
            Map<String, Set<String>> targetsByRoot,
            Set<String> reachableClasses) {
        StringBuilder manifest = new StringBuilder();
        manifest.append("format=2\n");
        manifest.append("platform=").append(EXPECTED_PLATFORM).append('\n');
        for (Map.Entry<String, Set<String>> entry : targetsByRoot.entrySet()) {
            manifest.append("root=").append(entry.getKey()).append('\n');
            for (String target : entry.getValue()) {
                manifest.append("target=")
                        .append(entry.getKey())
                        .append(' ')
                        .append(target)
                        .append('\n');
            }
        }
        for (String className : reachableClasses) {
            manifest.append("reflection-class=").append(className).append('\n');
        }
        for (String className : reachableClasses) {
            manifest.append("jni-class=").append(className).append('\n');
        }
        return manifest.toString();
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static void writeAtomically(Path output, String content) throws IOException {
        Path parent = output.getParent();
        if (parent == null) {
            throw new IllegalArgumentException("Output path has no parent: " + output);
        }
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, output.getFileName().toString(), ".tmp");
        try {
            Files.writeString(temporary, content, StandardCharsets.UTF_8);
            try {
                Files.move(
                        temporary,
                        output,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException ignored) {
                Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }
}
