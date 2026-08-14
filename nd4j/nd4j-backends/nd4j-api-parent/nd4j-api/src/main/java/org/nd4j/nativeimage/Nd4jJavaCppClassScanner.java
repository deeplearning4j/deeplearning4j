/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  * *****************************************************************************
 */

package org.nd4j.nativeimage;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.TreeMap;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Discovers ND4J-owned JavaCPP/JNI binding families from the effective
 * Native Image application classpath.
 *
 * <p>The generated backend bindings contain hundreds of nested pointer and
 * callback types. A manually maintained list of only the outer binding class is
 * not a valid reachability contract: JavaCPP resolves those nested classes
 * through JNI and reflection as soon as a corresponding native signature is
 * used. This scanner derives the complete recursive family from the actual
 * classpath, so CPU, CUDA, minimal, Vulkan, SDX, LiteRT-LM, tokenizers, and
 * future bindings all follow the same rule.</p>
 */
public final class Nd4jJavaCppClassScanner {

    private static final List<String> OWNED_CLASS_PREFIXES = List.of(
            "org/nd4j/",
            "org/eclipse/deeplearning4j/");

    private static final byte[] JAVACPP_MARKER =
            "org/bytedeco/javacpp/".getBytes(StandardCharsets.US_ASCII);

    private Nd4jJavaCppClassScanner() {
    }

    /**
     * Discover the complete recursive set of ND4J-owned JavaCPP binding
     * classes visible on {@code applicationClasspath}.
     *
     * @param applicationClasspath the exact Native Image application classpath
     * @param loader the matching application class loader
     * @return an immutable, class-name-sorted binding closure
     */
    public static Set<Class<?>> discoverBindingClasses(
            List<Path> applicationClasspath, ClassLoader loader) {
        if (applicationClasspath == null || applicationClasspath.isEmpty()) {
            throw new IllegalArgumentException("Native Image application classpath is empty");
        }
        if (loader == null) {
            throw new IllegalArgumentException("Native Image application class loader is null");
        }

        Set<String> candidates = new LinkedHashSet<>();
        for (Path entry : applicationClasspath) {
            collectCandidates(entry.toAbsolutePath().normalize(), candidates);
        }

        TreeMap<String, Class<?>> topLevelBindings = new TreeMap<>();
        for (String className : candidates) {
            Class<?> candidate = load(className, loader);
            if (isJavaCppBinding(candidate)) {
                Class<?> root = topLevelClass(candidate);
                topLevelBindings.put(root.getName(), root);
            }
        }
        if (topLevelBindings.isEmpty()) {
            throw new IllegalStateException(
                    "No ND4J-owned JavaCPP binding classes found on the Native Image classpath");
        }

        TreeMap<String, Class<?>> closure = new TreeMap<>();
        for (Class<?> root : topLevelBindings.values()) {
            addRecursiveClassFamily(root, closure);
        }
        return Collections.unmodifiableSet(new LinkedHashSet<>(closure.values()));
    }

    /**
     * Return the top-level binding roots represented by a discovered closure.
     */
    public static Set<Class<?>> topLevelClasses(Set<Class<?>> bindingClasses) {
        TreeMap<String, Class<?>> roots = new TreeMap<>();
        for (Class<?> bindingClass : bindingClasses) {
            Class<?> root = topLevelClass(bindingClass);
            roots.put(root.getName(), root);
        }
        return Collections.unmodifiableSet(new LinkedHashSet<>(roots.values()));
    }

    private static void collectCandidates(Path classpathEntry, Set<String> candidates) {
        if (!Files.exists(classpathEntry)) {
            throw new IllegalStateException(
                    "Native Image classpath entry does not exist: " + classpathEntry);
        }
        try {
            if (Files.isDirectory(classpathEntry)) {
                collectDirectoryCandidates(classpathEntry, candidates);
            } else if (classpathEntry.toString().endsWith(".jar")
                    || classpathEntry.toString().endsWith(".zip")) {
                collectArchiveCandidates(classpathEntry, candidates);
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Unable to inspect JavaCPP bindings in " + classpathEntry, e);
        }
    }

    private static void collectDirectoryCandidates(Path root, Set<String> candidates)
            throws IOException {
        for (String prefix : OWNED_CLASS_PREFIXES) {
            Path packageRoot = root.resolve(prefix);
            if (!Files.isDirectory(packageRoot)) {
                continue;
            }
            try (java.util.stream.Stream<Path> files = Files.walk(packageRoot)) {
                java.util.Iterator<Path> iterator = files
                        .filter(Files::isRegularFile)
                        .filter(path -> isClassEntry(root.relativize(path).toString()))
                        .iterator();
                while (iterator.hasNext()) {
                    Path classFile = iterator.next();
                    byte[] bytes = Files.readAllBytes(classFile);
                    if (contains(bytes, JAVACPP_MARKER)) {
                        candidates.add(entryToClassName(
                                root.relativize(classFile).toString()));
                    }
                }
            }
        }
    }

    private static void collectArchiveCandidates(Path archive, Set<String> candidates)
            throws IOException {
        try (JarFile jar = new JarFile(archive.toFile())) {
            Enumeration<JarEntry> entries = jar.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                String name = entry.getName();
                if (entry.isDirectory() || !isClassEntry(name)) {
                    continue;
                }
                try (InputStream input = jar.getInputStream(entry)) {
                    if (contains(input.readAllBytes(), JAVACPP_MARKER)) {
                        candidates.add(entryToClassName(name));
                    }
                }
            }
        }
    }

    private static boolean isClassEntry(String rawEntryName) {
        String entryName = rawEntryName.replace('\\', '/');
        boolean owned = false;
        for (String prefix : OWNED_CLASS_PREFIXES) {
            if (entryName.startsWith(prefix)) {
                owned = true;
                break;
            }
        }
        return owned
                && entryName.endsWith(".class")
                && !entryName.endsWith("module-info.class")
                && !entryName.endsWith("package-info.class");
    }

    private static String entryToClassName(String rawEntryName) {
        String entryName = rawEntryName.replace('\\', '/');
        return entryName.substring(0, entryName.length() - ".class".length())
                .replace('/', '.');
    }

    private static boolean isJavaCppBinding(Class<?> candidate) {
        for (Class<?> type = candidate; type != null; type = type.getSuperclass()) {
            String name = type.getName();
            if ("org.bytedeco.javacpp.Pointer".equals(name)
                    || "org.bytedeco.javacpp.FunctionPointer".equals(name)
                    || name.startsWith("org.bytedeco.javacpp.")) {
                return true;
            }
        }
        try {
            for (Method method : candidate.getDeclaredMethods()) {
                if (Modifier.isNative(method.getModifiers())) {
                    return true;
                }
            }
            return false;
        } catch (LinkageError e) {
            throw new IllegalStateException(
                    "Unable to inspect JavaCPP candidate " + candidate.getName(), e);
        }
    }

    private static Class<?> topLevelClass(Class<?> type) {
        Class<?> root = type;
        while (root.getEnclosingClass() != null) {
            root = root.getEnclosingClass();
        }
        return root;
    }

    private static void addRecursiveClassFamily(
            Class<?> root, TreeMap<String, Class<?>> closure) {
        Queue<Class<?>> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Class<?> current = pending.remove();
            if (closure.putIfAbsent(current.getName(), current) != null) {
                continue;
            }
            try {
                Collections.addAll(pending, current.getDeclaredClasses());
            } catch (LinkageError e) {
                throw new IllegalStateException(
                        "Unable to inspect nested JavaCPP bindings of " + current.getName(), e);
            }
        }
    }

    private static Class<?> load(String className, ClassLoader loader) {
        try {
            return Class.forName(className, false, loader);
        } catch (ClassNotFoundException | LinkageError e) {
            throw new IllegalStateException(
                    "Unable to load JavaCPP binding candidate " + className, e);
        }
    }

    private static boolean contains(byte[] haystack, byte[] needle) {
        outer:
        for (int index = 0; index <= haystack.length - needle.length; index++) {
            for (int offset = 0; offset < needle.length; offset++) {
                if (haystack[index + offset] != needle[offset]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }
}
