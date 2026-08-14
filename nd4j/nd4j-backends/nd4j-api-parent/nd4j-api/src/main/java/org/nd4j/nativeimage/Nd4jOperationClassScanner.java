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

import org.nd4j.autodiff.functions.DifferentialFunction;

import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Discovers the Java operation implementations shipped by {@code nd4j-api}.
 *
 * <p>ND4J constructs these classes reflectively through
 * {@code DifferentialFunctionClassHolder}. Keeping discovery next to the
 * artifact avoids a second, manually maintained list of operation classes for
 * native-image builds.</p>
 */
public final class Nd4jOperationClassScanner {

    private static final String OPS_PREFIX = "org/nd4j/linalg/api/ops/";

    private Nd4jOperationClassScanner() {
    }

    /**
     * Discover every ND4J operation class visible from the defining
     * {@link DifferentialFunction} artifact.
     *
     * @return an immutable set of operation classes
     */
    public static Set<Class<? extends DifferentialFunction>> discover() {
        ClassLoader loader = DifferentialFunction.class.getClassLoader();
        if (loader == null) {
            loader = Thread.currentThread().getContextClassLoader();
        }
        if (loader == null) {
            loader = ClassLoader.getSystemClassLoader();
        }
        return discover(loader);
    }

    static Set<Class<? extends DifferentialFunction>> discover(ClassLoader loader) {
        Set<String> classNames = new LinkedHashSet<>();
        URL location = codeSourceLocation(DifferentialFunction.class);
        if (location != null) {
            collectFromLocation(location, classNames);
        }
        if (classNames.isEmpty()) {
            collectFromClassPath(classNames);
        }
        if (classNames.isEmpty()) {
            throw new IllegalStateException(
                    "No ND4J operation classes found in " + location);
        }

        Set<Class<? extends DifferentialFunction>> operations = new LinkedHashSet<>();
        for (String className : classNames) {
            try {
                Class<?> candidate = Class.forName(className, false, loader);
                if (DifferentialFunction.class.isAssignableFrom(candidate)) {
                    operations.add(candidate.asSubclass(DifferentialFunction.class));
                }
            } catch (ClassNotFoundException | LinkageError e) {
                throw new IllegalStateException(
                        "Unable to inspect ND4J operation class " + className, e);
            }
        }
        if (operations.isEmpty()) {
            throw new IllegalStateException(
                    "No DifferentialFunction implementations found in " + location);
        }
        return Collections.unmodifiableSet(operations);
    }

    private static URL codeSourceLocation(Class<?> type) {
        try {
            ProtectionDomain domain = type.getProtectionDomain();
            CodeSource source = domain != null ? domain.getCodeSource() : null;
            return source != null ? source.getLocation() : null;
        } catch (SecurityException e) {
            return null;
        }
    }

    private static void collectFromLocation(URL location, Set<String> names) {
        try {
            File file = new File(location.toURI());
            if (file.isFile()) {
                collectFromJar(file, names);
            } else if (file.isDirectory()) {
                collectFromDirectory(file.toPath(), file, names);
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Unable to scan ND4J operation classes from " + location, e);
        }
    }

    private static void collectFromClassPath(Set<String> names) {
        String classPath = System.getProperty("java.class.path", "");
        for (String entry : classPath.split(File.pathSeparator)) {
            if (entry.isEmpty()) {
                continue;
            }
            File file = new File(entry);
            if (file.isFile() && entry.endsWith(".jar")) {
                collectFromJar(file, names);
            } else if (file.isDirectory()) {
                collectFromDirectory(file.toPath(), file, names);
            }
        }
    }

    private static void collectFromJar(File jar, Set<String> names) {
        try (ZipFile zip = new ZipFile(jar)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                String entryName = entries.nextElement().getName();
                if (isOperationClassEntry(entryName)) {
                    names.add(entryToClassName(entryName));
                }
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Unable to scan ND4J operation classes from " + jar, e);
        }
    }

    private static void collectFromDirectory(
            Path root, File directory, Set<String> names) {
        File[] children = directory.listFiles();
        if (children == null) {
            throw new IllegalStateException(
                    "Unable to list ND4J class directory " + directory);
        }
        for (File child : children) {
            if (child.isDirectory()) {
                collectFromDirectory(root, child, names);
            } else {
                String relative = root.relativize(child.toPath()).toString()
                        .replace(File.separatorChar, '/');
                if (isOperationClassEntry(relative)) {
                    names.add(entryToClassName(relative));
                }
            }
        }
    }

    static boolean isOperationClassEntry(String entryName) {
        return entryName.startsWith(OPS_PREFIX)
                && entryName.endsWith(".class")
                && !entryName.endsWith("module-info.class")
                && !entryName.endsWith("package-info.class");
    }

    static String entryToClassName(String entryName) {
        return entryName.substring(0, entryName.length() - ".class".length())
                .replace('/', '.');
    }
}
