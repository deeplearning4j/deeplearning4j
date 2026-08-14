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

package org.eclipse.deeplearning4j.nd4j.nativeimage;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TypicalPFilter;
import org.nd4j.linalg.api.ops.impl.transforms.custom.XtcFilter;
import org.nd4j.nativeimage.Nd4jJavaCppClassScanner;
import org.nd4j.nativeimage.Nd4jOperationClassScanner;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Nd4jOperationClassScannerTest {

    private static final String NATIVE_IMAGE_PROPERTIES =
            "META-INF/native-image/org.eclipse.deeplearning4j/"
                    + "nd4j-api/native-image.properties";

    @Test
    public void discoversTheNd4jOperationCatalog() {
        Set<Class<? extends DifferentialFunction>> operations =
                Nd4jOperationClassScanner.discover();

        assertTrue(operations.size() >= 800,
                "Expected the complete ND4J operation catalog, found "
                        + operations.size());
        assertTrue(operations.contains(TypicalPFilter.class),
                "TypicalPFilter must be discovered automatically");
        assertTrue(operations.contains(XtcFilter.class),
                "XtcFilter must be discovered automatically");
        assertTrue(operations.stream()
                        .allMatch(DifferentialFunction.class::isAssignableFrom),
                "Every discovered class must be a DifferentialFunction");
    }

    @Test
    public void discoversCompleteCpuAndSharedJavaCppBindingFamilies() throws Exception {
        List<Path> classpath = Arrays.stream(
                        System.getProperty("java.class.path").split(
                                Pattern.quote(File.pathSeparator)))
                .filter(entry -> !entry.isEmpty())
                .map(Path::of)
                .collect(Collectors.toList());

        Set<Class<?>> bindings = Nd4jJavaCppClassScanner.discoverBindingClasses(
                classpath,
                Thread.currentThread().getContextClassLoader());
        Set<String> bindingNames = bindings.stream()
                .map(Class::getName)
                .collect(Collectors.toSet());

        assertTrue(bindingNames.contains("org.nd4j.nativeblas.StringVector"),
                "Shared JavaCPP wrappers must not depend on a backend list");
        assertTrue(bindingNames.contains("org.nd4j.nativeblas.StringVector$Iterator"),
                "Nested shared JavaCPP wrappers must be discovered");

        if (isPresent("org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu")) {
            assertTrue(bindingNames.contains(
                            "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu"),
                    "CPU outer binding must be discovered");
            assertTrue(bindingNames.contains(
                            "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu$Environment"),
                    "CPU Environment JNI binding must be discovered");
            assertTrue(bindingNames.contains(
                            "org.nd4j.linalg.cpu.nativecpu.bindings."
                                    + "Nd4jCpu$ConstNDArrayVector$Iterator"),
                    "Doubly nested CPU JNI binding must be discovered");
        }
    }

    @Test
    public void nd4jApiActivatesItsHostedReachabilityFeature() throws IOException {
        Properties properties = new Properties();
        try (InputStream input = Nd4jOperationClassScannerTest.class
                .getClassLoader().getResourceAsStream(NATIVE_IMAGE_PROPERTIES)) {
            assertNotNull(input,
                    "Missing ND4J native-image properties: "
                            + NATIVE_IMAGE_PROPERTIES);
            properties.load(input);
        }

        String args = properties.getProperty("Args");
        assertNotNull(args, "ND4J native-image properties must define Args");
        assertTrue(args.contains(
                        "--features=org.nd4j.nativeimage.Nd4jOpsReflectionFeature"),
                "nd4j-api must activate its operation and JavaCPP reachability feature");
    }

    private static boolean isPresent(String className) {
        try {
            Class.forName(
                    className,
                    false,
                    Thread.currentThread().getContextClassLoader());
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
