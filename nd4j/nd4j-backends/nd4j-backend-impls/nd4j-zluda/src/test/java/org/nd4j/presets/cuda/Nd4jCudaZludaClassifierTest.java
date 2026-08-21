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

package org.nd4j.presets.cuda;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.annotation.Properties;
import org.junit.jupiter.api.Test;
import org.nd4j.presets.SharedCompilerRuntime;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Nd4jCudaZludaClassifierTest {

    @Test
    void discoversVersionedZludaManifestExtension() throws Exception {
        Properties preset =
                Nd4jCudaPresets.class.getAnnotation(Properties.class);
        assertNotNull(preset, "CUDA preset must declare JavaCPP properties");

        List<String> extensions = Arrays.stream(preset.value())
                .flatMap(platform -> Arrays.stream(platform.extension()))
                .collect(Collectors.toList());
        assertTrue(extensions.contains("-zluda-rocm-7.2.4"));
        assertTrue(extensions.contains("-zluda-rocm-6.2.4"));

        List<String> resources = Arrays.stream(preset.value())
                .flatMap(platform -> Arrays.stream(platform.resource()))
                .collect(Collectors.toList());
        assertTrue(resources.contains("rocblas/library"),
                "ZLUDA JavaCPP properties must extract rocBLAS resources");

        Path pomPath = Path.of("pom.xml");
        String pom = Files.readString(pomPath);
        if (!pom.contains("${javacpp.platform}${javacpp.platform.extension}/**</exclude>")) {
            pomPath = Path.of("nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda/pom.xml");
            pom = Files.readString(pomPath);
        }
        assertTrue(pom.contains("${javacpp.platform}${javacpp.platform.extension}/**</exclude>"),
                "Unclassified JAR must exclude the complete classifier tree");
        assertTrue(pom.contains("${javacpp.platform}${javacpp.platform.extension}/**</include>"),
                "Classifier JAR must include nested ROCm resources");

        ClassProperties properties = new ClassProperties();
        properties.addAll("platform.extension", extensions);

        String platform = "linux-x86_64";
        String selectedExtension = "-zluda-rocm-7.2.4";
        String selectedManifest = "org/nd4j/linalg/jcublas/bindings/"
                + platform + selectedExtension + "/"
                + SharedCompilerRuntime.MANIFEST_NAME;
        URL manifestUrl = new URL("file:/fixture/" + SharedCompilerRuntime.MANIFEST_NAME);
        ClassLoader resourceLoader = new ClassLoader(null) {
            @Override
            public URL getResource(String name) {
                return selectedManifest.equals(name) ? manifestUrl : null;
            }
        };

        assertEquals(
                selectedExtension,
                Nd4jCudaPresets.resolveBundledZludaExtension(
                        properties, platform, resourceLoader));
        assertTrue(Nd4jCudaPresets.isZludaExtension(selectedExtension));
        assertTrue(Nd4jCudaPresets.isZludaExtension("-zluda"));
    }
}
