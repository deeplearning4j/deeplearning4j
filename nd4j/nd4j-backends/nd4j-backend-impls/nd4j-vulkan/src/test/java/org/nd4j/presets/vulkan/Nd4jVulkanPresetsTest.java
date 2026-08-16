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

package org.nd4j.presets.vulkan;

import org.bytedeco.javacpp.annotation.Properties;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Nd4jVulkanPresetsTest {

    @Test
    void declaresCompileClassifierExtension() {
        Properties preset =
                Nd4jVulkanPresets.class.getAnnotation(Properties.class);
        assertNotNull(preset, "Vulkan preset must declare JavaCPP properties");

        Set<String> extensions = Arrays.stream(preset.value())
                .flatMap(platform -> Arrays.stream(platform.extension()))
                .collect(Collectors.toSet());

        assertTrue(
                extensions.contains("-compile"),
                "The Vulkan MLIR classifier is packaged below "
                        + "linux-x86_64-compile and must be discoverable by JavaCPP");
    }
}
