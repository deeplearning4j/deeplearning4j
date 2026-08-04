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

package org.nd4j.presets.cpu;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Nd4jCpuPresetsTest {

    @Test
    void linksGccRuntimeOnlyForLinuxArm64() {
        List<String> arm64Links = platformLinks("linux-arm64");
        assertTrue(arm64Links.contains("nd4jcpu"));
        assertTrue(arm64Links.contains("dl"));
        assertTrue(arm64Links.contains("gcc"));

        for (String platform : List.of("linux-x86_64", "android-arm64", "macosx-arm64")) {
            assertFalse(platformLinks(platform).contains("gcc"), platform);
        }
    }

    private static List<String> platformLinks(String platform) {
        Properties platformProperties = new Properties();
        platformProperties.putAll(Loader.loadProperties());
        platformProperties.setProperty("platform", platform);
        platformProperties.setProperty("platform.extension", "");

        org.bytedeco.javacpp.annotation.Properties preset =
                Nd4jCpuPresets.class.getAnnotation(org.bytedeco.javacpp.annotation.Properties.class);
        List<String> links = new ArrayList<>();
        for (Platform configuration : preset.value()) {
            if (Loader.checkPlatform(configuration, platformProperties)) {
                links.addAll(Arrays.asList(configuration.link()));
            }
        }
        return links;
    }
}
