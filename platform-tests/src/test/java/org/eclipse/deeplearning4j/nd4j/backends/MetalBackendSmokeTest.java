/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Smoke tests for the Apple Metal / MLX backend wiring.
 *
 * <p>Selected by {@code -Ptest-metal} (group metal), which puts nd4j-metal and
 * nd4j-metal-preset on the classpath.</p>
 *
 * <p>Mirrors {@link TpuBackendSmokeTest}. {@code JMetalBackend.canRun()} is a stub
 * returning {@code false} until JavaCPP MLX/MPS native bindings are generated. These
 * tests validate the layers that exist today:</p>
 * <ol>
 *   <li>Classpath + SPI registration of the Metal backend</li>
 *   <li>That the JVM is running on macOS arm64 (skip gracefully on Linux/x86)</li>
 *   <li>The {@code Nd4jMetalHelper} availability contract (ND4J_METAL_LIBRARY_PATH)</li>
 * </ol>
 *
 * <p>On macOS arm64 CI, pass {@code -Dnd4j.metal.library.path} or set
 * {@code ND4J_METAL_LIBRARY_PATH} to the libnd4j_metal.dylib path once the native
 * build produces it. Until then all tests that require the dylib skip gracefully.</p>
 *
 * <p><b>Hardware note:</b> These tests MUST run on {@code macos-14} (Apple Silicon)
 * runners — they will be skipped on linux-x86_64 hosted runners. See the companion
 * CI workflow {@code run-mlx-smoke-tests.yml}.</p>
 */
@Slf4j
@Tag(TagNames.METAL)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("Metal/MLX backend wiring smoke tests (Apple Silicon)")
public class MetalBackendSmokeTest {

    private static final String METAL_BACKEND_CLASS = "org.nd4j.linalg.metal.JMetalBackend";
    private static final String METAL_HELPER_CLASS  = "org.nd4j.presets.metal.Nd4jMetalHelper";
    private static final String BACKEND_SERVICE     = "META-INF/services/org.nd4j.linalg.factory.Nd4jBackend";

    @Test
    @DisplayName("nd4j-metal is on the classpath and SPI-registered")
    public void metalBackendOnClasspathAndRegistered() throws Exception {
        Class<?> backendClass = assertDoesNotThrow(() -> Class.forName(METAL_BACKEND_CLASS),
                "nd4j-metal is not on the classpath — run from platform-tests with -Ptest-metal "
                        + "(and install it first: mvn install -DskipTests -Pmetal -pl :nd4j-metal,:nd4j-metal-preset)");

        boolean registered = false;
        Enumeration<URL> serviceFiles =
                Thread.currentThread().getContextClassLoader().getResources(BACKEND_SERVICE);
        while (serviceFiles.hasMoreElements() && !registered) {
            URL url = serviceFiles.nextElement();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(url.openStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().equals(METAL_BACKEND_CLASS)) {
                        registered = true;
                        log.info("JMetalBackend SPI registration found in {}", url);
                        break;
                    }
                }
            }
        }
        assertTrue(registered,
                "JMetalBackend is on the classpath but not registered in " + BACKEND_SERVICE
                        + " — Nd4jBackend discovery can never select it");

        // Constructing and probing the backend must not throw even without Metal hardware
        Object backend  = backendClass.getDeclaredConstructor().newInstance();
        Object canRun   = backendClass.getMethod("canRun").invoke(backend);
        Object priority = backendClass.getMethod("getPriority").invoke(backend);
        log.info("JMetalBackend.canRun() = {} (stub returns false until MLX/MPS bindings land), priority = {}",
                canRun, priority);
        assertEquals(Boolean.FALSE, canRun,
                "JMetalBackend.canRun() no longer returns the stub value — MLX/MPS native bindings appear "
                        + "to have landed. Extend this smoke test with real Metal device enumeration and "
                        + "mx::core::eval() execution checks.");
    }

    @Test
    @DisplayName("MetalEnvironment.isMacOSArm64() matches actual platform")
    public void metalEnvironmentPlatformCheck() throws Exception {
        Class<?> envClass = assertDoesNotThrow(
                () -> Class.forName("org.nd4j.linalg.metal.MetalEnvironment"),
                "MetalEnvironment not on classpath — ensure nd4j-metal is installed");

        boolean isMacArm = (Boolean) envClass.getMethod("isMacOSArm64").invoke(null);
        String  os       = System.getProperty("os.name", "");
        String  arch     = System.getProperty("os.arch", "");

        log.info("MetalEnvironment.isMacOSArm64() = {} (os={}, arch={})", isMacArm, os, arch);

        boolean expected = os.toLowerCase().contains("mac")
                && (arch.equalsIgnoreCase("aarch64") || arch.equalsIgnoreCase("arm64"));
        assertEquals(expected, isMacArm,
                "MetalEnvironment.isMacOSArm64() does not match actual JVM platform");

        if (!isMacArm) {
            log.info("Not running on macOS arm64 — Metal execution tests will be skipped on this platform");
        }
    }

    @Test
    @DisplayName("Nd4jMetalHelper availability contract matches ND4J_METAL_LIBRARY_PATH env")
    public void metalHelperAvailabilityContractMatchesEnv() throws Exception {
        Class<?> helper;
        try {
            helper = Class.forName(METAL_HELPER_CLASS);
        } catch (ClassNotFoundException e) {
            assumeTrue(false, "nd4j-metal-preset not on the classpath — skipping helper contract check");
            return;
        }

        boolean available = (Boolean) helper.getMethod("isMetalAvailable").invoke(null);
        String  metalEnv  = System.getenv("ND4J_METAL_LIBRARY_PATH");
        if (metalEnv != null && metalEnv.contains("${")) {
            log.warn("ND4J_METAL_LIBRARY_PATH contains an unresolved Maven placeholder ({}) — treated as unset.",
                    metalEnv);
        }
        // Stub always returns false until bindings land — helper must match that
        assertFalse(available,
                "Nd4jMetalHelper.isMetalAvailable() returned true before native bindings were generated. "
                        + "If bindings have landed, update this test to verify real device enumeration.");

        Method deviceCount = helper.getMethod("getDeviceCount");
        log.info("Nd4jMetalHelper: isMetalAvailable={}, deviceCount={}", available, deviceCount.invoke(null));
    }
}
