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
import java.io.File;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Smoke tests for the (embryonic) Hexagon NPU backend wiring. See ADR 0088 (Hexagon MLIR
 * Backend) and ADR 0102 (Accelerator and CPU-Architecture CI Test Tiers).
 *
 * Selected by {@code -Ptest-hexagon} (group hexagon), which puts nd4j-hexagon on the classpath.
 * HexagonBackend.canRun() is currently a stub returning false until hexagon-mlir native bindings
 * land, so these tests validate the layers that exist today:
 *
 * 1. classpath + SPI registration of the Hexagon backend,
 * 2. that a hexagon-mlir runtime library can be dlopen'd in-process when one is available
 *    (point -Dhexagon.library.path / HEXAGON_MLIR_PATH / HEXAGON_SDK_ROOT at it),
 * 3. the HexagonEnvironment static device-info contract.
 *
 * hexagon-mlir is BSD-3 open source (Qualcomm, Dec 2025), so unlike libtpu the runtime can in
 * principle be built from source for CI. Until then the dlopen test skips when no library is
 * present. All nd4j-hexagon classes are accessed reflectively so this class compiles without the
 * nd4j-hexagon dependency on the classpath.
 */
@Slf4j
@Tag(TagNames.HEXAGON)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("Hexagon NPU backend wiring smoke tests (hexagon-mlir)")
public class HexagonBackendSmokeTest {

    private static final String HEXAGON_BACKEND_CLASS = "org.nd4j.linalg.hexagon.HexagonBackend";
    private static final String HEXAGON_ENV_CLASS = "org.nd4j.linalg.hexagon.HexagonEnvironment";
    private static final String BACKEND_SERVICE = "META-INF/services/org.nd4j.linalg.factory.Nd4jBackend";

    @Test
    @DisplayName("nd4j-hexagon is on the classpath and SPI-registered")
    public void hexagonBackendOnClasspathAndRegistered() throws Exception {
        Class<?> backendClass = assertDoesNotThrow(() -> Class.forName(HEXAGON_BACKEND_CLASS),
                "nd4j-hexagon is not on the classpath — run from platform-tests with -Ptest-hexagon "
                        + "(and install it first: mvn install -DskipTests -Phexagon -pl :nd4j-hexagon-preset,:nd4j-hexagon)");

        boolean registered = false;
        Enumeration<URL> serviceFiles = Thread.currentThread().getContextClassLoader().getResources(BACKEND_SERVICE);
        while (serviceFiles.hasMoreElements() && !registered) {
            URL url = serviceFiles.nextElement();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(url.openStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().equals(HEXAGON_BACKEND_CLASS)) {
                        registered = true;
                        log.info("HexagonBackend SPI registration found in {}", url);
                        break;
                    }
                }
            }
        }
        assertTrue(registered, "HexagonBackend is on the classpath but not registered in " + BACKEND_SERVICE
                + " — Nd4jBackend discovery can never select it");

        // Constructing and probing the backend must not throw even without NPU hardware
        Object backend = backendClass.getDeclaredConstructor().newInstance();
        Object canRun = backendClass.getMethod("canRun").invoke(backend);
        Object priority = backendClass.getMethod("getPriority").invoke(backend);
        log.info("HexagonBackend.canRun() = {} (stub returns false until hexagon-mlir bindings land), priority = {}",
                canRun, priority);
        assertEquals(Boolean.FALSE, canRun,
                "HexagonBackend.canRun() no longer returns the stub value — hexagon-mlir bindings appear to have "
                        + "landed. Extend this smoke test with real NPU enumeration and execution checks.");
    }

    @Test
    @DisplayName("hexagon-mlir runtime library loads in-process")
    public void hexagonRuntimeLibraryLoads() {
        String resolved = resolveHexagonLibrary();
        assumeTrue(resolved != null,
                "No hexagon-mlir runtime library found — set -Dhexagon.library.path or the HEXAGON_MLIR_PATH / "
                        + "HEXAGON_SDK_ROOT env vars to a runtime .so or its directory. Skipping.");

        final String libToLoad = resolved;
        assertDoesNotThrow(() -> System.load(libToLoad), "Failed to dlopen " + libToLoad);
        log.info("Successfully loaded hexagon-mlir runtime library in-process: {}", libToLoad);
    }

    @Test
    @DisplayName("HexagonEnvironment device-info contract")
    public void hexagonEnvironmentDeviceInfoContract() throws Exception {
        Class<?> envClass;
        try {
            envClass = Class.forName(HEXAGON_ENV_CLASS);
        } catch (ClassNotFoundException e) {
            assumeTrue(false, "nd4j-hexagon not on the classpath — skipping environment contract check");
            return;
        }

        Object env = envClass.getMethod("getInstance").invoke(null);
        String npuVersion = (String) envClass.getMethod("getNpuVersion").invoke(env);
        int hvxWidth = (Integer) envClass.getMethod("getHvxVectorWidth").invoke(env);
        long tcm = (Long) envClass.getMethod("getTcmCapacity").invoke(env);
        boolean int8 = (Boolean) envClass.getMethod("prefersInt8").invoke(env);

        log.info("HexagonEnvironment: npuVersion={}, hvxWidth={}B, tcm={}KB, prefersInt8={}",
                npuVersion, hvxWidth, tcm / 1024, int8);
        assertEquals(128, hvxWidth, "HVX vector width is architecturally 128 bytes");
        assertTrue(tcm > 0, "TCM capacity must be positive");
        assertTrue(int8, "Hexagon NPUs are INT8-first");
    }

    /**
     * Resolve a loadable hexagon-mlir runtime library from system properties and env vars.
     * Accepts a direct .so path or a directory containing one. Values with unresolved Maven
     * placeholders ("${...}") are treated as unset.
     */
    private String resolveHexagonLibrary() {
        List<String> candidates = new ArrayList<>();
        candidates.add(System.getProperty("hexagon.library.path"));
        candidates.add(System.getenv("HEXAGON_MLIR_PATH"));
        candidates.add(System.getenv("HEXAGON_SDK_ROOT"));

        for (String candidate : candidates) {
            if (candidate == null || candidate.isEmpty() || candidate.contains("${"))
                continue;
            File f = new File(candidate);
            if (f.isFile()) {
                log.info("hexagon-mlir library candidate (direct file): {}", f.getAbsolutePath());
                return f.getAbsolutePath();
            }
            if (f.isDirectory()) {
                for (String known : new String[]{"libhexagon_mlir_runtime.so", "libnd4jhexagon.so"}) {
                    File lib = new File(f, known);
                    if (lib.isFile()) {
                        log.info("hexagon-mlir library candidate (in directory {}): {}", f, lib.getAbsolutePath());
                        return lib.getAbsolutePath();
                    }
                }
                File[] anyHexagon = f.listFiles((dir, name) ->
                        name.endsWith(".so") && name.contains("hexagon"));
                if (anyHexagon != null && anyHexagon.length > 0) {
                    log.info("hexagon-mlir library candidate (pattern match in {}): {}", f, anyHexagon[0].getAbsolutePath());
                    return anyHexagon[0].getAbsolutePath();
                }
                log.warn("Directory {} contains no recognizable hexagon-mlir runtime library", f);
            } else {
                log.warn("hexagon-mlir candidate path does not exist: {}", candidate);
            }
        }
        return null;
    }
}
