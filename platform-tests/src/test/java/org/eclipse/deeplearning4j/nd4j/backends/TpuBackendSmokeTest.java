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
import java.lang.reflect.Method;
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
 * Smoke tests for the (embryonic) TPU backend wiring. See ADR 0072 (TPU Backend)
 * and ADR 0102 (Accelerator and CPU-Architecture CI Test Tiers).
 *
 * Selected by {@code -Ptest-tpu} (group tpu), which puts nd4j-tpu on the classpath.
 * JTpuBackend.canRun() is currently a stub returning false until PJRT native
 * bindings land, so these tests validate the layers that exist today:
 *
 * 1. classpath + SPI registration of the TPU backend,
 * 2. that a PJRT-compatible library (libtpu.so on a TPU VM, or the CPU PJRT
 *    plugin/libtpu wheel .so on ordinary x86 CI) can be dlopen'd in-process,
 * 3. the Nd4jTpuHelper availability contract (PJRT_PATH env).
 *
 * Point the library check at a real .so via -Dpjrt.path / -Dtpu.library.path or
 * the PJRT_PATH / TPU_LIBRARY_PATH environment variables. On CI without TPU
 * hardware, {@code pip install libtpu} provides a loadable libtpu.so — TPU VMs
 * are x86_64 hosts, so the same artifact loads on hosted runners.
 *
 * All nd4j-tpu classes are accessed reflectively so this class compiles without
 * the nd4j-tpu dependency on the classpath.
 */
@Slf4j
@Tag(TagNames.TPU)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("TPU backend wiring smoke tests (PJRT/libtpu)")
public class TpuBackendSmokeTest {

    private static final String TPU_BACKEND_CLASS = "org.nd4j.linalg.jtpu.JTpuBackend";
    private static final String TPU_HELPER_CLASS = "org.nd4j.presets.tpu.Nd4jTpuHelper";
    private static final String BACKEND_SERVICE = "META-INF/services/org.nd4j.linalg.factory.Nd4jBackend";

    @Test
    @DisplayName("nd4j-tpu is on the classpath and SPI-registered")
    public void tpuBackendOnClasspathAndRegistered() throws Exception {
        Class<?> backendClass = assertDoesNotThrow(() -> Class.forName(TPU_BACKEND_CLASS),
                "nd4j-tpu is not on the classpath — run from platform-tests with -Ptest-tpu "
                        + "(and install it first: mvn install -DskipTests -Ptpu -pl :nd4j-tpu,:nd4j-tpu-preset)");

        boolean registered = false;
        Enumeration<URL> serviceFiles = Thread.currentThread().getContextClassLoader().getResources(BACKEND_SERVICE);
        while (serviceFiles.hasMoreElements() && !registered) {
            URL url = serviceFiles.nextElement();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(url.openStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().equals(TPU_BACKEND_CLASS)) {
                        registered = true;
                        log.info("JTpuBackend SPI registration found in {}", url);
                        break;
                    }
                }
            }
        }
        assertTrue(registered, "JTpuBackend is on the classpath but not registered in " + BACKEND_SERVICE
                + " — Nd4jBackend discovery can never select it");

        // Constructing and probing the backend must not throw even without TPU hardware
        Object backend = backendClass.getDeclaredConstructor().newInstance();
        Object canRun = backendClass.getMethod("canRun").invoke(backend);
        Object priority = backendClass.getMethod("getPriority").invoke(backend);
        log.info("JTpuBackend.canRun() = {} (stub returns false until PJRT bindings land), priority = {}",
                canRun, priority);
        assertEquals(Boolean.FALSE, canRun,
                "JTpuBackend.canRun() no longer returns the stub value — PJRT bindings appear to have landed. "
                        + "Extend this smoke test with real device enumeration and execution checks.");
    }

    @Test
    @DisplayName("PJRT/libtpu native library loads in-process")
    public void pjrtOrLibtpuLibraryLoads() {
        String resolved = resolvePjrtLibrary();
        assumeTrue(resolved != null,
                "No PJRT/libtpu library found — set -Dpjrt.path / -Dtpu.library.path or the PJRT_PATH / "
                        + "TPU_LIBRARY_PATH env vars to a libtpu.so or PJRT plugin .so. Skipping.");

        final String libToLoad = resolved;
        assertDoesNotThrow(() -> System.load(libToLoad), "Failed to dlopen " + libToLoad);
        log.info("Successfully loaded PJRT/libtpu library in-process: {}", libToLoad);
    }

    @Test
    @DisplayName("Nd4jTpuHelper availability contract matches PJRT_PATH env")
    public void tpuHelperAvailabilityContractMatchesEnv() throws Exception {
        Class<?> helper;
        try {
            helper = Class.forName(TPU_HELPER_CLASS);
        } catch (ClassNotFoundException e) {
            assumeTrue(false, "nd4j-tpu-preset not on the classpath — skipping helper contract check");
            return;
        }

        boolean available = (Boolean) helper.getMethod("isTpuAvailable").invoke(null);
        String pjrtEnv = System.getenv("PJRT_PATH");
        if (pjrtEnv != null && pjrtEnv.contains("${")) {
            // Unresolved Maven placeholder leaked by surefire environmentVariables when the
            // PJRT_PATH env var is unset — the helper must treat it as unavailable.
            log.warn("PJRT_PATH contains an unresolved Maven placeholder ({}) — treated as unset. "
                    + "Pass a real path via -Dpjrt.path or export PJRT_PATH before invoking maven.", pjrtEnv);
        }
        boolean expected = pjrtEnv != null && !pjrtEnv.isEmpty() && !pjrtEnv.contains("${");
        assertEquals(expected, available,
                "Nd4jTpuHelper.isTpuAvailable() contract drifted from its documented PJRT_PATH env check");

        Method deviceCount = helper.getMethod("getDeviceCount");
        Method tpuVersion = helper.getMethod("getTpuVersion");
        log.info("Nd4jTpuHelper: isTpuAvailable={}, deviceCount={}, tpuVersion={}",
                available, deviceCount.invoke(null), tpuVersion.invoke(null));
    }

    /**
     * Resolve a loadable PJRT/libtpu shared library from system properties and env vars.
     * Accepts a direct .so path or a directory containing one. Values with unresolved
     * Maven placeholders ("${...}") are treated as unset.
     */
    private String resolvePjrtLibrary() {
        List<String> candidates = new ArrayList<>();
        candidates.add(System.getProperty("pjrt.path"));
        candidates.add(System.getProperty("tpu.library.path"));
        candidates.add(System.getenv("PJRT_PATH"));
        candidates.add(System.getenv("TPU_LIBRARY_PATH"));

        for (String candidate : candidates) {
            if (candidate == null || candidate.isEmpty() || candidate.contains("${"))
                continue;
            File f = new File(candidate);
            if (f.isFile()) {
                log.info("PJRT library candidate (direct file): {}", f.getAbsolutePath());
                return f.getAbsolutePath();
            }
            if (f.isDirectory()) {
                for (String known : new String[]{"libtpu.so", "libpjrt_c_api_cpu_dynamic.so", "libpjrt_c_api_cpu_plugin.so"}) {
                    File lib = new File(f, known);
                    if (lib.isFile()) {
                        log.info("PJRT library candidate (in directory {}): {}", f, lib.getAbsolutePath());
                        return lib.getAbsolutePath();
                    }
                }
                File[] anyPjrt = f.listFiles((dir, name) ->
                        name.endsWith(".so") && (name.startsWith("libtpu") || name.contains("pjrt")));
                if (anyPjrt != null && anyPjrt.length > 0) {
                    log.info("PJRT library candidate (pattern match in {}): {}", f, anyPjrt[0].getAbsolutePath());
                    return anyPjrt[0].getAbsolutePath();
                }
                log.warn("Directory {} contains no recognizable PJRT/libtpu library", f);
            } else {
                log.warn("PJRT candidate path does not exist: {}", candidate);
            }
        }
        return null;
    }
}
