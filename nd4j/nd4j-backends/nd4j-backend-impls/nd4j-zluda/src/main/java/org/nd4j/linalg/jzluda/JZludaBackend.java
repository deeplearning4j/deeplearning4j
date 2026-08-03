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

package org.nd4j.linalg.jzluda;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryManager;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * ZLUDA Backend for ND4J
 *
 * This backend enables running CUDA-based ND4J operations on AMD and Intel GPUs
 * through the ZLUDA transpiler. ZLUDA translates CUDA API calls to:
 * - HIP/ROCm for AMD GPUs
 * - Level Zero for Intel GPUs
 *
 * Requirements:
 * - ZLUDA_PATH environment variable must point to ZLUDA installation
 * - On Windows, launch the JVM through zluda.exe or place the complete runtime
 *   on PATH/java.library.path (or beside java.exe) so nvcuda.dll is actually resolved
 * - For AMD: ROCm toolkit must be installed
 * - For Intel: oneAPI Level Zero runtime must be installed
 */
public class JZludaBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JZludaBackend.class);
    private static final String LINALG_PROPS = "/nd4j-jzluda.properties";
    private static final String[] WINDOWS_ZLUDA_RUNTIME_FILES = {
            "nvcuda.dll", "nvcudart_hybrid64.dll", "zluda.exe", "zluda_redirect.dll"
    };
    private static final int MAX_DETECTOR_OUTPUT_CHARS = 2 * 1024 * 1024;

    /**
     * ZLUDA target GPU vendors
     */
    public enum ZludaTarget {
        AMD,    // ROCm/HIP backend
        INTEL,  // Level Zero backend
        UNKNOWN
    }

    private ZludaTarget detectedTarget = ZludaTarget.UNKNOWN;
    private boolean zludaAvailable = false;

    public JZludaBackend() {
        // Check ZLUDA availability on construction
        zludaAvailable = checkZludaAvailable();
        if (zludaAvailable) {
            detectedTarget = detectTarget();
            log.info("ZLUDA backend initialized for {} GPUs", detectedTarget);
        }
    }

    @Override
    public boolean isAvailable() {
        return zludaAvailable && detectedTarget != ZludaTarget.UNKNOWN;
    }

    @Override
    public boolean canRun() {
        if (!isAvailable()) {
            return false;
        }

        try {
            // Verify the target GPU is actually available
            switch (detectedTarget) {
                case AMD:
                    return checkAmdGpuAvailable();
                case INTEL:
                    return checkIntelGpuAvailable();
                default:
                    return false;
            }
        } catch (Exception e) {
            log.warn("ZLUDA GPU check failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public int getPriority() {
        // Lower priority than native CUDA (100), higher than CPU (0)
        // This ensures native CUDA is preferred when available
        return BACKEND_PRIORITY_GPU - 10;  // 90
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JZludaBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        // Reuse CUDA NDArray class since ZLUDA is CUDA API-compatible
        try {
            return Class.forName("org.nd4j.linalg.jcublas.JCublasNDArray");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("CUDA NDArray class not found - nd4j-cuda dependency required", e);
        }
    }

    @Override
    public Environment getEnvironment() {
        return ZludaEnvironment.getInstance();
    }

    /**
     * Get the detected ZLUDA target GPU vendor
     */
    public ZludaTarget getTarget() {
        return detectedTarget;
    }

    /**
     * Check if ZLUDA runtime is available
     */
    private boolean checkZludaAvailable() {
        String zludaPath = System.getenv("ZLUDA_PATH");
        if (zludaPath == null || zludaPath.isEmpty()) {
            log.debug("ZLUDA_PATH environment variable not set");
            return false;
        }

        File zludaDir = new File(zludaPath);
        if (!zludaDir.exists() || !zludaDir.isDirectory()) {
            log.debug("ZLUDA_PATH does not point to valid directory: {}", zludaPath);
            return false;
        }

        File runtime = findZludaRuntime(zludaDir);
        if (runtime == null) {
            log.debug("ZLUDA library not found in: {}", zludaPath);
            return false;
        }
        if (isWindows() && !isWindowsRuntimeActivated(runtime.getParentFile())) {
            log.warn("ZLUDA runtime exists at {}, but it is not active for this JVM. "
                    + "Launch through zluda.exe or put the complete runtime on PATH/java.library.path", zludaPath);
            return false;
        }

        log.info("Found active ZLUDA installation at: {}", zludaPath);
        return true;
    }

    private static File findZludaRuntime(File zludaDir) {
        String[] directoryNames = {"", "bin", "lib", "lib64"};
        for (String directoryName : directoryNames) {
            File directory = directoryName.isEmpty() ? zludaDir : new File(zludaDir, directoryName);
            if (isWindows()) {
                if (hasCompleteWindowsRuntime(directory)) {
                    return new File(directory, "nvcuda.dll");
                }
            } else {
                for (String libraryName : new String[]{"libcuda.so", "libnvcuda.so"}) {
                    File candidate = new File(directory, libraryName);
                    if (candidate.isFile()) {
                        return candidate;
                    }
                }
            }
        }
        return null;
    }

    private static boolean hasCompleteWindowsRuntime(File directory) {
        for (String fileName : WINDOWS_ZLUDA_RUNTIME_FILES) {
            if (!new File(directory, fileName).isFile()) {
                return false;
            }
        }
        return true;
    }

    private static boolean isWindowsRuntimeActivated(File runtimeDirectory) {
        boolean launchedByZluda = ProcessHandle.current().parent()
                .flatMap(parent -> parent.info().command())
                .map(command -> new File(command).getName().equalsIgnoreCase("zluda.exe"))
                .orElse(false);
        if (launchedByZluda) {
            return true;
        }
        if (pathContainsDirectory(System.getenv("PATH"), runtimeDirectory)
                || pathContainsDirectory(System.getProperty("java.library.path"), runtimeDirectory)) {
            return true;
        }
        String javaHome = System.getProperty("java.home");
        return javaHome != null && hasCompleteWindowsRuntime(new File(javaHome, "bin"));
    }

    private static boolean pathContainsDirectory(String searchPath, File directory) {
        if (searchPath == null || searchPath.isEmpty()) {
            return false;
        }
        for (String entry : searchPath.split(Pattern.quote(File.pathSeparator))) {
            if (!entry.isEmpty() && sameDirectory(new File(entry), directory)) {
                return true;
            }
        }
        return false;
    }

    private static boolean sameDirectory(File left, File right) {
        try {
            return left.getCanonicalFile().equals(right.getCanonicalFile());
        } catch (IOException e) {
            return left.getAbsoluteFile().equals(right.getAbsoluteFile());
        }
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
    }

    /**
     * Detect the target GPU vendor
     */
    private ZludaTarget detectTarget() {
        // First check for explicit target setting
        String target = System.getenv("ZLUDA_TARGET");
        if (target != null) {
            if (target.equalsIgnoreCase("AMD")) {
                return ZludaTarget.AMD;
            } else if (target.equalsIgnoreCase("INTEL")) {
                return ZludaTarget.INTEL;
            }
        }

        // Auto-detect based on available GPUs
        if (checkAmdGpuAvailable()) {
            return ZludaTarget.AMD;
        }
        if (checkIntelGpuAvailable()) {
            return ZludaTarget.INTEL;
        }

        return ZludaTarget.UNKNOWN;
    }

    /**
     * Check if AMD GPU is available via ROCm
     */
    private boolean checkAmdGpuAvailable() {
        for (String command : amdGpuDetectionCommands()) {
            String output = runDetector(Collections.singletonList(command), 10);
            if (output != null
                    && output.toLowerCase(Locale.ROOT).matches("(?s).*gfx[0-9a-f]+.*")) {
                log.debug("AMD GPU detected via {}", command);
                return true;
            }
        }
        return false;
    }

    private static List<String> amdGpuDetectionCommands() {
        List<String> commands = new ArrayList<>();
        String executableSuffix = isWindows() ? ".exe" : "";
        String[] executableNames = {"rocminfo" + executableSuffix, "hipInfo" + executableSuffix};
        String[] roots = {
                System.getenv("ROCM_PATH"),
                System.getenv("ROCM_HOME"),
                System.getenv("HIP_PATH")
        };
        for (String root : roots) {
            if (root != null && !root.isEmpty()) {
                for (String executableName : executableNames) {
                    File executable = new File(new File(root, "bin"), executableName);
                    if (executable.isFile()) {
                        commands.add(executable.getAbsolutePath());
                    }
                }
            }
        }
        Collections.addAll(commands, executableNames);
        return commands;
    }

    /**
     * Check if Intel GPU is available via Level Zero
     */
    private boolean checkIntelGpuAvailable() {
        String output = runDetector(Collections.singletonList("sycl-ls"), 10);
        if (output != null && output.toLowerCase(Locale.ROOT).contains("intel")) {
            log.debug("Intel GPU detected via sycl-ls");
            return true;
        }

        // Check for oneAPI path as fallback
        String oneapiPath = System.getenv("ONEAPI_ROOT");
        if (oneapiPath == null) {
            oneapiPath = "/opt/intel/oneapi";
        }
        return new File(oneapiPath).exists();
    }

    private static String runDetector(List<String> command, long timeoutSeconds) {
        Path outputFile = null;
        Process process = null;
        try {
            outputFile = Files.createTempFile("nd4j-zluda-detector-", ".log");
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            processBuilder.redirectOutput(outputFile.toFile());
            process = processBuilder.start();
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                if (!process.waitFor(5, TimeUnit.SECONDS)) {
                    log.warn("GPU detector did not terminate after forced shutdown: {}", String.join(" ", command));
                }
                return null;
            }
            if (process.exitValue() != 0) {
                return null;
            }
            return readDetectorOutput(outputFile);
        } catch (InterruptedException e) {
            if (process != null) {
                process.destroyForcibly();
            }
            Thread.currentThread().interrupt();
            return null;
        } catch (IOException | RuntimeException e) {
            log.debug("GPU detector failed ({}): {}", String.join(" ", command), e.getMessage());
            return null;
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
            }
            if (outputFile != null) {
                try {
                    Files.deleteIfExists(outputFile);
                } catch (IOException e) {
                    outputFile.toFile().deleteOnExit();
                }
            }
        }
    }

    private static String readDetectorOutput(Path outputFile) throws IOException {
        StringBuilder output = new StringBuilder();
        char[] buffer = new char[4096];
        try (BufferedReader reader = Files.newBufferedReader(outputFile, StandardCharsets.UTF_8)) {
            int count;
            while (output.length() < MAX_DETECTOR_OUTPUT_CHARS
                    && (count = reader.read(buffer, 0,
                    Math.min(buffer.length, MAX_DETECTOR_OUTPUT_CHARS - output.length()))) >= 0) {
                output.append(buffer, 0, count);
            }
        }
        return output.toString();
    }

    @Override
    public String toString() {
        return "ZLUDA Backend [target=" + detectedTarget + ", available=" + zludaAvailable + "]";
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public String buildInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("ZLUDA Backend\n");
        sb.append("Target: ").append(detectedTarget).append("\n");
        sb.append("ZLUDA Path: ").append(System.getenv("ZLUDA_PATH")).append("\n");
        return sb.toString();
    }

    @Override
    public void logBackendInit() {
        String logInitProperty = System.getProperty(ND4JSystemProperties.LOG_INITIALIZATION, "true");
        boolean logInit = Boolean.parseBoolean(logInitProperty);

        if (logInit) {
            try {
                log.info("ZLUDA Backend build information:\n{}", buildInfo());
                log.info("ZLUDA target: {}", detectedTarget);
            } catch (Throwable t) {
                log.debug("Error logging ZLUDA backend versions", t);
            }
        }
    }

    @Override
    public List<DeviceDescriptor> discoverDevices() {
        // ZLUDA exposes devices through CUDA API
        // For now return empty list - device discovery would require native bindings
        return Collections.emptyList();
    }

    @Override
    public OpExecutioner createExecutioner() {
        return Nd4j.getExecutioner();
    }

    @Override
    public MemoryManager createMemoryManager() {
        return Nd4j.getMemoryManager();
    }

    @Override
    public String getBackendId() {
        return "zluda";
    }
}
