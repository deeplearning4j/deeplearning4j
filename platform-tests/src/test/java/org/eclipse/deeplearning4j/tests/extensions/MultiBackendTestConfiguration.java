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

package org.eclipse.deeplearning4j.tests.extensions;

import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Configuration class for multi-backend testing support.
 *
 * This class provides:
 * - Detection of available helpers (OneDNN, cuDNN, ARM Compute, etc.)
 * - Backend capability querying
 * - Helper priority configuration
 * - Dynamic kernel selection configuration for tests
 *
 * System Properties:
 * - sd.test.helpers: Comma-separated list of helpers to test (e.g., "onednn,cudnn")
 * - sd.test.helper.priority: Helper priority order for testing
 * - sd.test.kernel.strategy: Kernel selection strategy (fastest, first, roundrobin)
 * - sd.test.backend: Specific backend to use (cpu, cuda)
 * - sd.test.skip.helpers: Helpers to skip during testing
 *
 * Environment Variables:
 * - SD_TEST_HELPERS: Same as sd.test.helpers
 * - SD_TEST_BACKEND: Same as sd.test.backend
 * - SD_KERNEL_STRATEGY: Kernel selection strategy
 * - SD_HELPER_PRIORITY: Helper priority order
 *
 * @author Eclipse Deeplearning4j Development Team
 */
public class MultiBackendTestConfiguration {

    private static final Logger log = LoggerFactory.getLogger(MultiBackendTestConfiguration.class);

    // System property keys
    public static final String PROP_TEST_HELPERS = "sd.test.helpers";
    public static final String PROP_HELPER_PRIORITY = "sd.test.helper.priority";
    public static final String PROP_KERNEL_STRATEGY = "sd.test.kernel.strategy";
    public static final String PROP_TEST_BACKEND = "sd.test.backend";
    public static final String PROP_SKIP_HELPERS = "sd.test.skip.helpers";
    public static final String PROP_ENABLE_ALL_HELPERS = "sd.test.enable.all.helpers";

    // Environment variable keys
    public static final String ENV_TEST_HELPERS = "SD_TEST_HELPERS";
    public static final String ENV_TEST_BACKEND = "SD_TEST_BACKEND";
    public static final String ENV_KERNEL_STRATEGY = "SD_KERNEL_STRATEGY";
    public static final String ENV_HELPER_PRIORITY = "SD_HELPER_PRIORITY";

    // Singleton instance
    private static volatile MultiBackendTestConfiguration instance;

    // Detected helpers and capabilities
    private final Set<Helper> availableHelpers;
    private final Set<Helper> enabledHelpers;
    private final List<Helper> helperPriority;
    private final Backend currentBackend;
    private final KernelStrategy kernelStrategy;
    private final Map<Helper, HelperCapabilities> helperCapabilities;

    /**
     * Available helper implementations
     */
    public enum Helper {
        CPU("cpu", "Native CPU implementation"),
        ONEDNN("onednn", "Intel OneDNN (MKL-DNN)"),
        CUDNN("cudnn", "NVIDIA cuDNN"),
        ARMCOMPUTE("armcompute", "ARM Compute Library"),
        MPS("mps", "Apple Metal Performance Shaders"),
        ACCELERATE("accelerate", "Apple Accelerate Framework"),
        MLIR("mlir", "MLIR/LLVM JIT Compilation"),
        MIOPEN("miopen", "AMD MIOpen"),
        PJRT("pjrt", "PJRT/TPU");

        private final String id;
        private final String description;

        Helper(String id, String description) {
            this.id = id;
            this.description = description;
        }

        public String getId() { return id; }
        public String getDescription() { return description; }

        public static Helper fromString(String str) {
            if (str == null) return null;
            String normalized = str.toLowerCase().trim();
            // Handle aliases
            if (normalized.equals("mkldnn") || normalized.equals("dnnl")) {
                normalized = "onednn";
            }
            for (Helper h : values()) {
                if (h.id.equals(normalized)) {
                    return h;
                }
            }
            return null;
        }
    }

    /**
     * Backend types (CPU vs GPU)
     */
    public enum Backend {
        CPU("cpu"),
        CUDA("cuda"),
        ROCM("rocm"),
        TPU("tpu");

        private final String id;

        Backend(String id) {
            this.id = id;
        }

        public String getId() { return id; }

        public static Backend fromString(String str) {
            if (str == null) return CPU;
            for (Backend b : values()) {
                if (b.id.equalsIgnoreCase(str.trim())) {
                    return b;
                }
            }
            return CPU;
        }
    }

    /**
     * Kernel selection strategies
     */
    public enum KernelStrategy {
        FASTEST("fastest", "Select the fastest available kernel"),
        FIRST("first", "Select the first available kernel in priority order"),
        ROUNDROBIN("roundrobin", "Rotate between available kernels"),
        MEMORY("memory", "Select the most memory-efficient kernel"),
        BENCHMARK("benchmark", "Benchmark all kernels and select best");

        private final String id;
        private final String description;

        KernelStrategy(String id, String description) {
            this.id = id;
            this.description = description;
        }

        public String getId() { return id; }

        public static KernelStrategy fromString(String str) {
            if (str == null) return FASTEST;
            for (KernelStrategy s : values()) {
                if (s.id.equalsIgnoreCase(str.trim())) {
                    return s;
                }
            }
            return FASTEST;
        }
    }

    /**
     * Helper capabilities for feature detection
     */
    public static class HelperCapabilities {
        private final Helper helper;
        private final boolean available;
        private final boolean supportsFloat16;
        private final boolean supportsBFloat16;
        private final boolean supportsInt8;
        private final boolean supportsConvolution;
        private final boolean supportsRNN;
        private final boolean supportsAttention;
        private final boolean supportsBatchNorm;
        private final String version;

        public HelperCapabilities(Helper helper, boolean available) {
            this(helper, available, false, false, false, true, true, true, true, "unknown");
        }

        public HelperCapabilities(Helper helper, boolean available, boolean supportsFloat16,
                                  boolean supportsBFloat16, boolean supportsInt8,
                                  boolean supportsConvolution, boolean supportsRNN,
                                  boolean supportsAttention, boolean supportsBatchNorm,
                                  String version) {
            this.helper = helper;
            this.available = available;
            this.supportsFloat16 = supportsFloat16;
            this.supportsBFloat16 = supportsBFloat16;
            this.supportsInt8 = supportsInt8;
            this.supportsConvolution = supportsConvolution;
            this.supportsRNN = supportsRNN;
            this.supportsAttention = supportsAttention;
            this.supportsBatchNorm = supportsBatchNorm;
            this.version = version;
        }

        public Helper getHelper() { return helper; }
        public boolean isAvailable() { return available; }
        public boolean supportsFloat16() { return supportsFloat16; }
        public boolean supportsBFloat16() { return supportsBFloat16; }
        public boolean supportsInt8() { return supportsInt8; }
        public boolean supportsConvolution() { return supportsConvolution; }
        public boolean supportsRNN() { return supportsRNN; }
        public boolean supportsAttention() { return supportsAttention; }
        public boolean supportsBatchNorm() { return supportsBatchNorm; }
        public String getVersion() { return version; }

        @Override
        public String toString() {
            return String.format("HelperCapabilities{helper=%s, available=%s, version=%s}",
                    helper, available, version);
        }
    }

    private MultiBackendTestConfiguration() {
        log.info("Initializing MultiBackendTestConfiguration...");

        // Detect current backend
        this.currentBackend = detectCurrentBackend();
        log.info("Detected backend: {}", currentBackend);

        // Detect available helpers
        this.availableHelpers = detectAvailableHelpers();
        log.info("Available helpers: {}", availableHelpers);

        // Get enabled helpers from configuration
        this.enabledHelpers = getConfiguredHelpers();
        log.info("Enabled helpers for testing: {}", enabledHelpers);

        // Get helper priority
        this.helperPriority = getConfiguredPriority();
        log.info("Helper priority: {}", helperPriority);

        // Get kernel strategy
        this.kernelStrategy = getConfiguredKernelStrategy();
        log.info("Kernel strategy: {}", kernelStrategy);

        // Detect capabilities for each helper
        this.helperCapabilities = detectHelperCapabilities();

        printConfiguration();
    }

    /**
     * Get the singleton instance
     */
    public static MultiBackendTestConfiguration getInstance() {
        if (instance == null) {
            synchronized (MultiBackendTestConfiguration.class) {
                if (instance == null) {
                    instance = new MultiBackendTestConfiguration();
                }
            }
        }
        return instance;
    }

    /**
     * Reset the configuration (useful for testing)
     */
    public static void reset() {
        synchronized (MultiBackendTestConfiguration.class) {
            instance = null;
        }
    }

    private Backend detectCurrentBackend() {
        try {
            Environment env = Nd4j.getEnvironment();
            if (env.isCPU()) {
                return Backend.CPU;
            } else {
                // Check for CUDA
                String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
                if (backendName.contains("cuda") || backendName.contains("gpu")) {
                    return Backend.CUDA;
                }
            }
        } catch (Exception e) {
            log.warn("Could not detect backend, defaulting to CPU: {}", e.getMessage());
        }
        return Backend.CPU;
    }

    private Set<Helper> detectAvailableHelpers() {
        Set<Helper> helpers = EnumSet.of(Helper.CPU); // CPU is always available

        try {
            // Check environment properties set by libnd4j
            String enabledHelpers = System.getProperty("sd.enabled.helpers",
                    System.getenv().getOrDefault("SD_ENABLED_HELPERS", ""));

            if (!enabledHelpers.isEmpty()) {
                for (String h : enabledHelpers.split(",")) {
                    Helper helper = Helper.fromString(h.trim());
                    if (helper != null) {
                        helpers.add(helper);
                    }
                }
            }

            // Check for specific helper availability via compile-time flags
            if (checkHelperAvailable("HAVE_ONEDNN")) {
                helpers.add(Helper.ONEDNN);
            }
            if (checkHelperAvailable("HAVE_CUDNN")) {
                helpers.add(Helper.CUDNN);
            }
            if (checkHelperAvailable("HAVE_ARMCOMPUTE")) {
                helpers.add(Helper.ARMCOMPUTE);
            }
            if (checkHelperAvailable("HAVE_MPS")) {
                helpers.add(Helper.MPS);
            }
            if (checkHelperAvailable("HAVE_ACCELERATE")) {
                helpers.add(Helper.ACCELERATE);
            }
            if (checkHelperAvailable("HAVE_MLIR")) {
                helpers.add(Helper.MLIR);
            }
            if (checkHelperAvailable("HAVE_MIOPEN")) {
                helpers.add(Helper.MIOPEN);
            }

            // Platform-based detection
            String os = System.getProperty("os.name", "").toLowerCase();
            String arch = System.getProperty("os.arch", "").toLowerCase();

            // OneDNN is typically available on x86/x64
            if (arch.contains("amd64") || arch.contains("x86_64") || arch.contains("x86")) {
                // Check if OneDNN lib exists
                if (isLibraryAvailable("dnnl") || isLibraryAvailable("mkldnn")) {
                    helpers.add(Helper.ONEDNN);
                }
            }

            // ARM Compute on ARM64
            if (arch.contains("aarch64") || arch.contains("arm64")) {
                if (isLibraryAvailable("arm_compute")) {
                    helpers.add(Helper.ARMCOMPUTE);
                }
            }

            // MPS/Accelerate on macOS
            if (os.contains("mac")) {
                helpers.add(Helper.ACCELERATE);
                // MPS requires macOS 10.13+ and Metal-capable GPU
                if (isMPSAvailable()) {
                    helpers.add(Helper.MPS);
                }
            }

        } catch (Exception e) {
            log.warn("Error detecting available helpers: {}", e.getMessage());
        }

        return helpers;
    }

    private boolean checkHelperAvailable(String propertyName) {
        String value = System.getProperty(propertyName, System.getenv(propertyName));
        return "1".equals(value) || "true".equalsIgnoreCase(value) || "ON".equalsIgnoreCase(value);
    }

    private boolean isLibraryAvailable(String libName) {
        try {
            System.loadLibrary(libName);
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    private boolean isMPSAvailable() {
        // MPS availability check would require native code
        // For now, assume available on macOS ARM64
        String arch = System.getProperty("os.arch", "").toLowerCase();
        return arch.contains("aarch64") || arch.contains("arm64");
    }

    private Set<Helper> getConfiguredHelpers() {
        // Check if all helpers should be enabled
        if (Boolean.parseBoolean(System.getProperty(PROP_ENABLE_ALL_HELPERS, "false"))) {
            return new HashSet<>(availableHelpers);
        }

        // Get from system property or environment variable
        String helpersStr = System.getProperty(PROP_TEST_HELPERS,
                System.getenv().getOrDefault(ENV_TEST_HELPERS, ""));

        if (helpersStr.isEmpty()) {
            // Default to all available helpers
            return new HashSet<>(availableHelpers);
        }

        Set<Helper> enabled = EnumSet.noneOf(Helper.class);
        for (String h : helpersStr.split(",")) {
            Helper helper = Helper.fromString(h.trim());
            if (helper != null && availableHelpers.contains(helper)) {
                enabled.add(helper);
            }
        }

        // Remove skipped helpers
        String skipStr = System.getProperty(PROP_SKIP_HELPERS, "");
        if (!skipStr.isEmpty()) {
            for (String h : skipStr.split(",")) {
                Helper helper = Helper.fromString(h.trim());
                if (helper != null) {
                    enabled.remove(helper);
                }
            }
        }

        // Always include CPU as fallback
        enabled.add(Helper.CPU);

        return enabled;
    }

    private List<Helper> getConfiguredPriority() {
        String priorityStr = System.getProperty(PROP_HELPER_PRIORITY,
                System.getenv().getOrDefault(ENV_HELPER_PRIORITY, ""));

        List<Helper> priority = new ArrayList<>();

        if (!priorityStr.isEmpty()) {
            for (String h : priorityStr.split(",")) {
                Helper helper = Helper.fromString(h.trim());
                if (helper != null && enabledHelpers.contains(helper)) {
                    priority.add(helper);
                }
            }
        }

        // Add any enabled helpers not in the priority list
        for (Helper h : enabledHelpers) {
            if (!priority.contains(h)) {
                priority.add(h);
            }
        }

        // Ensure CPU is last (fallback)
        priority.remove(Helper.CPU);
        priority.add(Helper.CPU);

        return priority;
    }

    private KernelStrategy getConfiguredKernelStrategy() {
        String strategyStr = System.getProperty(PROP_KERNEL_STRATEGY,
                System.getenv().getOrDefault(ENV_KERNEL_STRATEGY, "fastest"));
        return KernelStrategy.fromString(strategyStr);
    }

    private Map<Helper, HelperCapabilities> detectHelperCapabilities() {
        Map<Helper, HelperCapabilities> caps = new EnumMap<>(Helper.class);

        for (Helper helper : availableHelpers) {
            caps.put(helper, detectCapabilitiesFor(helper));
        }

        return caps;
    }

    private HelperCapabilities detectCapabilitiesFor(Helper helper) {
        // Default capabilities - in production these would be detected from native code
        switch (helper) {
            case ONEDNN:
                return new HelperCapabilities(helper, true, true, true, true,
                        true, true, true, true, "3.x");
            case CUDNN:
                return new HelperCapabilities(helper, true, true, true, true,
                        true, true, true, true, "8.x");
            case ARMCOMPUTE:
                return new HelperCapabilities(helper, true, true, false, true,
                        true, true, false, true, "25.x");
            case MPS:
                return new HelperCapabilities(helper, true, true, false, false,
                        true, true, true, true, "macOS");
            case ACCELERATE:
                return new HelperCapabilities(helper, true, false, false, false,
                        false, false, false, false, "macOS");
            case MLIR:
                return new HelperCapabilities(helper, true, true, true, true,
                        true, true, true, true, "18.x");
            default:
                return new HelperCapabilities(helper, true);
        }
    }

    private void printConfiguration() {
        log.info("========================================");
        log.info("Multi-Backend Test Configuration");
        log.info("========================================");
        log.info("Current Backend: {}", currentBackend);
        log.info("Kernel Strategy: {}", kernelStrategy);
        log.info("Available Helpers: {}", availableHelpers);
        log.info("Enabled Helpers: {}", enabledHelpers);
        log.info("Helper Priority: {}", helperPriority);
        log.info("========================================");
    }

    // Public API methods

    public Backend getCurrentBackend() {
        return currentBackend;
    }

    public Set<Helper> getAvailableHelpers() {
        return Collections.unmodifiableSet(availableHelpers);
    }

    public Set<Helper> getEnabledHelpers() {
        return Collections.unmodifiableSet(enabledHelpers);
    }

    public List<Helper> getHelperPriority() {
        return Collections.unmodifiableList(helperPriority);
    }

    public KernelStrategy getKernelStrategy() {
        return kernelStrategy;
    }

    public boolean isHelperAvailable(Helper helper) {
        return availableHelpers.contains(helper);
    }

    public boolean isHelperEnabled(Helper helper) {
        return enabledHelpers.contains(helper);
    }

    public HelperCapabilities getHelperCapabilities(Helper helper) {
        return helperCapabilities.get(helper);
    }

    public boolean isCPU() {
        return currentBackend == Backend.CPU;
    }

    public boolean isCUDA() {
        return currentBackend == Backend.CUDA;
    }

    /**
     * Get helpers that support a specific feature
     */
    public Set<Helper> getHelpersWithCapability(String capability) {
        return enabledHelpers.stream()
                .filter(h -> {
                    HelperCapabilities caps = helperCapabilities.get(h);
                    if (caps == null) return false;
                    switch (capability.toLowerCase()) {
                        case "float16": return caps.supportsFloat16();
                        case "bfloat16": return caps.supportsBFloat16();
                        case "int8": return caps.supportsInt8();
                        case "convolution": return caps.supportsConvolution();
                        case "rnn": return caps.supportsRNN();
                        case "attention": return caps.supportsAttention();
                        case "batchnorm": return caps.supportsBatchNorm();
                        default: return true;
                    }
                })
                .collect(Collectors.toSet());
    }

    /**
     * Get test configurations for parameterized tests
     * Returns configurations for each enabled helper
     */
    public List<HelperTestConfig> getTestConfigurations() {
        List<HelperTestConfig> configs = new ArrayList<>();
        for (Helper helper : helperPriority) {
            if (enabledHelpers.contains(helper)) {
                configs.add(new HelperTestConfig(helper, helperCapabilities.get(helper)));
            }
        }
        return configs;
    }

    /**
     * Configuration object for parameterized tests
     */
    public static class HelperTestConfig {
        private final Helper helper;
        private final HelperCapabilities capabilities;

        public HelperTestConfig(Helper helper, HelperCapabilities capabilities) {
            this.helper = helper;
            this.capabilities = capabilities;
        }

        public Helper getHelper() { return helper; }
        public HelperCapabilities getCapabilities() { return capabilities; }

        @Override
        public String toString() {
            return helper.getId();
        }
    }
}
