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

package org.nd4j.linalg.api.ops.executioner;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * KernelManager provides a unified interface for discovering, enabling, disabling,
 * and configuring kernel (backend) implementations for operations.
 *
 * This class allows fine-grained control over which backend implementations
 * (CPU, CUDA, oneDNN, MPS, etc.) are used for each operation.
 *
 * <h3>Example Usage:</h3>
 * <pre>{@code
 * // Get the kernel manager instance
 * KernelManager km = KernelManager.getInstance();
 *
 * // List all available kernels for conv2d
 * OpKernelInfo info = km.getOpKernelInfo("conv2d");
 * for (KernelInfo kernel : info.getKernels()) {
 *     System.out.println(kernel.getEngineName() + ": enabled=" + kernel.isEnabled());
 * }
 *
 * // Disable CUDA for a specific operation
 * km.disableKernel("conv2d", Engine.CUDA);
 *
 * // Set preferred engine for an operation
 * km.setPreferredEngine("matmul", Engine.ONEDNN);
 *
 * // Globally disable an engine
 * km.disableEngineGlobally(Engine.CUDA);
 *
 * // Search for operations
 * List<OpKernelInfo> convOps = km.searchOperations("conv*");
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Slf4j
public class KernelManager {

    private static volatile KernelManager INSTANCE;

    // Per-operation disabled engines
    private final Map<String, Set<Engine>> disabledKernels = new ConcurrentHashMap<>();

    // Per-operation preferred engine
    private final Map<String, Engine> preferredEngines = new ConcurrentHashMap<>();

    // Globally disabled engines
    private final Set<Engine> globallyDisabledEngines = ConcurrentHashMap.newKeySet();

    // Global preferred engine
    @Getter
    private volatile Engine globalPreferredEngine = Engine.CPU;

    // Registered operations (discovered from native layer)
    private final Map<String, OpKernelInfo> registeredOps = new ConcurrentHashMap<>();

    // Native binding available flag
    private volatile boolean nativeBindingAvailable = false;

    /**
     * Supported backend engines
     */
    public enum Engine {
        CPU(0, "CPU", "Native CPU implementation"),
        CUDA(1, "CUDA", "NVIDIA CUDA GPU"),
        ONEDNN(2, "oneDNN", "Intel oneAPI Deep Neural Network Library"),
        CUDNN(3, "cuDNN", "NVIDIA cuDNN"),
        MPS(4, "MPS", "Apple Metal Performance Shaders"),
        ACCELERATE(5, "Accelerate", "Apple Accelerate Framework"),
        ARM_COMPUTE(6, "ARM Compute", "ARM Compute Library"),
        OPENCL(7, "OpenCL", "OpenCL GPU"),
        VULKAN(8, "Vulkan", "Vulkan Compute"),
        MLIR(9, "MLIR", "MLIR Compiler"),
        PJRT(10, "PJRT", "PyTorch JAX Runtime"),
        LLAMA_CPP(11, "LlamaCpp", "Llama.cpp"),
        VLM(12, "VLM", "Vision-Language Models");

        @Getter
        private final int id;
        @Getter
        private final String displayName;
        @Getter
        private final String description;

        Engine(int id, String displayName, String description) {
            this.id = id;
            this.displayName = displayName;
            this.description = description;
        }

        public static Engine fromId(int id) {
            for (Engine e : values()) {
                if (e.id == id) return e;
            }
            return CPU;
        }

        public static Engine fromName(String name) {
            if (name == null) return CPU;
            String lower = name.toLowerCase().trim();
            for (Engine e : values()) {
                if (e.name().toLowerCase().equals(lower) ||
                    e.displayName.toLowerCase().equals(lower)) {
                    return e;
                }
            }
            return CPU;
        }
    }

    /**
     * Information about a specific kernel implementation
     */
    @Data
    public static class KernelInfo {
        private String opName;
        private Engine engine;
        private String engineName;
        private boolean enabled = true;
        private boolean usable = true;
        private double avgTimeNanos = 0;
        private long executionCount = 0;
        private String description;

        public KernelInfo() {}

        public KernelInfo(String opName, Engine engine) {
            this.opName = opName;
            this.engine = engine;
            this.engineName = engine.getDisplayName();
        }
    }

    /**
     * Information about an operation and all its available kernels
     */
    @Data
    public static class OpKernelInfo {
        private String opName;
        private List<KernelInfo> kernels = new ArrayList<>();
        private Engine preferredEngine = Engine.CPU;
        private boolean hasNativeImplementation = true;

        public OpKernelInfo() {}

        public OpKernelInfo(String opName) {
            this.opName = opName;
        }

        public void addKernel(KernelInfo kernel) {
            kernels.add(kernel);
        }

        public Optional<KernelInfo> getKernel(Engine engine) {
            return kernels.stream()
                .filter(k -> k.getEngine() == engine)
                .findFirst();
        }

        public List<Engine> getAvailableEngines() {
            return kernels.stream()
                .map(KernelInfo::getEngine)
                .collect(Collectors.toList());
        }

        public List<Engine> getEnabledEngines() {
            return kernels.stream()
                .filter(KernelInfo::isEnabled)
                .map(KernelInfo::getEngine)
                .collect(Collectors.toList());
        }
    }

    private KernelManager() {
        // Try to initialize native bindings
        initializeNativeBindings();
    }

    /**
     * Get the singleton instance
     */
    public static KernelManager getInstance() {
        if (INSTANCE == null) {
            synchronized (KernelManager.class) {
                if (INSTANCE == null) {
                    INSTANCE = new KernelManager();
                }
            }
        }
        return INSTANCE;
    }

    /**
     * Initialize native bindings if available
     */
    private void initializeNativeBindings() {
        try {
            // Try to discover operations from native layer
            // This will be populated when native library is loaded
            nativeBindingAvailable = true;
            log.debug("KernelManager: Native bindings available");
        } catch (Exception e) {
            log.debug("KernelManager: Native bindings not available, using Java-only mode");
            nativeBindingAvailable = false;
        }
    }

    // ========================
    // Discovery
    // ========================

    /**
     * Get all registered operations with their kernel info
     */
    public List<OpKernelInfo> getAllOperations() {
        return new ArrayList<>(registeredOps.values());
    }

    /**
     * Get kernel info for a specific operation
     */
    public OpKernelInfo getOpKernelInfo(@NonNull String opName) {
        OpKernelInfo info = registeredOps.get(opName.toLowerCase());
        if (info == null) {
            info = new OpKernelInfo(opName);
            info.setHasNativeImplementation(true);
        }

        // Update enabled status based on current configuration
        for (KernelInfo kernel : info.getKernels()) {
            kernel.setEnabled(isKernelEnabled(opName, kernel.getEngine()));
        }

        // Update preferred engine
        info.setPreferredEngine(getPreferredEngine(opName));

        return info;
    }

    /**
     * Get all available engines across all operations
     */
    public Set<Engine> getAllAvailableEngines() {
        Set<Engine> engines = EnumSet.noneOf(Engine.class);
        for (OpKernelInfo op : registeredOps.values()) {
            engines.addAll(op.getAvailableEngines());
        }
        // Always include CPU as fallback
        engines.add(Engine.CPU);
        return engines;
    }

    /**
     * Get available engines for a specific operation
     */
    public List<Engine> getAvailableEngines(@NonNull String opName) {
        OpKernelInfo info = registeredOps.get(opName.toLowerCase());
        if (info != null) {
            return info.getAvailableEngines();
        }
        return Collections.singletonList(Engine.CPU);
    }

    /**
     * Search for operations by name pattern (supports * and ? wildcards)
     */
    public List<OpKernelInfo> searchOperations(@NonNull String pattern) {
        String regex = pattern
            .replace(".", "\\.")
            .replace("*", ".*")
            .replace("?", ".");
        Pattern p = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);

        return registeredOps.values().stream()
            .filter(op -> p.matcher(op.getOpName()).matches())
            .collect(Collectors.toList());
    }

    /**
     * Search operations with a custom predicate
     */
    public List<OpKernelInfo> searchOperations(@NonNull Predicate<OpKernelInfo> predicate) {
        return registeredOps.values().stream()
            .filter(predicate)
            .collect(Collectors.toList());
    }

    // ========================
    // Enable/Disable Kernels
    // ========================

    /**
     * Enable a specific kernel for an operation
     */
    public KernelManager enableKernel(@NonNull String opName, @NonNull Engine engine) {
        String key = opName.toLowerCase();
        Set<Engine> disabled = disabledKernels.get(key);
        if (disabled != null) {
            disabled.remove(engine);
            if (disabled.isEmpty()) {
                disabledKernels.remove(key);
            }
        }
        log.debug("Enabled {} kernel for op: {}", engine.getDisplayName(), opName);
        return this;
    }

    /**
     * Disable a specific kernel for an operation
     */
    public KernelManager disableKernel(@NonNull String opName, @NonNull Engine engine) {
        String key = opName.toLowerCase();
        disabledKernels.computeIfAbsent(key, k -> ConcurrentHashMap.newKeySet()).add(engine);
        log.debug("Disabled {} kernel for op: {}", engine.getDisplayName(), opName);
        return this;
    }

    /**
     * Check if a kernel is enabled for an operation
     */
    public boolean isKernelEnabled(@NonNull String opName, @NonNull Engine engine) {
        // Check global disable
        if (globallyDisabledEngines.contains(engine)) {
            return false;
        }

        // Check op-specific disable
        Set<Engine> disabled = disabledKernels.get(opName.toLowerCase());
        return disabled == null || !disabled.contains(engine);
    }

    /**
     * Enable all kernels for an operation
     */
    public KernelManager enableAllKernels(@NonNull String opName) {
        disabledKernels.remove(opName.toLowerCase());
        log.debug("Enabled all kernels for op: {}", opName);
        return this;
    }

    /**
     * Disable all kernels for an operation (will use native CPU implementation)
     */
    public KernelManager disableAllKernels(@NonNull String opName) {
        String key = opName.toLowerCase();
        OpKernelInfo info = registeredOps.get(key);
        if (info != null) {
            Set<Engine> toDisable = ConcurrentHashMap.newKeySet();
            toDisable.addAll(info.getAvailableEngines());
            disabledKernels.put(key, toDisable);
        }
        log.debug("Disabled all kernels for op: {}", opName);
        return this;
    }

    /**
     * Globally enable an engine (across all operations)
     */
    public KernelManager enableEngineGlobally(@NonNull Engine engine) {
        globallyDisabledEngines.remove(engine);
        log.info("Globally enabled engine: {}", engine.getDisplayName());
        return this;
    }

    /**
     * Globally disable an engine (across all operations)
     */
    public KernelManager disableEngineGlobally(@NonNull Engine engine) {
        globallyDisabledEngines.add(engine);
        log.info("Globally disabled engine: {}", engine.getDisplayName());
        return this;
    }

    /**
     * Check if an engine is globally disabled
     */
    public boolean isEngineGloballyDisabled(@NonNull Engine engine) {
        return globallyDisabledEngines.contains(engine);
    }

    /**
     * Get list of globally disabled engines
     */
    public Set<Engine> getGloballyDisabledEngines() {
        return new HashSet<>(globallyDisabledEngines);
    }

    // ========================
    // Preferred Engine
    // ========================

    /**
     * Set preferred engine for an operation
     */
    public KernelManager setPreferredEngine(@NonNull String opName, @NonNull Engine engine) {
        preferredEngines.put(opName.toLowerCase(), engine);
        log.debug("Set preferred engine for {}: {}", opName, engine.getDisplayName());
        return this;
    }

    /**
     * Get preferred engine for an operation
     */
    public Engine getPreferredEngine(@NonNull String opName) {
        Engine preferred = preferredEngines.get(opName.toLowerCase());
        return preferred != null ? preferred : globalPreferredEngine;
    }

    /**
     * Clear preferred engine for an operation (use auto-selection)
     */
    public KernelManager clearPreferredEngine(@NonNull String opName) {
        preferredEngines.remove(opName.toLowerCase());
        log.debug("Cleared preferred engine for: {}", opName);
        return this;
    }

    /**
     * Set global preferred engine
     */
    public KernelManager setGlobalPreferredEngine(@NonNull Engine engine) {
        this.globalPreferredEngine = engine;
        log.info("Set global preferred engine: {}", engine.getDisplayName());
        return this;
    }

    // ========================
    // Bulk Operations
    // ========================

    /**
     * Disable an engine for multiple operations
     */
    public KernelManager disableEngineForOps(@NonNull Engine engine, @NonNull String... opNames) {
        for (String opName : opNames) {
            disableKernel(opName, engine);
        }
        return this;
    }

    /**
     * Enable an engine for multiple operations
     */
    public KernelManager enableEngineForOps(@NonNull Engine engine, @NonNull String... opNames) {
        for (String opName : opNames) {
            enableKernel(opName, engine);
        }
        return this;
    }

    /**
     * Set preferred engine for multiple operations
     */
    public KernelManager setPreferredEngineForOps(@NonNull Engine engine, @NonNull String... opNames) {
        for (String opName : opNames) {
            setPreferredEngine(opName, engine);
        }
        return this;
    }

    /**
     * Disable an engine for all operations matching a pattern
     */
    public KernelManager disableEngineForPattern(@NonNull Engine engine, @NonNull String pattern) {
        for (OpKernelInfo op : searchOperations(pattern)) {
            disableKernel(op.getOpName(), engine);
        }
        return this;
    }

    // ========================
    // Configuration
    // ========================

    /**
     * Reset all kernel configurations to defaults
     */
    public KernelManager resetToDefaults() {
        disabledKernels.clear();
        preferredEngines.clear();
        globallyDisabledEngines.clear();
        globalPreferredEngine = Engine.CPU;
        log.info("Reset all kernel configurations to defaults");
        return this;
    }

    /**
     * Get a summary of the current kernel configuration
     */
    public String getConfigurationSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Kernel Manager Configuration ===\n\n");

        sb.append("Global Settings:\n");
        sb.append("  Preferred Engine: ").append(globalPreferredEngine.getDisplayName()).append("\n");

        if (!globallyDisabledEngines.isEmpty()) {
            sb.append("  Globally Disabled: ");
            sb.append(globallyDisabledEngines.stream()
                .map(Engine::getDisplayName)
                .collect(Collectors.joining(", ")));
            sb.append("\n");
        }

        if (!preferredEngines.isEmpty() || !disabledKernels.isEmpty()) {
            sb.append("\nPer-Operation Settings:\n");

            Set<String> allOps = new TreeSet<>();
            allOps.addAll(preferredEngines.keySet());
            allOps.addAll(disabledKernels.keySet());

            for (String opName : allOps) {
                sb.append("  ").append(opName).append(":\n");

                Engine pref = preferredEngines.get(opName);
                if (pref != null) {
                    sb.append("    Preferred: ").append(pref.getDisplayName()).append("\n");
                }

                Set<Engine> disabled = disabledKernels.get(opName);
                if (disabled != null && !disabled.isEmpty()) {
                    sb.append("    Disabled: ");
                    sb.append(disabled.stream()
                        .map(Engine::getDisplayName)
                        .collect(Collectors.joining(", ")));
                    sb.append("\n");
                }
            }
        }

        return sb.toString();
    }

    /**
     * Register an operation with its available kernels (called from native layer)
     */
    public void registerOperation(@NonNull String opName, @NonNull List<Engine> availableEngines) {
        OpKernelInfo info = new OpKernelInfo(opName);
        for (Engine engine : availableEngines) {
            KernelInfo kernel = new KernelInfo(opName, engine);
            info.addKernel(kernel);
        }
        registeredOps.put(opName.toLowerCase(), info);
    }

    /**
     * Update kernel performance statistics
     */
    public void updateKernelStats(@NonNull String opName, @NonNull Engine engine,
                                   double avgTimeNanos, long executionCount) {
        OpKernelInfo opInfo = registeredOps.get(opName.toLowerCase());
        if (opInfo != null) {
            opInfo.getKernel(engine).ifPresent(k -> {
                k.setAvgTimeNanos(avgTimeNanos);
                k.setExecutionCount(executionCount);
            });
        }
    }

    /**
     * Check if native bindings are available
     */
    public boolean isNativeBindingAvailable() {
        return nativeBindingAvailable;
    }

    /**
     * Get number of registered operations
     */
    public int getOperationCount() {
        return registeredOps.size();
    }

    @Override
    public String toString() {
        return String.format("KernelManager[ops=%d, globalPreferred=%s, globallyDisabled=%d]",
            registeredOps.size(),
            globalPreferredEngine.getDisplayName(),
            globallyDisabledEngines.size());
    }
}
