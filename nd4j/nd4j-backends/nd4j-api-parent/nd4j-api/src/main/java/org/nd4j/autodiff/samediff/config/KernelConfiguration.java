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

package org.nd4j.autodiff.samediff.config;

import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.executioner.KernelManager;
import org.nd4j.linalg.api.ops.executioner.KernelManager.Engine;
import org.nd4j.linalg.api.ops.executioner.KernelManager.OpKernelInfo;

import java.util.*;

/**
 * KernelConfiguration provides a fluent API for configuring kernel/backend
 * selection for SameDiff models.
 *
 * <h3>Example Usage:</h3>
 * <pre>{@code
 * SameDiff sd = SameDiff.create();
 *
 * // Configure kernels for this model
 * KernelConfiguration config = sd.kernelConfiguration()
 *     .preferCuda()                              // Use CUDA when available
 *     .disableEngine(Engine.ONEDNN)              // Don't use oneDNN
 *     .forConvolutions().useCudnn()              // Use cuDNN for convolutions
 *     .forMatMul().useOneDnn()                   // Use oneDNN for matmul
 *     .apply();
 *
 * // Or use presets
 * sd.kernelConfiguration()
 *     .usePreset(Preset.GPU_OPTIMIZED)
 *     .apply();
 *
 * // Query available kernels
 * List<OpKernelInfo> convKernels = config.findKernels("conv*");
 * }</pre>
 *
 * @author Adam Gibson
 */
public class KernelConfiguration {

    @Getter
    private final SameDiff sameDiff;

    @Getter
    private final KernelManager kernelManager;

    // Pending configuration changes
    private final List<Runnable> pendingChanges = new ArrayList<>();

    /**
     * Preset configurations for common use cases
     */
    public enum Preset {
        /**
         * Default CPU-optimized configuration
         */
        CPU_OPTIMIZED,

        /**
         * GPU-optimized configuration (prefers CUDA/cuDNN)
         */
        GPU_OPTIMIZED,

        /**
         * Intel-optimized configuration (prefers oneDNN)
         */
        INTEL_OPTIMIZED,

        /**
         * Apple Silicon optimized (prefers MPS/Accelerate)
         */
        APPLE_SILICON_OPTIMIZED,

        /**
         * Maximum compatibility (CPU only, no accelerated kernels)
         */
        MAXIMUM_COMPATIBILITY,

        /**
         * Auto-tune to find best kernel for each operation
         */
        AUTO_TUNE
    }

    public KernelConfiguration(@NonNull SameDiff sameDiff) {
        this.sameDiff = sameDiff;
        this.kernelManager = KernelManager.getInstance();
    }

    /**
     * Package-private hook used by {@link CategoryConfiguration},
     * {@link OperationConfiguration}, and {@link PatternConfiguration} to
     * enqueue deferred configuration changes.
     */
    void addPendingChange(Runnable change) {
        pendingChanges.add(change);
    }

    // ========================
    // Global Settings
    // ========================

    /**
     * Prefer CPU for all operations
     */
    public KernelConfiguration preferCpu() {
        pendingChanges.add(() -> kernelManager.setGlobalPreferredEngine(Engine.CPU));
        return this;
    }

    /**
     * Prefer CUDA GPU for all operations
     */
    public KernelConfiguration preferCuda() {
        pendingChanges.add(() -> kernelManager.setGlobalPreferredEngine(Engine.CUDA));
        return this;
    }

    /**
     * Prefer oneDNN for all operations
     */
    public KernelConfiguration preferOneDnn() {
        pendingChanges.add(() -> kernelManager.setGlobalPreferredEngine(Engine.ONEDNN));
        return this;
    }

    /**
     * Prefer MPS (Metal) for all operations
     */
    public KernelConfiguration preferMps() {
        pendingChanges.add(() -> kernelManager.setGlobalPreferredEngine(Engine.MPS));
        return this;
    }

    /**
     * Set the global preferred engine
     */
    public KernelConfiguration preferEngine(@NonNull Engine engine) {
        pendingChanges.add(() -> kernelManager.setGlobalPreferredEngine(engine));
        return this;
    }

    /**
     * Globally disable an engine
     */
    public KernelConfiguration disableEngine(@NonNull Engine engine) {
        pendingChanges.add(() -> kernelManager.disableEngineGlobally(engine));
        return this;
    }

    /**
     * Globally enable an engine
     */
    public KernelConfiguration enableEngine(@NonNull Engine engine) {
        pendingChanges.add(() -> kernelManager.enableEngineGlobally(engine));
        return this;
    }

    /**
     * Disable all accelerated kernels (CPU only)
     */
    public KernelConfiguration cpuOnly() {
        pendingChanges.add(() -> {
            for (Engine e : Engine.values()) {
                if (e != Engine.CPU) {
                    kernelManager.disableEngineGlobally(e);
                }
            }
        });
        return this;
    }

    // ========================
    // Category-based Configuration
    // ========================

    /**
     * Configure convolution operations
     */
    public CategoryConfiguration forConvolutions() {
        return new CategoryConfiguration(this, OperationCategory.CONVOLUTIONS);
    }

    /**
     * Configure pooling operations
     */
    public CategoryConfiguration forPooling() {
        return new CategoryConfiguration(this, OperationCategory.POOLING);
    }

    /**
     * Configure normalization operations
     */
    public CategoryConfiguration forNormalization() {
        return new CategoryConfiguration(this, OperationCategory.NORMALIZATION);
    }

    /**
     * Configure activation operations
     */
    public CategoryConfiguration forActivations() {
        return new CategoryConfiguration(this, OperationCategory.ACTIVATIONS);
    }

    /**
     * Configure linear algebra operations (matmul, gemm, etc.)
     */
    public CategoryConfiguration forLinearAlgebra() {
        return new CategoryConfiguration(this, OperationCategory.LINEAR_ALGEBRA);
    }

    /**
     * Configure element-wise operations
     */
    public CategoryConfiguration forElementWise() {
        return new CategoryConfiguration(this, OperationCategory.ELEMENT_WISE);
    }

    /**
     * Configure reduction operations
     */
    public CategoryConfiguration forReductions() {
        return new CategoryConfiguration(this, OperationCategory.REDUCTION);
    }

    /**
     * Configure attention operations
     */
    public CategoryConfiguration forAttention() {
        return new CategoryConfiguration(this, OperationCategory.ATTENTION);
    }

    /**
     * Configure recurrent operations
     */
    public CategoryConfiguration forRecurrent() {
        return new CategoryConfiguration(this, OperationCategory.RECURRENT);
    }

    /**
     * Configure a specific operation
     */
    public OperationConfiguration forOperation(@NonNull String opName) {
        return new OperationConfiguration(this, opName);
    }

    /**
     * Configure operations matching a pattern
     */
    public PatternConfiguration forPattern(@NonNull String pattern) {
        return new PatternConfiguration(this, pattern);
    }

    // ========================
    // Presets
    // ========================

    /**
     * Apply a preset configuration
     */
    public KernelConfiguration usePreset(@NonNull Preset preset) {
        switch (preset) {
            case CPU_OPTIMIZED:
                applyCpuOptimizedPreset();
                break;
            case GPU_OPTIMIZED:
                applyGpuOptimizedPreset();
                break;
            case INTEL_OPTIMIZED:
                applyIntelOptimizedPreset();
                break;
            case APPLE_SILICON_OPTIMIZED:
                applyAppleSiliconPreset();
                break;
            case MAXIMUM_COMPATIBILITY:
                cpuOnly();
                break;
            case AUTO_TUNE:
                // Auto-tune is handled at runtime
                break;
        }
        return this;
    }

    private void applyCpuOptimizedPreset() {
        preferCpu();
        forLinearAlgebra().useOneDnn();
        forConvolutions().useOneDnn();
    }

    private void applyGpuOptimizedPreset() {
        preferCuda();
        forConvolutions().useCudnn();
        forNormalization().useCudnn();
        forPooling().useCudnn();
        forActivations().useCuda();
    }

    private void applyIntelOptimizedPreset() {
        preferOneDnn();
        forConvolutions().useOneDnn();
        forNormalization().useOneDnn();
        forLinearAlgebra().useOneDnn();
    }

    private void applyAppleSiliconPreset() {
        preferMps();
        pendingChanges.add(() -> kernelManager.setGlobalPreferredEngine(Engine.MPS));
        forLinearAlgebra().useAccelerate();
    }

    // ========================
    // Query
    // ========================

    /**
     * Find kernels matching a pattern
     */
    public List<OpKernelInfo> findKernels(@NonNull String pattern) {
        return kernelManager.searchOperations(pattern);
    }

    /**
     * Get kernel info for a specific operation
     */
    public OpKernelInfo getKernelInfo(@NonNull String opName) {
        return kernelManager.getOpKernelInfo(opName);
    }

    /**
     * Get all available engines
     */
    public Set<Engine> getAvailableEngines() {
        return kernelManager.getAllAvailableEngines();
    }

    /**
     * Get current configuration summary
     */
    public String getSummary() {
        return kernelManager.getConfigurationSummary();
    }

    // ========================
    // Apply
    // ========================

    /**
     * Apply all pending configuration changes
     */
    public KernelConfiguration apply() {
        for (Runnable change : pendingChanges) {
            change.run();
        }
        pendingChanges.clear();
        return this;
    }

    /**
     * Reset to default configuration
     */
    public KernelConfiguration reset() {
        kernelManager.resetToDefaults();
        pendingChanges.clear();
        return this;
    }
}
