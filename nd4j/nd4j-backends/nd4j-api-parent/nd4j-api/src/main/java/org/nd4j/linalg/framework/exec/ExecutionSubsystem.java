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

package org.nd4j.linalg.framework.exec;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Execution subsystem - provides access to op execution, DSP, and kernel management.
 * 
 * Hierarchy:
 *   Nd4j.framework.execution()
 *       ├── dsp()           - DSP diagnostics and plan tracking
 *       │   └── introspection() - DSP plan introspection (NEW)
 *       ├── opExecutioner() - Op execution control
 *       └── kernels()       - Kernel selection and plugins
 * 
 * @author ND4J Team
 */
@Slf4j
public final class ExecutionSubsystem {
    
    private final DspSubsystem dsp;
    private final OpExecutionerSubsystem opExecutioner;
    private final KernelSubsystem kernels;
    private final ExecutionEnvironmentAccess environment;
    
    public ExecutionSubsystem() {
        this.dsp = new DspSubsystem();
        this.opExecutioner = new OpExecutionerSubsystem();
        this.kernels = new KernelSubsystem();
        this.environment = new ExecutionEnvironmentAccess();
    }
    
    /**
     * DSP (Dynamic Shape Plan) diagnostics and tracking
     */
    public DspSubsystem dsp() {
        return dsp;
    }
    
    /**
     * Op executioner access
     */
    public OpExecutionerSubsystem opExecutioner() {
        return opExecutioner;
    }
    
    /**
     * Kernel selection and plugin management
     */
    public KernelSubsystem kernels() {
        return kernels;
    }
    
    /**
     * Execution environment configuration
     */
    public ExecutionEnvironmentAccess environment() {
        return environment;
    }
    
    // =========================================================================
    // DSP Subsystem
    // =========================================================================

    /**
     * DSP diagnostics subsystem
     */
    public static class DspSubsystem {

        private DspPlanIntrospection introspection;

        public DspSubsystem() {
            // Lazy initialization - don't create NativeOpsHolder during static init
            this.introspection = null;
        }

        /**
         * Get DSP plan introspection instance (lazy initialization)
         */
        public DspPlanIntrospection getIntrospection() {
            if (introspection == null) {
                introspection = new DspPlanIntrospection();
            }
            return introspection;
        }

        /**
         * Initialize DSP diagnostics from system properties
         */
        public void initialize() {
            DspDiagnostics.initialize();
        }
        
        /**
         * Enable DSP diagnostic categories
         * @param categories Bitmask of categories (COMPILE, EXECUTE, MEMORY, etc.)
         */
        public void enableCategories(int categories) {
            DspDiagnostics.enableCategories(categories);
        }
        
        /**
         * Enable all DSP diagnostic categories
         */
        public void enableAll() {
            DspDiagnostics.enableCategories(DspDiagnostics.ALL);
        }
        
        /**
         * Disable DSP diagnostic categories
         */
        public void disableCategories(int categories) {
            DspDiagnostics.disableCategories(categories);
        }
        
        /**
         * Check if a diagnostic category is enabled
         */
        public boolean isEnabled(int category) {
            return DspDiagnostics.isEnabled(category);
        }
        
        /**
         * Set diagnostic detail level
         * @param level LEVEL_SUMMARY (0), LEVEL_DETAILED (1), or LEVEL_FULL (2)
         */
        public void setLevel(int level) {
            DspDiagnostics.setLevel(level);
        }

        /**
         * Set JSON output file path
         */
        public void setJsonOutputPath(String path) {
            // JSON path setting not available in current DspDiagnostics API
            // This is a no-op stub
        }

        /**
         * Record a diagnostic event
         */
        public void record(int category, String message) {
            DspDiagnostics.record(category, message);
        }

        /**
         * Record a slot-specific diagnostic event
         */
        public void recordSlot(int category, int slotId, String opName, String message) {
            DspDiagnostics.recordSlot(category, slotId, opName, message);
        }
        
        /**
         * Get structured text plan report
         */
        public String getPlanReport() {
            return DspDiagnostics.getPlanReport();
        }
        
        /**
         * Get JSON report
         */
        public String getJsonReport() {
            return DspDiagnostics.getJsonReport();
        }
        
        /**
         * Clear all diagnostic events
         */
        public void clear() {
            DspDiagnostics.clear();
        }
        
        /**
         * Get DSP statistics
         */
        public DspStats stats() {
            return DspStats.builder()
                .enabledCategories(getEnabledCategories())
                .detailLevel(DspDiagnostics.LEVEL_SUMMARY)
                .build();
        }
        
        /**
         * Get enabled category names
         */
        public String[] getEnabledCategories() {
            int mask = DspDiagnostics.isEnabled(DspDiagnostics.ALL) ? 
                       DspDiagnostics.ALL : 0;
            return new String[0];
        }
        
        /**
         * Get DSP plan introspection instance for detailed plan inspection
         */
        public DspPlanIntrospection introspection() {
            return introspection;
        }
        
        // Category constants for convenience
        public static final int COMPILE  = DspDiagnostics.COMPILE;
        public static final int JIT      = DspDiagnostics.JIT;
        public static final int EXECUTE  = DspDiagnostics.EXECUTE;
        public static final int TIMING   = DspDiagnostics.TIMING;
        public static final int MEMORY   = DspDiagnostics.MEMORY;
        public static final int BACKEND  = DspDiagnostics.BACKEND;
        public static final int SHAPE    = DspDiagnostics.SHAPE;
        public static final int SEGMENT  = DspDiagnostics.SEGMENT;
        public static final int FUSION   = DspDiagnostics.FUSION;
        public static final int VERIFY   = DspDiagnostics.VERIFY;
        public static final int KV_CACHE = DspDiagnostics.KV_CACHE;
        public static final int FALLBACK = DspDiagnostics.FALLBACK;
        public static final int ALL      = DspDiagnostics.ALL;
        public static final int NONE     = DspDiagnostics.NONE;
        public static final int LEVEL_SUMMARY  = DspDiagnostics.LEVEL_SUMMARY;
        public static final int LEVEL_DETAILED = DspDiagnostics.LEVEL_DETAILED;
        public static final int LEVEL_FULL     = DspDiagnostics.LEVEL_FULL;
    }

    // =========================================================================
    // Op Executioner Subsystem
    // =========================================================================

    /**
     * Op executioner subsystem - provides access to the current op executioner
     */
    public static class OpExecutionerSubsystem {

        /**
         * Get the current op executioner
         */
        public OpExecutioner get() {
            return Nd4j.getExecutioner();
        }

        /**
         * Commit pending operations
         */
        public void commit() {
            Nd4j.getExecutioner().commit();
        }
    }

    // =========================================================================
    // Kernel Subsystem
    // =========================================================================

    /**
     * Kernel selection and plugin management subsystem (placeholder)
     */
    public static class KernelSubsystem {
        // Placeholder - kernel selection APIs to be added
    }

    // =========================================================================
    // Execution Environment Access
    // =========================================================================

    /**
     * Execution environment configuration access
     */
    public static class ExecutionEnvironmentAccess {

        private final ExecutionEnvironment env = new ExecutionEnvironment();

        /**
         * Get the execution environment configuration
         */
        public ExecutionEnvironment get() {
            return env;
        }

        /**
         * Check if DSP is enabled
         */
        public boolean isDspEnabled() {
            return env.isDspEnabled();
        }

        /**
         * Get summary of execution environment settings
         */
        public String getSummary() {
            return env.getSummary();
        }
    }
}