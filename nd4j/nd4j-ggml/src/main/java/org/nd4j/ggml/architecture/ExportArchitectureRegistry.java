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

package org.nd4j.ggml.architecture;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for export architectures.
 * Provides architecture detection and lookup for SameDiff to GGML export.
 */
@Slf4j
public class ExportArchitectureRegistry {

    private static final Map<String, ExportArchitecture> architectures = new ConcurrentHashMap<>();

    static {
        // Register built-in architectures
        register(new LLaMAExportArchitecture());
    }

    private ExportArchitectureRegistry() {
        // Utility class
    }

    /**
     * Register an export architecture
     */
    public static void register(ExportArchitecture architecture) {
        architectures.put(architecture.getArchitectureName().toLowerCase(), architecture);
        log.debug("Registered export architecture: {}", architecture.getArchitectureName());
    }

    /**
     * Get an architecture by name
     *
     * @param name the architecture name (case-insensitive)
     * @return the architecture, or null if not found
     */
    public static ExportArchitecture get(String name) {
        if (name == null) return null;
        return architectures.get(name.toLowerCase());
    }

    /**
     * Check if an architecture is registered
     */
    public static boolean hasArchitecture(String name) {
        return name != null && architectures.containsKey(name.toLowerCase());
    }

    /**
     * Get all registered architecture names
     */
    public static Set<String> getRegisteredArchitectures() {
        return architectures.keySet();
    }

    /**
     * Detect the appropriate architecture for a SameDiff model
     *
     * @param model the SameDiff model
     * @return the detected architecture, or null if none matches
     */
    public static ExportArchitecture detect(SameDiff model) {
        if (model == null) return null;

        for (ExportArchitecture arch : architectures.values()) {
            if (arch.canHandle(model)) {
                log.debug("Detected architecture: {} for model", arch.getArchitectureName());
                return arch;
            }
        }

        log.warn("Could not detect architecture for model");
        return null;
    }

    /**
     * Get the architecture for a model, using hint if provided
     *
     * @param model            the SameDiff model
     * @param architectureHint optional architecture name hint
     * @return the architecture, or null if none matches
     */
    public static ExportArchitecture getOrDetect(SameDiff model, String architectureHint) {
        if (architectureHint != null) {
            ExportArchitecture arch = get(architectureHint);
            if (arch != null) {
                return arch;
            }
            log.warn("Specified architecture '{}' not found, attempting auto-detection", architectureHint);
        }
        return detect(model);
    }
}
