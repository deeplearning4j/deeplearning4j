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

package org.eclipse.deeplearning4j.safetensors.architecture;

import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for SafeTensors architecture builders.
 *
 * <p>Maps HuggingFace architecture names (from config.json "architectures" field)
 * to their corresponding graph builders. Auto-detects architecture from config.json.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * Map<String, Object> config = parseConfigJson(configFile);
 * SafeTensorsArchitecture arch = SafeTensorsArchitectureRegistry.detect(config);
 * SameDiff graph = arch.buildGraph(weights, config);
 * }</pre>
 */
@Slf4j
public class SafeTensorsArchitectureRegistry {

    private static final Map<String, SafeTensorsArchitecture> architectures = new ConcurrentHashMap<>();

    static {
        register(new Qwen3VLSafeTensorsBuilder());
        register(new SmolVLM2SafeTensorsBuilder());
    }

    private SafeTensorsArchitectureRegistry() {
    }

    /**
     * Register an architecture builder.
     */
    public static void register(SafeTensorsArchitecture architecture) {
        architectures.put(architecture.getName().toLowerCase(), architecture);
        for (String arch : architecture.getSupportedArchitectures()) {
            architectures.put(arch.toLowerCase(), architecture);
        }
        log.debug("Registered SafeTensors architecture: {} with variants: {}",
                architecture.getName(), architecture.getSupportedArchitectures());
    }

    /**
     * Detect the appropriate architecture from a parsed config.json.
     *
     * @param config parsed config.json map
     * @return the matching architecture builder, or null if none found
     */
    public static SafeTensorsArchitecture detect(Map<String, Object> config) {
        // Check "architectures" field (HuggingFace standard)
        Object archField = config.get("architectures");
        if (archField instanceof List) {
            for (Object arch : (List<?>) archField) {
                String archName = arch.toString().toLowerCase();
                SafeTensorsArchitecture builder = architectures.get(archName);
                if (builder != null && builder.canHandle(config)) {
                    return builder;
                }
            }
        }

        // Check "model_type" field
        Object modelType = config.get("model_type");
        if (modelType != null) {
            SafeTensorsArchitecture builder = architectures.get(modelType.toString().toLowerCase());
            if (builder != null && builder.canHandle(config)) {
                return builder;
            }
        }

        // Try each registered architecture
        for (SafeTensorsArchitecture arch : architectures.values()) {
            if (arch.canHandle(config)) {
                return arch;
            }
        }

        return null;
    }

    /**
     * Get an architecture by name.
     */
    public static SafeTensorsArchitecture get(String name) {
        return architectures.get(name.toLowerCase());
    }

    /**
     * Check if an architecture is registered.
     */
    public static boolean has(String name) {
        return architectures.containsKey(name.toLowerCase());
    }

    /**
     * Get all registered architecture names.
     */
    public static Set<String> getRegisteredNames() {
        return architectures.keySet();
    }
}
