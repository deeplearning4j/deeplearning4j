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
import org.nd4j.ggml.format.GGMLMetadata;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for model architecture handlers.
 * Maps architecture names to their implementations.
 */
@Slf4j
public class ArchitectureRegistry {

    private static final Map<String, ModelArchitecture> architectures = new ConcurrentHashMap<>();

    static {
        // Register built-in architectures
        register(new LLaMAArchitecture());
        register(new GraniteArchitecture());
        register(new MistralArchitecture());
        register(new GemmaArchitecture());
        register(new NemotronArchitecture());
        register(new LFM2Architecture());
        register(new OLMoArchitecture());
        register(new PhiArchitecture());
        register(new OpenELMArchitecture());
        register(new GptOssArchitecture());
        register(new WhisperArchitecture());
        register(new GLMArchitecture());
        register(new Llama4Architecture());
        register(new GenericArchitecture());
    }

    private ArchitectureRegistry() {
        // Utility class
    }

    /**
     * Register an architecture handler.
     * Registers both the primary name and all variant names.
     */
    public static void register(ModelArchitecture architecture) {
        architectures.put(architecture.getName().toLowerCase(), architecture);
        for (String variant : architecture.getSupportedVariants()) {
            architectures.put(variant.toLowerCase(), architecture);
        }
        log.debug("Registered architecture: {} with variants: {}",
                architecture.getName(), architecture.getSupportedVariants());
    }

    /**
     * Get an architecture by name
     */
    public static ModelArchitecture getArchitecture(String name) {
        return architectures.get(name.toLowerCase());
    }

    /**
     * Detect the appropriate architecture from model metadata
     */
    public static ModelArchitecture detectArchitecture(GGMLMetadata metadata) {
        String archName = metadata.getArchitecture();

        if (archName != null) {
            ModelArchitecture arch = architectures.get(archName.toLowerCase());
            if (arch != null && arch.canHandle(metadata)) {
                return arch;
            }
        }

        // Try each registered architecture
        for (ModelArchitecture arch : architectures.values()) {
            if (arch.canHandle(metadata)) {
                return arch;
            }
        }

        // Fall back to generic architecture
        return architectures.get("generic");
    }

    /**
     * Check if an architecture is registered
     */
    public static boolean hasArchitecture(String name) {
        return architectures.containsKey(name.toLowerCase());
    }

    /**
     * Get all registered architecture names
     */
    public static Set<String> getRegisteredArchitectures() {
        return architectures.keySet();
    }
}
