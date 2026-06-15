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

package org.nd4j.torchscript.architecture;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.torchscript.format.TorchScriptMetadata;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for vision model architectures.
 * Provides architecture lookup and auto-detection.
 */
@Slf4j
public class ArchitectureRegistry {

    private static final Map<String, VisionArchitecture> ARCHITECTURES = new ConcurrentHashMap<>();
    private static final List<VisionArchitecture> ORDERED_ARCHITECTURES = new ArrayList<>();

    static {
        // Register built-in architectures
        register(new ResNetArchitecture());
        register(new VGGArchitecture());
        register(new EfficientNetArchitecture());
        register(new GenericCNNArchitecture()); // Fallback, lowest priority
    }

    private ArchitectureRegistry() {
        // Utility class
    }

    /**
     * Register a new architecture.
     *
     * @param architecture the architecture to register
     */
    public static synchronized void register(VisionArchitecture architecture) {
        // Register by primary name
        ARCHITECTURES.put(architecture.getName().toLowerCase(), architecture);

        // Register all variants
        for (String variant : architecture.getSupportedVariants()) {
            ARCHITECTURES.put(variant.toLowerCase(), architecture);
        }

        // Add to ordered list and sort by priority
        if (!ORDERED_ARCHITECTURES.contains(architecture)) {
            ORDERED_ARCHITECTURES.add(architecture);
            ORDERED_ARCHITECTURES.sort((a, b) -> Integer.compare(b.getPriority(), a.getPriority()));
        }

        log.debug("Registered architecture: {} with variants: {}",
                architecture.getName(), architecture.getSupportedVariants());
    }

    /**
     * Get an architecture by name.
     *
     * @param name the architecture name or variant
     * @return the architecture or null if not found
     */
    public static VisionArchitecture getArchitecture(String name) {
        if (name == null) {
            return null;
        }
        return ARCHITECTURES.get(name.toLowerCase());
    }

    /**
     * Detect the architecture for a model based on its metadata.
     *
     * @param metadata the model metadata
     * @return detected architecture or GenericCNNArchitecture as fallback
     */
    public static VisionArchitecture detectArchitecture(TorchScriptMetadata metadata) {
        // First check if architecture is explicitly specified
        String arch = metadata.getArchitecture();
        if (arch != null && !arch.equals("unknown")) {
            VisionArchitecture found = getArchitecture(arch);
            if (found != null) {
                log.debug("Found architecture from metadata: {}", arch);
                return found;
            }
        }

        // Try to detect based on tensor patterns
        for (VisionArchitecture architecture : ORDERED_ARCHITECTURES) {
            if (architecture.canHandle(metadata)) {
                log.debug("Detected architecture: {}", architecture.getName());
                return architecture;
            }
        }

        // Fallback to generic
        log.debug("Using generic CNN architecture as fallback");
        return getArchitecture("generic_cnn");
    }

    /**
     * Check if an architecture is registered.
     *
     * @param name the architecture name
     * @return true if registered
     */
    public static boolean hasArchitecture(String name) {
        return name != null && ARCHITECTURES.containsKey(name.toLowerCase());
    }

    /**
     * Get all registered architecture names.
     *
     * @return set of architecture names
     */
    public static Set<String> getRegisteredArchitectures() {
        Set<String> names = new HashSet<>();
        for (VisionArchitecture arch : ORDERED_ARCHITECTURES) {
            names.add(arch.getName());
        }
        return names;
    }

    /**
     * Get all registered architectures in priority order.
     *
     * @return list of architectures
     */
    public static List<VisionArchitecture> getAllArchitectures() {
        return Collections.unmodifiableList(ORDERED_ARCHITECTURES);
    }
}
