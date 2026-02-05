/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.pipeline;

import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for pipeline loaders.
 */
@Slf4j
public class PipelineLoaderRegistry {

    private static final Map<ModelFormat, PipelineLoader> LOADERS = new ConcurrentHashMap<>();
    private static volatile boolean initialized = false;

    private PipelineLoaderRegistry() {}

    public static synchronized void initialize() {
        if (initialized) return;
        log.debug("Initializing PipelineLoaderRegistry");
        try {
            ServiceLoader<PipelineLoader> sl = ServiceLoader.load(PipelineLoader.class);
            for (PipelineLoader loader : sl) register(loader);
        } catch (Exception e) {
            log.warn("Failed to load pipeline loaders via ServiceLoader: {}", e.getMessage());
        }
        initialized = true;
        log.info("PipelineLoaderRegistry initialized with {} loaders", LOADERS.size());
    }

    public static void register(PipelineLoader loader) {
        if (loader == null) return;
        LOADERS.put(loader.getFormat(), loader);
        log.debug("Registered loader for {}: {}", loader.getFormat(), loader.getName());
    }

    public static PipelineLoader getLoader(ModelFormat format) {
        ensureInitialized();
        return LOADERS.get(format);
    }

    public static PipelineLoader requireLoader(ModelFormat format) {
        PipelineLoader loader = getLoader(format);
        if (loader == null) {
            throw new UnsupportedOperationException("No loader for format: " + format);
        }
        return loader;
    }

    public static boolean hasLoader(ModelFormat format) {
        ensureInitialized();
        return LOADERS.containsKey(format);
    }

    public static Set<ModelFormat> getRegisteredFormats() {
        ensureInitialized();
        return new HashSet<>(LOADERS.keySet());
    }

    public static void clear() {
        LOADERS.clear();
        initialized = false;
    }

    private static void ensureInitialized() {
        if (!initialized) initialize();
    }
}
