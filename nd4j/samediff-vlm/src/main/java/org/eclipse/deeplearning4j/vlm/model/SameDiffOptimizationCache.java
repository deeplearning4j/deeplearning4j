/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.vlm.model;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reusable GraphOptimizer + optimized-SDZ cache for SameDiff models.
 *
 * <p>The source file can be ONNX, SDZ, or any other model artifact that should
 * invalidate the optimized cache when it changes. Optimized models are cached
 * next to the source as {@code <basename>.opt.sdz}.</p>
 */
@Slf4j
public class SameDiffOptimizationCache {

    /**
     * System property to disable SDZ optimization caching entirely.
     */
    public static final String DISABLE_CACHE_PROPERTY = "vlm.model.cache.disable";

    /**
     * System property to control graph optimization after loading.
     * Defaults to ON. Set to "false" to disable GraphOptimizer.
     */
    public static final String OPTIMIZER_ENABLED_PROPERTY = "nd4j.optimizer.enabled";

    private static final int MAX_LOAD_RETRIES = 3;
    private static final long RETRY_DELAY_MS = 500;

    private SameDiffOptimizationCache() {
        // utility class
    }

    /**
     * Check if optimization caches are disabled.
     */
    public static boolean isCacheDisabled() {
        return Boolean.getBoolean(DISABLE_CACHE_PROPERTY);
    }

    /**
     * Check if the optimizer is enabled. Defaults to true unless explicitly disabled.
     */
    public static boolean isOptimizerEnabled() {
        String prop = System.getProperty(OPTIMIZER_ENABLED_PROPERTY);
        return prop == null || !"false".equalsIgnoreCase(prop.trim());
    }

    /**
     * Return the conventional optimized SDZ cache file for a source artifact.
     */
    public static File getOptimizedSdzCacheFile(File sourceFile) {
        String name = sourceFile.getName();
        String baseName;
        int dotIdx = name.lastIndexOf('.');
        if (dotIdx > 0) {
            baseName = name.substring(0, dotIdx);
        } else {
            baseName = name;
        }
        return new File(sourceFile.getParentFile(), baseName + ".opt.sdz");
    }

    /**
     * Load a valid optimized cache if one exists, otherwise delete stale cache files.
     */
    public static SameDiff loadOptimizedIfValid(File sourceFile, File baseCacheFile, boolean cacheDisabled) {
        if (cacheDisabled || !isOptimizerEnabled()) {
            return null;
        }

        File optSdzFile = getOptimizedSdzCacheFile(sourceFile);
        boolean optValid = optSdzFile.exists()
                && optSdzFile.lastModified() >= sourceFile.lastModified()
                && (baseCacheFile == null || !baseCacheFile.exists()
                || optSdzFile.lastModified() >= baseCacheFile.lastModified());
        if (optValid) {
            log.info("Loading cached optimized SDZ model: {} ({} bytes)", optSdzFile.getName(), optSdzFile.length());
            long start = System.currentTimeMillis();
            SameDiff sd = loadSdzWithRetry(optSdzFile);
            if (sd != null) {
                long elapsed = System.currentTimeMillis() - start;
                log.info("Loaded cached optimized SDZ model in {}ms: {}", elapsed, optSdzFile.getName());
                return sd;
            }
            log.warn("Failed to load optimized SDZ after retries, will regenerate: {}", optSdzFile.getName());
            optSdzFile.delete();
        } else if (optSdzFile.exists()) {
            log.info("Stale .opt.sdz detected, will re-optimize: {}", optSdzFile.getName());
            optSdzFile.delete();
        }
        return null;
    }

    /**
     * Run GraphOptimizer and optionally cache the optimized graph as {@code <source>.opt.sdz}.
     */
    public static SameDiff optimizeWithCache(SameDiff sd, File sourceFile, boolean cacheDisabled) {
        return optimizeWithCache(sd, sourceFile, cacheDisabled, Map.of());
    }

    /**
     * Run GraphOptimizer and optionally cache the optimized graph with metadata.
     */
    public static SameDiff optimizeWithCache(SameDiff sd, File sourceFile, boolean cacheDisabled,
                                             Map<String, String> extraMetadata) {
        if (!isOptimizerEnabled()) {
            return sd;
        }

        int opsBefore = sd.getOps().size();
        log.info("Running GraphOptimizer on {} ({} ops)...", sourceFile.getName(), opsBefore);
        long optStart = System.currentTimeMillis();

        List<String> outputs = sd.outputs() != null ? new ArrayList<>(sd.outputs()) : new ArrayList<>();
        SameDiff optimized = GraphOptimizer.optimize(sd, outputs);

        int opsAfter = optimized.getOps().size();
        long optElapsed = System.currentTimeMillis() - optStart;
        log.info("GraphOptimizer: {} -> {} ops ({} removed) in {}ms for {}",
                opsBefore, opsAfter, opsBefore - opsAfter, optElapsed, sourceFile.getName());

        if (!cacheDisabled) {
            File optSdzFile = getOptimizedSdzCacheFile(sourceFile);
            try {
                long saveStart = System.currentTimeMillis();
                Map<String, String> metadata = new HashMap<>();
                metadata.put("source_file", sourceFile.getName());
                metadata.put("optimized", "true");
                metadata.put("ops_before", String.valueOf(opsBefore));
                metadata.put("ops_after", String.valueOf(opsAfter));
                metadata.putAll(extraMetadata);
                SDZSerializer.save(optimized, optSdzFile, false, metadata);
                long saveElapsed = System.currentTimeMillis() - saveStart;
                log.info("Cached optimized SDZ in {}ms: {} ({} bytes)",
                        saveElapsed, optSdzFile.getName(), optSdzFile.length());
            } catch (Exception e) {
                log.warn("Failed to cache optimized SDZ (non-fatal): {}", e.getMessage());
                if (optSdzFile.exists()) {
                    optSdzFile.delete();
                }
            }
        }

        return optimized;
    }

    private static SameDiff loadSdzWithRetry(File sdzFile) {
        for (int attempt = 1; attempt <= MAX_LOAD_RETRIES; attempt++) {
            try {
                return SDZSerializer.load(sdzFile, false);
            } catch (Exception e) {
                log.warn("SDZ load attempt {}/{} failed for {}: {}",
                        attempt, MAX_LOAD_RETRIES, sdzFile.getName(), e.getMessage());
                if (attempt < MAX_LOAD_RETRIES) {
                    try {
                        Thread.sleep(RETRY_DELAY_MS * attempt);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        return null;
                    }
                }
            }
        }
        return null;
    }
}
