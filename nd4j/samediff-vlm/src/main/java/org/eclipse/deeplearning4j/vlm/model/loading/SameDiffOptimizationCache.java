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

package org.eclipse.deeplearning4j.vlm.model.loading;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
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

    /**
     * Key used in the .opt.sdz.meta sidecar file for the optimizer code fingerprint.
     * The fingerprint equals the native buildInfo() string which embeds
     * LIBND4J_BUILD_STAMP (a per-build timestamp injected by GenerateBuildStamp.cmake).
     * Any .so rebuild changes this value and forces re-optimization.
     */
    private static final String META_KEY_FINGERPRINT = "optimizerFingerprint";

    /**
     * Per-JVM cached optimizer fingerprint.  Computed once from buildInfo() on first
     * access.  Using the same native fingerprint as DspPlanDiskCache keeps the two
     * caches consistent: both are invalidated by the same .so rebuild.
     */
    private static volatile String OPTIMIZER_FINGERPRINT = null;

    /** Return the cached optimizer fingerprint, initialising it on first call. */
    static String getOptimizerFingerprint() {
        if (OPTIMIZER_FINGERPRINT == null) {
            synchronized (SameDiffOptimizationCache.class) {
                if (OPTIMIZER_FINGERPRINT == null) {
                    try {
                        String info = NativeOpsHolder.getInstance()
                                .getDeviceNativeOps().buildInfo();
                        OPTIMIZER_FINGERPRINT = (info != null) ? info : "unavailable";
                    } catch (Exception e) {
                        log.warn("SameDiffOptimizationCache: could not read native buildInfo(), " +
                                "optimizer cache fingerprint unavailable: {}", e.getMessage());
                        OPTIMIZER_FINGERPRINT = "unavailable";
                    }
                }
            }
        }
        return OPTIMIZER_FINGERPRINT;
    }

    /**
     * Return the sidecar metadata file for an optimized SDZ cache file.
     * Convention: {@code <opt-sdz-path>.meta}.
     */
    static File getOptimizedSdzMetaFile(File optSdzFile) {
        return new File(optSdzFile.getParentFile(), optSdzFile.getName() + ".meta");
    }

    /**
     * Write the optimizer fingerprint sidecar next to {@code optSdzFile}.
     * Uses atomic temp-file + rename to avoid partial writes.
     */
    private static void writeOptSdzMeta(File optSdzFile) {
        File metaFile = getOptimizedSdzMetaFile(optSdzFile);
        String content = META_KEY_FINGERPRINT + "=" + getOptimizerFingerprint() + "\n";
        long pid = ProcessHandle.current().pid();
        long tid = Thread.currentThread().getId();
        File tmpFile = new File(metaFile.getParentFile(),
                metaFile.getName() + ".tmp." + pid + "." + tid);
        try {
            Files.writeString(tmpFile.toPath(), content);
            Files.move(tmpFile.toPath(), metaFile.toPath(),
                    StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            log.warn("SameDiffOptimizationCache: failed to write opt.sdz meta sidecar {}: {}",
                    metaFile.getName(), e.getMessage());
            tmpFile.delete();
        }
    }

    /**
     * Read the optimizer fingerprint stored in an opt.sdz sidecar.
     *
     * @return the stored fingerprint, or {@code null} if the sidecar is absent or unreadable
     */
    private static String readOptSdzFingerprint(File optSdzFile) {
        File metaFile = getOptimizedSdzMetaFile(optSdzFile);
        if (!metaFile.exists()) return null;
        try (BufferedReader reader = new BufferedReader(new FileReader(metaFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith(META_KEY_FINGERPRINT + "=")) {
                    return line.substring((META_KEY_FINGERPRINT + "=").length()).trim();
                }
            }
        } catch (IOException e) {
            log.debug("SameDiffOptimizationCache: could not read opt.sdz meta sidecar {}: {}",
                    metaFile.getName(), e.getMessage());
        }
        return null;
    }

    /**
     * Is the native-build-fingerprint sidecar for {@code cacheFile} present AND equal to the
     * current build's fingerprint? The base-.sdz import cache (OnnxModelCache) and the .opt.sdz
     * optimizer cache share this ONE check — both keyed on the same buildInfo() fingerprint, so a
     * single .so rebuild invalidates both. Returns false (stale) when the sidecar is absent (a
     * legacy entry written before fingerprinting); true when the stored fingerprint was empty
     * (buildInfo() unavailable at write time — don't reject on that). Reuses the SAME sidecar
     * read + fingerprint as {@link #readOptSdzFingerprint}/{@link #getOptimizerFingerprint} — no
     * second fingerprint mechanism.
     */
    public static boolean isBuildFingerprintCurrent(File cacheFile) {
        String cached = readOptSdzFingerprint(cacheFile);
        if (cached == null) return false;
        if (cached.isEmpty()) return true;
        return getOptimizerFingerprint().equals(cached);
    }

    /** Write the native-build-fingerprint sidecar next to {@code cacheFile} (atomic temp+rename). */
    public static void writeBuildFingerprint(File cacheFile) {
        writeOptSdzMeta(cacheFile);
    }

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
        // Validate mtime: opt must be at least as new as source and base cache
        boolean mtimeValid = optSdzFile.exists()
                && optSdzFile.lastModified() >= sourceFile.lastModified()
                && (baseCacheFile == null || !baseCacheFile.exists()
                || optSdzFile.lastModified() >= baseCacheFile.lastModified());
        boolean optValid = mtimeValid;

        // Validate optimizer code fingerprint: reject if optimizer code changed
        // since this cache entry was written (catches e.g. ConstantFunctionOptimizations
        // changes silently served by a mtime-only check).
        if (optValid) {
            String cachedFingerprint = readOptSdzFingerprint(optSdzFile);
            if (cachedFingerprint == null) {
                // Sidecar absent — legacy cache entry without fingerprint; treat as stale
                // so we re-optimize and write the sidecar this time.
                log.info("Stale .opt.sdz detected (no optimizer fingerprint sidecar), " +
                        "will re-optimize: {}", optSdzFile.getName());
                optValid = false;
            } else if (!cachedFingerprint.isEmpty()) {
                String currentFingerprint = getOptimizerFingerprint();
                if (!currentFingerprint.equals(cachedFingerprint)) {
                    log.info("Stale .opt.sdz detected (optimizer fingerprint mismatch), " +
                            "will re-optimize: {}", optSdzFile.getName());
                    optValid = false;
                }
            }
        }

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
            getOptimizedSdzMetaFile(optSdzFile).delete();
        } else if (optSdzFile.exists()) {
            // Mtime check failed (fingerprint cases already logged above)
            if (mtimeValid) {
                // optSdzFile existed but fingerprint invalidated it — delete it now
                optSdzFile.delete();
                getOptimizedSdzMetaFile(optSdzFile).delete();
            } else {
                log.info("Stale .opt.sdz detected (source is newer), will re-optimize: {}", optSdzFile.getName());
                optSdzFile.delete();
                getOptimizedSdzMetaFile(optSdzFile).delete();
            }
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
                // Write optimizer fingerprint sidecar so future loads can detect stale cache
                writeOptSdzMeta(optSdzFile);
                long saveElapsed = System.currentTimeMillis() - saveStart;
                log.info("Cached optimized SDZ in {}ms: {} ({} bytes)",
                        saveElapsed, optSdzFile.getName(), optSdzFile.length());
            } catch (Exception e) {
                log.warn("Failed to cache optimized SDZ (non-fatal): {}", e.getMessage());
                if (optSdzFile.exists()) {
                    optSdzFile.delete();
                }
                getOptimizedSdzMetaFile(optSdzFile).delete();
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
