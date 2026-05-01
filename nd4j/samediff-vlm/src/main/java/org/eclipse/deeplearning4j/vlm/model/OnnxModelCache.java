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

package org.eclipse.deeplearning4j.vlm.model;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Caching layer for ONNX model imports.
 *
 * <p>Wraps {@link OnnxFrameworkImporter#runImport} with SDZ file caching.
 * On the first import, the ONNX model is fully parsed and converted to SameDiff,
 * then saved as an SDZ file alongside the original ONNX file. On subsequent runs,
 * the SDZ file is loaded directly, bypassing ONNX parsing entirely.</p>
 *
 * <p>This reduces model loading time from ~5 minutes to ~30 seconds for
 * typical VLM model sets (vision encoder + decoder + embed tokens).</p>
 *
 * <p>Cache invalidation uses file modification timestamps: if the ONNX file
 * is newer than the cached SDZ, the cache is regenerated.</p>
 *
 * Adam Gibson
 */
@Slf4j
public class OnnxModelCache {

    /**
     * System property to disable SDZ caching entirely.
     * Set to "true" to always import from ONNX.
     */
    public static final String DISABLE_CACHE_PROPERTY = "vlm.model.cache.disable";

    /**
     * System property to control graph optimization after loading.
     * Defaults to ON (true). Set to "false" to disable GraphOptimizer
     * (constant folding, identity removal, attention fusion, FP16 pre-cast, etc.).
     * Optimized graphs are cached separately as .opt.sdz files.
     */
    public static final String OPTIMIZER_ENABLED_PROPERTY = "nd4j.optimizer.enabled";

    private OnnxModelCache() {
        // utility class
    }

    /**
     * Check if the optimizer is enabled. Defaults to true unless
     * explicitly set to "false" via system property.
     */
    private static boolean isOptimizerEnabled() {
        String prop = System.getProperty(OPTIMIZER_ENABLED_PROPERTY);
        // Default to true — only disable if explicitly set to "false"
        return prop == null || !"false".equalsIgnoreCase(prop.trim());
    }

    /**
     * Import an ONNX model with SDZ file caching.
     *
     * <p>If a cached SDZ file exists alongside the ONNX file and is newer
     * than the ONNX source, the cached SDZ is loaded directly. Otherwise,
     * the full ONNX import is performed and the result is cached as SDZ
     * for future runs.</p>
     *
     * @param onnxFilePath absolute path to the ONNX model file
     * @return the imported SameDiff model
     * @throws IOException if import or cache operations fail
     */
    private static final int MAX_LOAD_RETRIES = 3;
    private static final long RETRY_DELAY_MS = 500;

    public static SameDiff importWithCache(String onnxFilePath) throws IOException {
        File onnxFile = new File(onnxFilePath);
        if (!onnxFile.exists()) {
            throw new IOException("ONNX file not found: " + onnxFilePath);
        }

        File sdzFile = getSdzCacheFile(onnxFile);
        boolean cacheDisabled = Boolean.getBoolean(DISABLE_CACHE_PROPERTY);

        // Acquire file lock to prevent concurrent read/write/delete of the SDZ cache.
        // The lock file is adjacent to the SDZ file — all processes/threads contending
        // for this model will serialize through this lock.
        File lockFile = new File(sdzFile.getAbsolutePath() + ".lock");
        lockFile.getParentFile().mkdirs();
        try (RandomAccessFile raf = new RandomAccessFile(lockFile, "rw");
             FileChannel channel = raf.getChannel();
             FileLock lock = channel.lock()) {
            return importWithCacheLocked(onnxFile, sdzFile, cacheDisabled);
        } catch (IOException e) {
            // If locking fails (e.g., NFS without lock support), fall through without lock
            log.warn("File locking failed for {}, proceeding without lock: {}", lockFile, e.getMessage());
            return importWithCacheLocked(onnxFile, sdzFile, cacheDisabled);
        }
    }

    private static SameDiff importWithCacheLocked(File onnxFile, File sdzFile, boolean cacheDisabled) throws IOException {

        boolean optimizerEnabled = isOptimizerEnabled();

        // Check for cached optimized SDZ first (if optimizer is enabled)
        // The .opt.sdz must be newer than BOTH the .onnx source AND the base .sdz cache.
        // If the base .sdz was regenerated (new import), the .opt.sdz is stale and must be re-optimized.
        if (optimizerEnabled && !cacheDisabled) {
            File optSdzFile = getOptimizedSdzCacheFile(onnxFile);
            File baseSdzFile = getSdzCacheFile(onnxFile);
            boolean optValid = optSdzFile.exists()
                    && optSdzFile.lastModified() >= onnxFile.lastModified()
                    && (!baseSdzFile.exists() || optSdzFile.lastModified() >= baseSdzFile.lastModified());
            if (optValid) {
                log.info("Loading cached optimized SDZ model: {} ({} bytes)", optSdzFile.getName(), optSdzFile.length());
                long start = System.currentTimeMillis();
                SameDiff sd = loadSdzWithRetry(optSdzFile);
                if (sd != null) {
                    long elapsed = System.currentTimeMillis() - start;
                    log.info("Loaded cached optimized SDZ model in {}ms: {}", elapsed, optSdzFile.getName());
                    return sd;
                }
                log.warn("Failed to load optimized SDZ after retries, will re-import: {}", optSdzFile.getName());
                optSdzFile.delete();
            } else if (optSdzFile.exists()) {
                log.info("Stale .opt.sdz detected, will re-optimize: {}", optSdzFile.getName());
                optSdzFile.delete();
            }
        }

        // Use cached SDZ if it exists and is newer than the ONNX source
        if (!cacheDisabled && sdzFile.exists() && sdzFile.lastModified() >= onnxFile.lastModified()) {
            log.info("Loading cached SDZ model: {} ({} bytes)", sdzFile.getName(), sdzFile.length());
            long start = System.currentTimeMillis();
            SameDiff sd = loadSdzWithRetry(sdzFile);
            if (sd != null) {
                long elapsed = System.currentTimeMillis() - start;
                log.info("Loaded cached SDZ model in {}ms: {}", elapsed, sdzFile.getName());
                return maybeOptimize(sd, onnxFile, optimizerEnabled, cacheDisabled);
            }
            log.warn("Failed to load cached SDZ after retries, will re-import: {}", sdzFile.getName());
            sdzFile.delete();
        }

        // Full ONNX import (first run or cache invalidated)
        log.info("Importing ONNX model: {} (will cache as SDZ)", onnxFile.getName());
        long importStart = System.currentTimeMillis();
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(onnxFile.getAbsolutePath(), Map.of(), false, false);
        long importElapsed = System.currentTimeMillis() - importStart;
        log.info("ONNX import completed in {}ms: {}", importElapsed, onnxFile.getName());

        // Cache for future runs
        try {
            long saveStart = System.currentTimeMillis();
            SDZSerializer.save(sd, sdzFile, false, Map.of(
                    "source_onnx", onnxFile.getName(),
                    "import_timestamp", String.valueOf(System.currentTimeMillis())
            ));
            long saveElapsed = System.currentTimeMillis() - saveStart;
            log.info("Cached SDZ model in {}ms: {} ({} bytes)", saveElapsed,
                    sdzFile.getName(), sdzFile.length());
        } catch (Exception e) {
            log.warn("Failed to cache SDZ model (non-fatal): {}", e.getMessage());
            // Delete partial SDZ file if save failed
            if (sdzFile.exists()) {
                sdzFile.delete();
            }
        }

        return maybeOptimize(sd, onnxFile, optimizerEnabled, cacheDisabled);
    }

    /**
     * Optionally run GraphOptimizer on the loaded model and cache the result.
     */
    private static SameDiff maybeOptimize(SameDiff sd, File onnxFile, boolean optimizerEnabled, boolean cacheDisabled) {
        if (!optimizerEnabled) {
            return sd;
        }

        int opsBefore = sd.getOps().size();
        log.info("Running GraphOptimizer on {} ({} ops)...", onnxFile.getName(), opsBefore);
        long optStart = System.currentTimeMillis();

        List<String> outputs = sd.outputs() != null ? new ArrayList<>(sd.outputs()) : new ArrayList<>();
        SameDiff optimized = GraphOptimizer.optimize(sd, outputs);

        int opsAfter = optimized.getOps().size();
        long optElapsed = System.currentTimeMillis() - optStart;
        log.info("GraphOptimizer: {} -> {} ops ({} removed) in {}ms for {}",
                opsBefore, opsAfter, opsBefore - opsAfter, optElapsed, onnxFile.getName());

        // Cache the optimized graph for future runs
        if (!cacheDisabled) {
            File optSdzFile = getOptimizedSdzCacheFile(onnxFile);
            try {
                long saveStart = System.currentTimeMillis();
                SDZSerializer.save(optimized, optSdzFile, false, Map.of(
                        "source_onnx", onnxFile.getName(),
                        "optimized", "true",
                        "ops_before", String.valueOf(opsBefore),
                        "ops_after", String.valueOf(opsAfter)
                ));
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

    /**
     * Load an SDZ file with retry logic for transient ZIP extraction failures.
     * Returns null if all retries fail (caller should fall back to re-import).
     */
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

    /**
     * Import multiple ONNX models in parallel with SDZ caching.
     *
     * <p>Each model is imported/loaded on a separate thread. This is safe
     * because ONNX parsing is CPU-bound and each OnnxFrameworkImporter
     * instance is independent. SDZ loading is I/O-bound.</p>
     *
     * <p>All import threads are pinned to the primary GPU (the device with the
     * most total memory). This prevents model constants from being scattered
     * across multiple GPUs during loading — a smaller GPU may not have enough
     * memory for a large model's constants, causing failover splits that lead
     * to cross-device CUDA errors (error 700) during execution.</p>
     *
     * @param onnxFilePaths absolute paths to ONNX model files
     * @return array of imported SameDiff models in the same order as the input paths
     * @throws IOException if any import fails
     */
    public static SameDiff[] importAllWithCache(String... onnxFilePaths) throws IOException {
        if (onnxFilePaths.length == 0) {
            return new SameDiff[0];
        }
        if (onnxFilePaths.length == 1) {
            return new SameDiff[]{importWithCache(onnxFilePaths[0])};
        }

        // Disable DSP and CUDA graphs during model loading. Loading multiple models
        // simultaneously is peak GPU memory usage — DSP compilation and CUDA graph
        // capture allocate additional cached arrays and graph copies that push past OOM.
        boolean dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
        String prevCudaGraphs = System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
        InferenceSession.setDynamicShapePlanEnabled(false);
        System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "false");
        log.info("DSP and CUDA graphs disabled during model loading");

        // Find the primary GPU (most total memory) for model loading.
        int primaryDevice = selectPrimaryDevice();

        // Check if ALL models have SDZ caches. If not, import sequentially to avoid
        // CUDA thread-safety issues with parallel ONNX imports (each import creates
        // thousands of NDArrays on the GPU, and concurrent CUDA context operations
        // from multiple threads can cause stream synchronization failures).
        boolean allCached = true;
        boolean cacheDisabled = Boolean.getBoolean(DISABLE_CACHE_PROPERTY);
        if (!cacheDisabled) {
            for (String path : onnxFilePaths) {
                File onnxFile = new File(path);
                boolean optimizerEnabled = isOptimizerEnabled();
                File optSdzFile = getOptimizedSdzCacheFile(onnxFile);
                File sdzFile = getSdzCacheFile(onnxFile);
                boolean hasCachedOpt = optimizerEnabled && optSdzFile.exists() &&
                        optSdzFile.lastModified() >= onnxFile.lastModified();
                boolean hasCached = sdzFile.exists() && sdzFile.lastModified() >= onnxFile.lastModified();
                if (!hasCachedOpt && !hasCached) {
                    allCached = false;
                    break;
                }
            }
        } else {
            allCached = false;
        }

        long start = System.currentTimeMillis();

        try {
            SameDiff[] results;
            if (allCached) {
                // All models have SDZ caches — safe to load in parallel (I/O-bound, no CUDA contention)
                log.info("Loading {} cached models in parallel (device {})...",
                        onnxFilePaths.length, primaryDevice);
                results = importAllParallel(onnxFilePaths, primaryDevice);
            } else {
                // Some models need ONNX import — import sequentially to avoid CUDA thread-safety issues
                log.info("Importing {} models sequentially (some need ONNX import)...",
                        onnxFilePaths.length);
                results = new SameDiff[onnxFilePaths.length];
                for (int i = 0; i < onnxFilePaths.length; i++) {
                    results[i] = importWithCache(onnxFilePaths[i]);
                }
            }

            long elapsed = System.currentTimeMillis() - start;
            log.info("All {} models loaded in {}ms (device {})",
                    onnxFilePaths.length, elapsed, primaryDevice);
            return results;
        } finally {
            // Restore DSP and CUDA graph settings
            InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled);
            if (prevCudaGraphs != null) {
                System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, prevCudaGraphs);
            } else {
                System.clearProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED);
            }
            log.info("DSP and CUDA graphs restored after model loading (dsp={}, cudaGraphs={})",
                    dspWasEnabled, prevCudaGraphs != null ? prevCudaGraphs : "default");
        }
    }

    /**
     * Import all models in parallel using a thread pool.
     */
    private static SameDiff[] importAllParallel(String[] onnxFilePaths, int primaryDevice) throws IOException {
        ExecutorService executor = Executors.newFixedThreadPool(onnxFilePaths.length);
        try {
            @SuppressWarnings("unchecked")
            Future<SameDiff>[] futures = new Future[onnxFilePaths.length];
            for (int i = 0; i < onnxFilePaths.length; i++) {
                final String path = onnxFilePaths[i];
                final int deviceId = primaryDevice;
                futures[i] = executor.submit(() -> {
                    DeviceMemoryManager.getInstance().switchDevice(
                            deviceId, "OnnxModelCache.importAllWithCache", "pin-import-thread");
                    return importWithCache(path);
                });
            }

            SameDiff[] results = new SameDiff[onnxFilePaths.length];
            for (int i = 0; i < futures.length; i++) {
                try {
                    results[i] = futures[i].get();
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    if (cause instanceof IOException) {
                        throw (IOException) cause;
                    }
                    throw new IOException("Failed to import: " + onnxFilePaths[i], cause);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Import interrupted: " + onnxFilePaths[i], e);
                }
            }
            return results;
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Select the primary GPU device (the one with the most total memory).
     * This is where model constants should be loaded to avoid cross-device splits.
     *
     * @return device ID of the GPU with the most total memory
     */
    private static int selectPrimaryDevice() {
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        // Always query ALL physical GPUs (cudaGetDeviceCount), not just the ND4J-visible list.
        // The ND4J affinity manager may restrict devices, but for model loading we want the
        // biggest GPU regardless.
        int physicalDevices = Nd4j.getNativeOps().getAvailableDevices();
        log.info("selectPrimaryDevice: affinityManager reports {} devices, native reports {} physical GPUs",
                numDevices, physicalDevices);

        // Use the max of ND4J visible and physical device count to search ALL GPUs
        int searchCount = Math.max(numDevices, physicalDevices);
        if (searchCount <= 1) {
            log.info("selectPrimaryDevice: single device system, using device 0");
            return 0;
        }

        int bestDevice = 0;
        long bestTotal = 0;
        for (int d = 0; d < searchCount; d++) {
            long totalMem = Nd4j.getNativeOps().getDeviceTotalMemory(d);
            long freeMem = Nd4j.getNativeOps().getDeviceFreeMemory(d);
            log.info("selectPrimaryDevice: device {} - total={} MB, free={} MB",
                    d, totalMem / (1024 * 1024), freeMem / (1024 * 1024));
            if (totalMem > bestTotal) {
                bestTotal = totalMem;
                bestDevice = d;
            }
        }
        log.info("Primary device for model loading: device {} ({} MB total, {} devices searched)",
                bestDevice, bestTotal / (1024 * 1024), searchCount);
        return bestDevice;
    }

    /**
     * Delete cached SDZ files for an ONNX model, forcing re-import on next load.
     *
     * @param onnxFilePath absolute path to the ONNX model file
     * @return true if a cache file was deleted
     */
    public static boolean invalidateCache(String onnxFilePath) {
        File onnxFile = new File(onnxFilePath);
        boolean anyDeleted = false;

        // Delete base SDZ cache
        File sdzFile = getSdzCacheFile(onnxFile);
        if (sdzFile.exists()) {
            boolean deleted = sdzFile.delete();
            if (deleted) {
                log.info("Invalidated cache: {}", sdzFile.getName());
                anyDeleted = true;
            }
        }

        // Also delete optimized SDZ cache
        File optSdzFile = getOptimizedSdzCacheFile(onnxFile);
        if (optSdzFile.exists()) {
            boolean deleted = optSdzFile.delete();
            if (deleted) {
                log.info("Invalidated optimized cache: {}", optSdzFile.getName());
                anyDeleted = true;
            }
        }

        return anyDeleted;
    }

    /**
     * Check if a cached SDZ file exists and is valid for the given ONNX file.
     *
     * @param onnxFilePath absolute path to the ONNX model file
     * @return true if a valid cache exists
     */
    public static boolean isCached(String onnxFilePath) {
        File onnxFile = new File(onnxFilePath);
        File sdzFile = getSdzCacheFile(onnxFile);
        return sdzFile.exists() && sdzFile.lastModified() >= onnxFile.lastModified();
    }

    /**
     * Get the SDZ cache file path for a given ONNX file.
     * The SDZ file is placed alongside the ONNX file with the same base name.
     */
    private static File getSdzCacheFile(File onnxFile) {
        String name = onnxFile.getName();
        String baseName;
        int dotIdx = name.lastIndexOf('.');
        if (dotIdx > 0) {
            baseName = name.substring(0, dotIdx);
        } else {
            baseName = name;
        }
        return new File(onnxFile.getParentFile(), baseName + ".sdz");
    }

    /**
     * Get the optimized SDZ cache file path for a given ONNX file.
     */
    private static File getOptimizedSdzCacheFile(File onnxFile) {
        String name = onnxFile.getName();
        String baseName;
        int dotIdx = name.lastIndexOf('.');
        if (dotIdx > 0) {
            baseName = name.substring(0, dotIdx);
        } else {
            baseName = name;
        }
        return new File(onnxFile.getParentFile(), baseName + ".opt.sdz");
    }
}
