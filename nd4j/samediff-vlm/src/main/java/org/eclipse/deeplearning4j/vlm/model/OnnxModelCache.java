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
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
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
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class OnnxModelCache {

    /**
     * System property to disable SDZ caching entirely.
     * Set to "true" to always import from ONNX.
     */
    public static final String DISABLE_CACHE_PROPERTY = "vlm.model.cache.disable";

    private OnnxModelCache() {
        // utility class
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
    public static SameDiff importWithCache(String onnxFilePath) throws IOException {
        File onnxFile = new File(onnxFilePath);
        if (!onnxFile.exists()) {
            throw new IOException("ONNX file not found: " + onnxFilePath);
        }

        File sdzFile = getSdzCacheFile(onnxFile);
        boolean cacheDisabled = Boolean.getBoolean(DISABLE_CACHE_PROPERTY);

        // Use cached SDZ if it exists and is newer than the ONNX source
        if (!cacheDisabled && sdzFile.exists() && sdzFile.lastModified() >= onnxFile.lastModified()) {
            log.info("Loading cached SDZ model: {} ({} bytes)", sdzFile.getName(), sdzFile.length());
            long start = System.currentTimeMillis();
            SameDiff sd = SDZSerializer.load(sdzFile, false);
            long elapsed = System.currentTimeMillis() - start;
            log.info("Loaded cached SDZ model in {}ms: {}", elapsed, sdzFile.getName());
            return sd;
        }

        // Full ONNX import (first run or cache invalidated)
        log.info("Importing ONNX model: {} (will cache as SDZ)", onnxFile.getName());
        long importStart = System.currentTimeMillis();
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(onnxFilePath, Map.of(), false, false);
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

        return sd;
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

        // Find the primary GPU (most total memory) for model loading.
        // All import threads are pinned to this device to prevent model constants
        // from being split across non-peer GPUs during allocation.
        int primaryDevice = selectPrimaryDevice();
        log.info("Importing {} ONNX models in parallel (all pinned to device {})...",
                onnxFilePaths.length, primaryDevice);
        long start = System.currentTimeMillis();

        ExecutorService executor = Executors.newFixedThreadPool(onnxFilePaths.length);
        try {
            @SuppressWarnings("unchecked")
            Future<SameDiff>[] futures = new Future[onnxFilePaths.length];
            for (int i = 0; i < onnxFilePaths.length; i++) {
                final String path = onnxFilePaths[i];
                final int deviceId = primaryDevice;
                futures[i] = executor.submit(() -> {
                    // Pin this thread to the primary GPU before loading
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

            long elapsed = System.currentTimeMillis() - start;
            log.info("All {} models loaded in {}ms (parallel, device {})",
                    onnxFilePaths.length, elapsed, primaryDevice);
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
        if (numDevices <= 1) return 0;

        int bestDevice = 0;
        long bestTotal = 0;
        for (int d = 0; d < numDevices; d++) {
            long totalMem = Nd4j.getNativeOps().getDeviceTotalMemory(d);
            if (totalMem > bestTotal) {
                bestTotal = totalMem;
                bestDevice = d;
            }
        }
        log.info("Primary device for model loading: device {} ({} MB total)",
                bestDevice, bestTotal / (1024 * 1024));
        return bestDevice;
    }

    /**
     * Delete cached SDZ files for an ONNX model, forcing re-import on next load.
     *
     * @param onnxFilePath absolute path to the ONNX model file
     * @return true if a cache file was deleted
     */
    public static boolean invalidateCache(String onnxFilePath) {
        File sdzFile = getSdzCacheFile(new File(onnxFilePath));
        if (sdzFile.exists()) {
            boolean deleted = sdzFile.delete();
            if (deleted) {
                log.info("Invalidated cache: {}", sdzFile.getName());
            }
            return deleted;
        }
        return false;
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
}
