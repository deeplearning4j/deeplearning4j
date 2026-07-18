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

package org.nd4j.autodiff.samediff.execution;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Disk cache for serialized DSP plan bytes.
 *
 * <p>Mirrors the Triton kernel disk cache pattern: FNV-1a keyed files with
 * atomic temp-file + rename writes, override directory support, and version-based
 * invalidation via a {@code .meta} sidecar file.</p>
 *
 * <p>Cache key: FNV-1a 64-bit hash of the serialized plan bytes
 * ({@link DynamicShapePlan#computeStructureHash(byte[])}). File naming:
 * {@code dsp_<16-hex>.bin} + {@code dsp_<16-hex>.meta}.</p>
 *
 * <p>The disk cache operates at the graph-structure level — one file per unique
 * graph topology. Per-thread plan instances are created by
 * {@code NativeDynamicShapePlan::fromSerializedPlan()} from the same bytes,
 * identically to the in-memory path.</p>
 */
@Slf4j
public class DspPlanDiskCache {

    private static final String DEFAULT_CACHE_SUBDIR = ".kompile/cache/dsp/dsp_plan_cache";
    private static final String DEFAULT_OVERRIDE_SUBDIR = ".kompile/cache/dsp/dsp_plan_override";

    // Current DSP serialization version — must match DynamicShapePlan.DSP_VERSION
    private static final int CURRENT_DSP_VERSION = 5;

    /**
     * Per-build native library fingerprint, computed once from buildInfo().
     * Changes on every .so rebuild (LIBND4J_BUILD_STAMP injected at compile time)
     * so plans cached by an older .so are automatically invalidated.
     * Stored as a volatile to allow lazy initialisation under the class lock.
     */
    private static volatile String NATIVE_BUILD_FINGERPRINT = null;

    /** Return the cached native build fingerprint, initialising it on first call. */
    static String getNativeBuildFingerprint() {
        if (NATIVE_BUILD_FINGERPRINT == null) {
            synchronized (DspPlanDiskCache.class) {
                if (NATIVE_BUILD_FINGERPRINT == null) {
                    try {
                        // buildInfo() is MULTI-LINE ("Build Info:\nGCC: ...\n...BuildStamp: ...").
                        // The .meta sidecar is line-oriented key=value and the reader parses a
                        // single line, so a raw multi-line value truncates on read to
                        // "Build Info:" and the equality check can never pass — every cached
                        // plan was rejected as a fingerprint mismatch. Prefer the native
                        // canonical single-line form; fall back to flattening buildInfo()
                        // identically for backend bindings generated before that API existed.
                        String info = null;
                        try {
                            info = NativeOpsHolder.getInstance()
                                    .getDeviceNativeOps().buildInfoFingerprint();
                        } catch (Throwable olderBindings) {
                            // pre-buildInfoFingerprint bindings — flatten below
                        }
                        if (info == null || info.isEmpty()) {
                            String raw = NativeOpsHolder.getInstance()
                                    .getDeviceNativeOps().buildInfo();
                            info = (raw != null)
                                    ? raw.replace('\r', ' ').replace('\n', ' ').trim()
                                    : null;
                        }
                        NATIVE_BUILD_FINGERPRINT = (info != null && !info.isEmpty())
                                ? info : "unavailable";
                    } catch (Exception e) {
                        log.warn("DspPlanDiskCache: could not read native buildInfo(), " +
                                "disk cache will use fallback fingerprint: {}", e.getMessage());
                        NATIVE_BUILD_FINGERPRINT = "unavailable";
                    }
                }
            }
        }
        return NATIVE_BUILD_FINGERPRINT;
    }

    private DspPlanDiskCache() {}

    // ── Configuration ──────────────────────────────────────────────────────

    /**
     * Whether the disk plan cache is enabled.
     * Controlled by {@code -Dnd4j.dsp.planCache.diskEnabled=true|false} (default: true).
     */
    public static boolean isEnabled() {
        return !"false".equalsIgnoreCase(
                System.getProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_ENABLED, "true"));
    }

    /**
     * Whether to force recompilation (bypass read, still write).
     * Controlled by {@code -Dnd4j.dsp.planCache.forceRecompile=true} (default: false).
     */
    public static boolean isForceRecompile() {
        return "true".equalsIgnoreCase(
                System.getProperty(ND4JSystemProperties.DSP_PLAN_CACHE_FORCE_RECOMPILE, "false"));
    }

    /**
     * Resolve the cache directory. Priority:
     * 1. {@code -Dnd4j.dsp.planCache.diskDir}
     * 2. {@code ND4J_DSP_PLAN_CACHE_DIR} env var
     * 3. {@code ~/.kompile/cache/dsp/dsp_plan_cache/}
     */
    public static String getCacheDir() {
        String prop = System.getProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_DIR);
        if (prop != null && !prop.isEmpty()) return prop;

        String env = System.getenv("ND4J_DSP_PLAN_CACHE_DIR");
        if (env != null && !env.isEmpty()) return env;

        return System.getProperty("user.home") + File.separator + DEFAULT_CACHE_SUBDIR;
    }

    /**
     * Resolve the override directory (highest priority lookup, never written to).
     */
    public static String getOverrideDir() {
        String prop = System.getProperty(ND4JSystemProperties.DSP_PLAN_CACHE_OVERRIDE_DIR);
        if (prop != null && !prop.isEmpty()) return prop;

        String env = System.getenv("ND4J_DSP_PLAN_CACHE_OVERRIDE_DIR");
        if (env != null && !env.isEmpty()) return env;

        return System.getProperty("user.home") + File.separator + DEFAULT_OVERRIDE_SUBDIR;
    }

    // ── File naming ────────────────────────────────────────────────────────

    private static String hashToHex(long hash) {
        return String.format("%016x", hash);
    }

    private static String binFileName(long structureHash) {
        return "dsp_" + hashToHex(structureHash) + ".bin";
    }

    private static String metaFileName(long structureHash) {
        return "dsp_" + hashToHex(structureHash) + ".meta";
    }

    // ── Read path ──────────────────────────────────────────────────────────

    /**
     * Try to load plan bytes from disk cache by structure hash.
     *
     * <p>Checks override directory first (highest priority), then the regular
     * cache directory. Validates the {@code .meta} sidecar for version compatibility.</p>
     *
     * @param structureHash FNV-1a hash of the serialized plan bytes
     * @return the cached plan bytes, or {@code null} on miss/stale/corrupt
     */
    public static byte[] tryLoadByHash(long structureHash) {
        if (!isEnabled() || isForceRecompile()) return null;

        String binName = binFileName(structureHash);
        String metaName = metaFileName(structureHash);

        // Check override directory first
        String overrideDir = getOverrideDir();
        File overrideBin = new File(overrideDir, binName);
        File overrideMeta = new File(overrideDir, metaName);
        if (overrideBin.exists()) {
            byte[] bytes = tryLoadFromDir(overrideBin, overrideMeta, structureHash, "override");
            if (bytes != null) return bytes;
        }

        // Check regular cache directory
        String cacheDir = getCacheDir();
        File cacheBin = new File(cacheDir, binName);
        File cacheMeta = new File(cacheDir, metaName);
        if (cacheBin.exists()) {
            return tryLoadFromDir(cacheBin, cacheMeta, structureHash, "cache");
        }

        return null;
    }

    private static byte[] tryLoadFromDir(File binFile, File metaFile, long structureHash, String source) {
        // Validate .meta sidecar
        if (metaFile.exists()) {
            int metaVersion = readMetaVersion(metaFile);
            if (metaVersion > 0 && metaVersion != CURRENT_DSP_VERSION) {
                log.debug("DSP disk cache {}: stale entry dsp_{} (meta version {} != current {})",
                        source, hashToHex(structureHash), metaVersion, CURRENT_DSP_VERSION);
                return null;
            }
            // Validate native .so fingerprint: reject plans compiled by an older .so.
            // A missing fingerprint (legacy .meta without the field) is also treated as
            // stale — it means the entry was written before this fix, so we can't know
            // which .so produced it.
            String cachedFingerprint = readMetaFingerprint(metaFile);
            if (cachedFingerprint == null || cachedFingerprint.isEmpty()) {
                log.debug("DSP disk cache {}: stale entry dsp_{} (no native build fingerprint in .meta — " +
                                "legacy entry, invalidating)",
                        source, hashToHex(structureHash));
                return null;
            }
            String currentFingerprint = getNativeBuildFingerprint();
            if (!currentFingerprint.equals(cachedFingerprint)) {
                log.debug("DSP disk cache {}: stale entry dsp_{} (native build fingerprint mismatch; " +
                                "cached .so != current .so — invalidating)",
                        source, hashToHex(structureHash));
                return null;
            }
        }

        // Read .bin file
        try {
            byte[] bytes = Files.readAllBytes(binFile.toPath());
            if (!DynamicShapePlan.isValidSerializedPlan(bytes)) {
                log.warn("DSP disk cache {}: corrupt .bin file dsp_{} (invalid DSP1 header)",
                        source, hashToHex(structureHash));
                return null;
            }
            log.info("DSP disk cache HIT ({}): loaded {} bytes from dsp_{}",
                    source, bytes.length, hashToHex(structureHash));
            return bytes;
        } catch (IOException e) {
            log.warn("DSP disk cache {}: failed to read dsp_{}: {}",
                    source, hashToHex(structureHash), e.getMessage());
            return null;
        }
    }

    /**
     * Read the dspVersion from a .meta file. Returns -1 on any parse failure.
     */
    private static int readMetaVersion(File metaFile) {
        try (BufferedReader reader = new BufferedReader(new FileReader(metaFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("dspVersion=")) {
                    return Integer.parseInt(line.substring("dspVersion=".length()).trim());
                }
            }
        } catch (IOException | NumberFormatException e) {
            // Corrupt or unreadable meta — treat as stale
        }
        return -1;
    }

    /**
     * Read the nativeBuildFingerprint field from a .meta file.
     * Returns {@code null} if the field is absent (legacy cache entry without fingerprint).
     */
    private static String readMetaFingerprint(File metaFile) {
        try (BufferedReader reader = new BufferedReader(new FileReader(metaFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("nativeBuildFingerprint=")) {
                    return line.substring("nativeBuildFingerprint=".length()).trim();
                }
            }
        } catch (IOException e) {
            // Corrupt or unreadable meta — treat as stale
        }
        return null;
    }

    // ── Write path ─────────────────────────────────────────────────────────

    /**
     * Check whether a cached entry exists for the given structure hash.
     */
    public static boolean exists(long structureHash) {
        String binName = binFileName(structureHash);
        File cacheBin = new File(getCacheDir(), binName);
        if (cacheBin.exists()) return true;

        File overrideBin = new File(getOverrideDir(), binName);
        return overrideBin.exists();
    }

    /**
     * Store plan bytes to the disk cache atomically.
     *
     * <p>Writes to temp files first, then renames atomically (mirrors the Triton
     * kernel cache pattern from {@code TritonGraphBackend_cache.cpp}).</p>
     *
     * @param structureHash FNV-1a hash of the plan bytes
     * @param planBytes     raw serialized plan bytes (DSP1 format)
     * @param numSlots      number of slots (for .meta)
     * @param numExtInputs  number of external inputs (for .meta)
     * @param numOutputs    number of requested outputs (for .meta)
     * @param outputSet     comma-separated sorted output names (for .meta)
     */
    public static void store(long structureHash, byte[] planBytes,
                             int numSlots, int numExtInputs, int numOutputs, String outputSet) {
        if (!isEnabled()) return;

        String cacheDir = getCacheDir();
        if (!ensureCacheDir(cacheDir)) {
            log.warn("DSP disk cache: failed to create cache directory: {}", cacheDir);
            return;
        }

        String hexHash = hashToHex(structureHash);
        String binName = binFileName(structureHash);
        String metaName = metaFileName(structureHash);

        // Unique temp file suffix: pid.threadId
        long pid = ProcessHandle.current().pid();
        long tid = Thread.currentThread().getId();
        String tmpSuffix = ".tmp." + pid + "." + tid;

        try {
            // Write .bin atomically
            Path binTmp = new File(cacheDir, binName + tmpSuffix).toPath();
            Path binFinal = new File(cacheDir, binName).toPath();
            Files.write(binTmp, planBytes);
            Files.move(binTmp, binFinal, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);

            // Write .meta atomically
            Path metaTmp = new File(cacheDir, metaName + tmpSuffix).toPath();
            Path metaFinal = new File(cacheDir, metaName).toPath();
            String metaContent = buildMetaContent(numSlots, numExtInputs, numOutputs,
                    planBytes.length, hexHash, outputSet);
            Files.writeString(metaTmp, metaContent);
            Files.move(metaTmp, metaFinal, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);

            log.info("DSP disk cache: stored {} bytes as dsp_{}", planBytes.length, hexHash);

        } catch (IOException e) {
            log.warn("DSP disk cache: failed to store dsp_{}: {}", hexHash, e.getMessage());
            // Clean up temp files on failure
            try {
                Files.deleteIfExists(new File(cacheDir, binName + tmpSuffix).toPath());
                Files.deleteIfExists(new File(cacheDir, metaName + tmpSuffix).toPath());
            } catch (IOException ignored) {}
        }
    }

    private static String buildMetaContent(int numSlots, int numExtInputs, int numOutputs,
                                           int planBytesLen, String hexHash, String outputSet) {
        StringBuilder sb = new StringBuilder();
        sb.append("dspVersion=").append(CURRENT_DSP_VERSION).append('\n');
        sb.append("numSlots=").append(numSlots).append('\n');
        sb.append("numExternalInputs=").append(numExtInputs).append('\n');
        sb.append("numRequestedOutputs=").append(numOutputs).append('\n');
        sb.append("outputSet=").append(outputSet != null ? outputSet : "").append('\n');
        sb.append("planBytes=").append(planBytesLen).append('\n');
        sb.append("structureHash=").append(hexHash).append('\n');
        // Per-build native library fingerprint (contains LIBND4J_BUILD_STAMP from
        // build_stamp.h — changes on every rebuild so stale cached plans are rejected).
        sb.append("nativeBuildFingerprint=").append(getNativeBuildFingerprint()).append('\n');
        sb.append("createdAt=").append(Instant.now().toString()).append('\n');
        return sb.toString();
    }

    // ── Model identity lookup ──────────────────────────────────────────────

    /**
     * Compute a model identity hash from output names, external input keys, and
     * the plan's slot count. The slot count is critical to prevent structurally
     * different graphs that share the same output and input variable names from
     * colliding in the disk cache. Without it, e.g. a rank-3 graph with reshape
     * ops and a rank-2 graph without them would hash identically if they both
     * output "out" and use inputs "x", "w1", "w_proj".
     *
     * @param requestedOutputs output variable names
     * @param externalInputKeys external input names (constants, variables, placeholders)
     * @param numSlots number of execution slots in the compiled plan
     * @return FNV-1a hash of the sorted output + input names + slot count
     */
    public static long computeModelIdentityHash(Set<String> requestedOutputs, String[] externalInputKeys, int numSlots) {
        long hash = 0xcbf29ce484222325L; // FNV-1a offset basis
        // Sort outputs for determinism
        List<String> sortedOutputs = new ArrayList<>(requestedOutputs);
        Collections.sort(sortedOutputs);
        for (String name : sortedOutputs) {
            for (int i = 0; i < name.length(); i++) {
                hash ^= (name.charAt(i) & 0xFFL);
                hash *= 0x100000001b3L;
            }
            hash ^= 0L; // separator
            hash *= 0x100000001b3L;
        }
        // Mix in external input keys (already ordered by compiler)
        if (externalInputKeys != null) {
            for (String key : externalInputKeys) {
                if (key == null) continue;
                for (int i = 0; i < key.length(); i++) {
                    hash ^= (key.charAt(i) & 0xFFL);
                    hash *= 0x100000001b3L;
                }
                hash ^= 0xFF; // key separator
                hash *= 0x100000001b3L;
            }
        }
        // Mix in structural slot count to distinguish graphs with different topology
        // but identical variable names (e.g. rank-3 vs rank-2 logits graphs).
        hash ^= (numSlots & 0xFFL);
        hash *= 0x100000001b3L;
        hash ^= ((numSlots >> 8) & 0xFFL);
        hash *= 0x100000001b3L;
        hash ^= ((numSlots >> 16) & 0xFFL);
        hash *= 0x100000001b3L;
        hash ^= ((numSlots >> 24) & 0xFFL);
        hash *= 0x100000001b3L;
        // Mix in DSP version so serialization format changes auto-invalidate the index.
        hash ^= (CURRENT_DSP_VERSION & 0xFFL);
        hash *= 0x100000001b3L;
        // Mix in per-build native fingerprint so index entries created by an older .so
        // never collide with (or shadow) entries for the current .so build.
        String fingerprint = getNativeBuildFingerprint();
        for (int i = 0; i < fingerprint.length(); i++) {
            hash ^= (fingerprint.charAt(i) & 0xFFL);
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    /**
     * Try to load plan bytes using model identity (output names + external input keys + slot count).
     * This enables cross-JVM plan reuse without needing to compile the plan first.
     *
     * <p>Looks up the model identity index file ({@code dsp_model_<hash>.idx}),
     * which maps to the structure hash of the cached plan bytes.</p>
     *
     * @param requestedOutputs output variable names
     * @param externalInputKeys external input names
     * @param numSlots number of execution slots in the compiled plan
     * @return cached plan bytes, or {@code null} on miss
     */
    public static byte[] tryLoadByModelIdentity(Set<String> requestedOutputs, String[] externalInputKeys, int numSlots) {
        if (!isEnabled() || isForceRecompile()) return null;

        long modelHash = computeModelIdentityHash(requestedOutputs, externalInputKeys, numSlots);
        String idxName = "dsp_model_" + hashToHex(modelHash) + ".idx";

        // Check override dir first, then cache dir
        for (String dir : new String[]{getOverrideDir(), getCacheDir()}) {
            File idxFile = new File(dir, idxName);
            if (!idxFile.exists()) continue;

            try (BufferedReader reader = new BufferedReader(new FileReader(idxFile))) {
                String structureHashHex = reader.readLine();
                if (structureHashHex != null && !structureHashHex.isEmpty()) {
                    long structureHash = Long.parseUnsignedLong(structureHashHex.trim(), 16);
                    byte[] bytes = tryLoadByHash(structureHash);
                    if (bytes != null) {
                        log.info("DSP disk cache: model identity HIT — loaded plan via index dsp_model_{}",
                                hashToHex(modelHash));
                        return bytes;
                    }
                }
            } catch (IOException | NumberFormatException e) {
                log.debug("DSP disk cache: failed to read model index {}: {}", idxName, e.getMessage());
            }
        }

        return null;
    }

    /**
     * Store the model identity → structure hash mapping.
     */
    public static void storeModelIdentityIndex(Set<String> requestedOutputs, String[] externalInputKeys,
                                               int numSlots, long structureHash) {
        if (!isEnabled()) return;

        long modelHash = computeModelIdentityHash(requestedOutputs, externalInputKeys, numSlots);
        String idxName = "dsp_model_" + hashToHex(modelHash) + ".idx";
        String cacheDir = getCacheDir();

        if (!ensureCacheDir(cacheDir)) return;

        long pid = ProcessHandle.current().pid();
        long tid = Thread.currentThread().getId();
        String tmpSuffix = ".tmp." + pid + "." + tid;

        try {
            Path idxTmp = new File(cacheDir, idxName + tmpSuffix).toPath();
            Path idxFinal = new File(cacheDir, idxName).toPath();
            Files.writeString(idxTmp, hashToHex(structureHash) + "\n");
            Files.move(idxTmp, idxFinal, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            log.debug("DSP disk cache: failed to write model index {}: {}", idxName, e.getMessage());
        }
    }

    // ── Utilities ──────────────────────────────────────────────────────────

    /**
     * Delete a specific cached entry (both .bin and .meta).
     */
    public static void invalidate(long structureHash) {
        String cacheDir = getCacheDir();
        try {
            Files.deleteIfExists(new File(cacheDir, binFileName(structureHash)).toPath());
            Files.deleteIfExists(new File(cacheDir, metaFileName(structureHash)).toPath());
        } catch (IOException e) {
            log.warn("DSP disk cache: failed to invalidate dsp_{}: {}",
                    hashToHex(structureHash), e.getMessage());
        }
    }

    /**
     * List all cached structure hashes.
     */
    public static List<String> listCachedHashes() {
        List<String> hashes = new ArrayList<>();
        File dir = new File(getCacheDir());
        if (!dir.isDirectory()) return hashes;

        File[] files = dir.listFiles((d, name) -> name.startsWith("dsp_") && name.endsWith(".bin")
                && !name.contains(".tmp.") && !name.contains("model_"));
        if (files != null) {
            for (File f : files) {
                String name = f.getName();
                // Extract hash from "dsp_<16hex>.bin"
                String hash = name.substring(4, name.length() - 4);
                hashes.add(hash);
            }
        }
        return hashes;
    }

    /**
     * Create the cache directory recursively (mirrors TritonGraphBackend_cache.cpp::ensureDiskCacheDir).
     */
    public static boolean ensureCacheDir(String dir) {
        File d = new File(dir);
        if (d.isDirectory()) return true;
        return d.mkdirs();
    }
}
