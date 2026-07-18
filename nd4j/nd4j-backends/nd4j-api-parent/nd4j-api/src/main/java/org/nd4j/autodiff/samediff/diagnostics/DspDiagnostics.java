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

package org.nd4j.autodiff.samediff.diagnostics;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Central DSP diagnostics reporting system. Thin Java wrapper over the C++ DspDiagnostics
 * singleton, providing:
 * <ul>
 *   <li>12-category event recording with per-category enable flags</li>
 *   <li>Structured text and JSON report generation</li>
 *   <li>System property and environment variable configuration</li>
 *   <li>Legacy flag auto-mapping</li>
 * </ul>
 *
 * <p>Configuration via system properties:</p>
 * <pre>
 * -Dnd4j.dsp.diagnostics=COMPILE,EXECUTE,TIMING   # Enable specific categories
 * -Dnd4j.dsp.diagnostics=all                       # Enable all categories
 * -Dnd4j.dsp.diagnostics.level=detailed            # summary|detailed|full
 * -Dnd4j.dsp.diagnostics.file=/tmp/dsp-report.json # JSON output file
 * </pre>
 *
 * <p>Or via environment variables:</p>
 * <pre>
 * ND4J_DSP_DIAGNOSTICS=all
 * ND4J_DSP_DIAGNOSTICS_LEVEL=full
 * ND4J_DSP_DIAGNOSTICS_FILE=/tmp/dsp-report.json
 * </pre>
 */
public class DspDiagnostics {
    private static final Logger log = LoggerFactory.getLogger(DspDiagnostics.class);

    // Category constants matching C++ DspDiagCategory bitfield positions
    public static final int COMPILE  = (1 << 0);
    public static final int JIT      = (1 << 1);
    public static final int EXECUTE  = (1 << 2);
    public static final int TIMING   = (1 << 3);
    public static final int MEMORY   = (1 << 4);
    public static final int BACKEND  = (1 << 5);
    public static final int SHAPE    = (1 << 6);
    public static final int SEGMENT  = (1 << 7);
    public static final int FUSION   = (1 << 8);
    public static final int VERIFY   = (1 << 9);
    public static final int KV_CACHE = (1 << 10);
    public static final int FALLBACK = (1 << 11);
    public static final int TRANSFER = (1 << 12);
    public static final int EMULATED_REPLAY = (1 << 13);
    public static final int STREAM_SYNC    = (1 << 14);
    public static final int MULTI_DEVICE   = (1 << 15);
    public static final int GRAPH_REPLAY     = (1 << 16);
    public static final int SEGMENT_BUCKETS  = (1 << 17);
    public static final int LIFECYCLE        = (1 << 18);
    public static final int COLORING         = (1 << 19);
    public static final int NONE     = 0;
    public static final int ALL      = 0xFFFFF;

    // Detail levels matching C++ DspDiagLevel
    public static final int LEVEL_SUMMARY  = 0;
    public static final int LEVEL_DETAILED = 1;
    public static final int LEVEL_FULL     = 2;

    // Must match the C++ sCategoryNames table in DspDiagnostics.cpp — this
    // class overrides the native mask, so a name missing here silently
    // disables a category the native side supports.
    private static final String[] CATEGORY_NAMES = {
        "COMPILE", "JIT", "EXECUTE", "TIMING",
        "MEMORY", "BACKEND", "SHAPE", "SEGMENT",
        "FUSION", "VERIFY", "KV_CACHE", "FALLBACK",
        "TRANSFER", "EMULATED_REPLAY",
        "STREAM_SYNC", "MULTI_DEVICE", "GRAPH_REPLAY",
        "SEGMENT_BUCKETS", "LIFECYCLE", "COLORING"
    };

    private static volatile boolean initialized = false;
    private static int cachedMask = NONE;

    private DspDiagnostics() {}

    /**
     * Initialize the diagnostics system from system properties. Called automatically
     * on first use, but can be called explicitly for early initialization.
     */
    public static synchronized void initialize() {
        if (initialized) return;
        initialized = true;

        try {
            // Read system properties
            String categories = System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS);
            // Fallback to env var (surefire passes properties as env vars)
            if (categories == null || categories.isEmpty()) {
                categories = System.getenv("ND4J_DSP_DIAGNOSTICS");
            }
            if (categories != null && !categories.isEmpty()) {
                int mask = parseCategories(categories);
                Nd4j.getNativeOps().dspDiagSetCategories(mask);
                cachedMask = mask;
                log.info("DSP diagnostics enabled: {} (mask=0x{}) ",
                         categories, Integer.toHexString(mask));
            }

            String level = System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL);
            if (level == null) level = System.getenv("ND4J_DSP_DIAGNOSTICS_LEVEL");
            if (level != null) {
                int lvl = parseLevelString(level);
                Nd4j.getNativeOps().dspDiagSetLevel(lvl);
            }

            String file = System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_FILE);
            if (file == null) file = System.getenv("ND4J_DSP_DIAGNOSTICS_FILE");
            if (file != null && !file.isEmpty()) {
                Nd4j.getNativeOps().dspDiagSetJsonPath(file);
            }

            // Also apply legacy flag mapping (done in C++ constructor too, but
            // Java-side properties need explicit forwarding)
            applyLegacyProperties();

            cachedMask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
        } catch (Exception e) {
            log.warn("Failed to initialize DSP diagnostics: {}", e.getMessage());
        }
    }

    /**
     * Check if a diagnostic category is enabled. Fast path: checks cached mask first,
     * falls back to JNI only if potentially stale.
     */
    public static boolean isEnabled(int category) {
        if (!initialized) initialize();
        return (cachedMask & category) != 0;
    }

    /**
     * Record a diagnostic event into the C++ ring buffer.
     */
    public static void record(int category, String message) {
        if (!isEnabled(category)) return;
        Nd4j.getNativeOps().dspDiagRecordJavaEvent(category, -1, -1, null, 0, message);
    }

    /**
     * Record a slot-specific diagnostic event.
     */
    public static void recordSlot(int category, int slotId, String opName, String message) {
        if (!isEnabled(category)) return;
        Nd4j.getNativeOps().dspDiagRecordJavaEvent(category, slotId, -1, opName, 0, message);
    }

    /**
     * Record a timed diagnostic event.
     */
    public static void recordTimed(int category, int slotId, int segmentId,
                                    String opName, long timingUs, String message) {
        if (!isEnabled(category)) return;
        Nd4j.getNativeOps().dspDiagRecordJavaEvent(
                category, slotId, segmentId, opName, timingUs, message);
    }

    /**
     * Set enabled categories (replaces current mask).
     *
     * <p>When {@code mask} is {@link #NONE}, the {@code initialized} flag is reset so that
     * the next call to {@link #isEnabled(int)} re-applies the environment-variable-configured
     * categories (ND4J_DSP_DIAGNOSTICS / nd4j.dsp.diagnostics). This ensures that test
     * cleanup code that calls {@code setCategories(NONE)} does not permanently silence
     * diagnostics configured via env vars for the remainder of the JVM lifetime.</p>
     *
     * <p>Env vars are re-read once on the next {@code isEnabled()} call, NOT per event,
     * so the hot-path overhead is a single atomic load per event.</p>
     */
    public static synchronized void setCategories(int mask) {
        Nd4j.getNativeOps().dspDiagSetCategories(mask);
        cachedMask = mask;
        // If diagnostics are being fully disabled, reset the initialization guard so
        // the env-var-configured mask is re-applied on the next isEnabled() call.
        if (mask == NONE) {
            initialized = false;
        }
    }

    /**
     * Enable additional categories.
     */
    public static void enableCategories(int mask) {
        Nd4j.getNativeOps().dspDiagEnableCategories(mask);
        cachedMask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
    }

    /**
     * Set the diagnostic output level.
     * @param level LEVEL_SUMMARY (0), LEVEL_DETAILED (1), or LEVEL_FULL (2)
     */
    public static void setLevel(int level) {
        Nd4j.getNativeOps().dspDiagSetLevel(level);
    }

    /**
     * Disable specific categories.
     */
    public static void disableCategories(int mask) {
        Nd4j.getNativeOps().dspDiagDisableCategories(mask);
        cachedMask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
    }

    /**
     * Get the structured text plan report from the C++ diagnostics singleton.
     */
    public static String getPlanReport() {
        return Nd4j.getNativeOps().dspDiagGetPlanReport();
    }

    /**
     * Get the JSON plan report from the C++ diagnostics singleton.
     */
    public static String getJsonReport() {
        return Nd4j.getNativeOps().dspDiagGetJsonReport();
    }

    /**
     * Clear all diagnostic events and stats.
     */
    public static void clear() {
        Nd4j.getNativeOps().dspDiagClear();
    }

    /**
     * Parse a comma-separated category string into a bitmask.
     * Supports "all", "none", or individual names like "COMPILE,EXECUTE,TIMING".
     */
    public static int parseCategories(String str) {
        if (str == null || str.isEmpty()) return NONE;
        str = str.trim().toUpperCase();
        if ("ALL".equals(str) || "*".equals(str)) return ALL;
        if ("NONE".equals(str) || "OFF".equals(str) || "0".equals(str)) return NONE;

        int mask = NONE;
        for (String token : str.split(",")) {
            token = token.trim();
            for (int i = 0; i < CATEGORY_NAMES.length; i++) {
                if (CATEGORY_NAMES[i].equals(token)) {
                    mask |= (1 << i);
                    break;
                }
            }
        }
        return mask;
    }

    private static int parseLevelString(String level) {
        if (level == null) return LEVEL_SUMMARY;
        switch (level.trim().toLowerCase()) {
            case "full":
            case "2":
                return LEVEL_FULL;
            case "detailed":
            case "1":
                return LEVEL_DETAILED;
            default:
                return LEVEL_SUMMARY;
        }
    }

    /**
     * Map legacy system properties to diagnostic categories.
     */
    private static void applyLegacyProperties() {
        int extra = NONE;

        // nd4j.dsp.trace -> EXECUTE
        if (System.getProperty(ND4JSystemProperties.DSP_TRACE) != null) {
            extra |= EXECUTE;
        }

        // nd4j.dsp.executionTiming -> TIMING
        if (System.getProperty(ND4JSystemProperties.DSP_EXECUTION_TIMING) != null) {
            extra |= TIMING;
        }

        // nd4j.dsp.native.dumpOutputs -> VERIFY
        if ("true".equalsIgnoreCase(
                System.getProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS))) {
            extra |= VERIFY;
        }

        // nd4j.dsp.java.dumpOutputs -> VERIFY
        if ("true".equalsIgnoreCase(
                System.getProperty(ND4JSystemProperties.DSP_JAVA_DUMP_OUTPUTS))) {
            extra |= VERIFY;
        }

        if (extra != NONE) {
            Nd4j.getNativeOps().dspDiagEnableCategories(extra);
        }
    }
}
