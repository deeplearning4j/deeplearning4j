/*
 * *****************************************************************************
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ****************************************************************************
 */

package org.nd4j.common.tests.results;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Persistent test result storage outside the source tree.
 * <p>
 * Results are written as JSON Lines to {@code ~/.dl4j/test-results/} (configurable
 * via system property {@code dl4j.test.results.dir}). Each line is a self-contained
 * JSON object with all metadata, so multiple forked JVMs can safely append to the
 * same daily file.
 * <p>
 * This class is intentionally dependency-free beyond the JDK — no Jackson, no Gson.
 * JSON is simple enough to format inline for the flat structures we write.
 */
@Slf4j
public class TestResultStore {

    /** System property to override the results directory. */
    public static final String RESULTS_DIR_PROPERTY = "dl4j.test.results.dir";

    /** System property to disable result recording entirely. */
    public static final String DISABLED_PROPERTY = "dl4j.test.results.disabled";

    private static final String DEFAULT_DIR = System.getProperty("user.home") + "/.dl4j/test-results";
    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    // Cached git/env metadata — resolved once per JVM
    private static volatile String cachedCommit;
    private static volatile String cachedBranch;
    private static volatile String cachedBackend;
    private static volatile Boolean disabled;

    private TestResultStore() {}

    // ─── Public API ─────────────────────────────────────────────

    /**
     * Returns true if result recording is disabled via system property.
     */
    public static boolean isDisabled() {
        if (disabled == null) {
            disabled = Boolean.parseBoolean(System.getProperty(DISABLED_PROPERTY, "false"));
        }
        return disabled;
    }

    /**
     * Returns the results directory, creating it if necessary.
     */
    public static Path getResultsDir() {
        String dir = System.getProperty(RESULTS_DIR_PROPERTY);
        // Maven passes unresolved placeholders like "${dl4j.test.results.dir}" when
        // the property wasn't set on the command line. Treat those as absent.
        if (dir == null || dir.isEmpty() || dir.startsWith("${")) {
            dir = DEFAULT_DIR;
        }
        Path path = Paths.get(dir);
        try {
            Files.createDirectories(path);
        } catch (IOException e) {
            // Will fail on write — caller can handle
        }
        return path;
    }

    /**
     * Appends a test result to the daily results file.
     *
     * @param testClass   fully qualified class name
     * @param testMethod  method name (may be null for class-level events)
     * @param displayName JUnit display name
     * @param status      PASS, FAIL, SKIP, ABORT
     * @param durationMs  wall-clock duration in milliseconds
     * @param tags        JUnit tags on the test (may be null/empty)
     * @param errorMessage error message if failed (may be null)
     * @param dspMetrics  DSP metrics map (may be null)
     */
    public static void appendResult(String testClass, String testMethod, String displayName,
                                     String status, long durationMs, Set<String> tags,
                                     String errorMessage, Map<String, Object> dspMetrics) {
        if (isDisabled()) return;
        try {
            StringBuilder sb = new StringBuilder(512);
            sb.append('{');
            jsonField(sb, "type", "test", true);
            jsonField(sb, "ts", Instant.now().toString(), false);
            jsonField(sb, "commit", getCommit(), false);
            jsonField(sb, "branch", getBranch(), false);
            jsonField(sb, "backend", getBackend(), false);
            jsonField(sb, "testClass", testClass, false);
            if (testMethod != null) {
                jsonField(sb, "testMethod", testMethod, false);
            }
            if (displayName != null) {
                jsonField(sb, "displayName", displayName, false);
            }
            jsonField(sb, "status", status, false);
            sb.append(",\"durationMs\":").append(durationMs);
            if (tags != null && !tags.isEmpty()) {
                sb.append(",\"tags\":[");
                boolean first = true;
                for (String tag : tags) {
                    if (!first) sb.append(',');
                    sb.append('"').append(escapeJson(tag)).append('"');
                    first = false;
                }
                sb.append(']');
            }
            if (errorMessage != null) {
                jsonField(sb, "error", truncate(errorMessage, 500), false);
            }
            if (dspMetrics != null && !dspMetrics.isEmpty()) {
                sb.append(",\"dsp\":");
                appendJsonObject(sb, dspMetrics);
            }
            sb.append('}');

            appendLine(getDailyFile(), sb.toString());
        } catch (Exception e) {
            // Never let recording break tests
            log.warn("Failed to record result: {}", e.getMessage());
        }
    }

    /**
     * Appends a milestone entry to the milestones file.
     */
    public static void appendMilestone(String description, Map<String, Object> metrics) {
        if (isDisabled()) return;
        try {
            StringBuilder sb = new StringBuilder(256);
            sb.append('{');
            jsonField(sb, "type", "milestone", true);
            jsonField(sb, "ts", Instant.now().toString(), false);
            jsonField(sb, "commit", getCommit(), false);
            jsonField(sb, "branch", getBranch(), false);
            jsonField(sb, "backend", getBackend(), false);
            jsonField(sb, "description", description, false);
            if (metrics != null && !metrics.isEmpty()) {
                sb.append(",\"metrics\":");
                appendJsonObject(sb, metrics);
            }
            sb.append('}');
            appendLine(getMilestoneFile(), sb.toString());
        } catch (Exception e) {
            log.warn("Failed to record milestone: {}", e.getMessage());
        }
    }

    /**
     * Reads all lines from the daily results file for a given date.
     */
    public static List<String> readResults(LocalDate date) {
        Path file = getResultsDir().resolve("results-" + date.format(DATE_FMT) + ".jsonl");
        return readLines(file);
    }

    /**
     * Reads all lines from the milestones file.
     */
    public static List<String> readMilestones() {
        return readLines(getMilestoneFile());
    }

    /**
     * Lists all daily result files, sorted newest first.
     */
    public static List<Path> listResultFiles() {
        Path dir = getResultsDir();
        if (!Files.isDirectory(dir)) return Collections.emptyList();
        try (Stream<Path> stream = Files.list(dir)) {
            return stream
                    .filter(p -> p.getFileName().toString().startsWith("results-")
                            && p.getFileName().toString().endsWith(".jsonl"))
                    .sorted(Comparator.reverseOrder())
                    .collect(Collectors.toList());
        } catch (IOException e) {
            return Collections.emptyList();
        }
    }

    /**
     * Returns the daily results file path for today.
     */
    public static Path getDailyFile() {
        String date = LocalDate.now().format(DATE_FMT);
        return getResultsDir().resolve("results-" + date + ".jsonl");
    }

    /**
     * Returns the milestones file path.
     */
    public static Path getMilestoneFile() {
        return getResultsDir().resolve("milestones.jsonl");
    }

    // ─── Metadata resolution ────────────────────────────────────

    public static String getCommit() {
        if (cachedCommit == null) {
            cachedCommit = runGit("rev-parse", "--short", "HEAD");
        }
        return cachedCommit;
    }

    public static String getBranch() {
        if (cachedBranch == null) {
            cachedBranch = runGit("branch", "--show-current");
        }
        return cachedBranch;
    }

    public static String getBackend() {
        if (cachedBackend == null) {
            // Check Maven-set property first
            String artifact = System.getProperty("backend.artifactId");
            if (artifact != null && !artifact.isEmpty() && !artifact.startsWith("${")) {
                cachedBackend = artifact;
            } else {
                // Try detecting from loaded backend
                try {
                    Class<?> nd4j = Class.forName("org.nd4j.linalg.factory.Nd4j");
                    Object backend = nd4j.getMethod("getBackend").invoke(null);
                    if (backend != null) {
                        cachedBackend = backend.getClass().getSimpleName();
                    } else {
                        cachedBackend = "unknown";
                    }
                } catch (Exception e) {
                    cachedBackend = "unknown";
                }
            }
        }
        return cachedBackend;
    }

    // ─── Internal helpers ───────────────────────────────────────

    static void appendLine(Path file, String line) {
        try {
            Files.createDirectories(file.getParent());
            // StandardOpenOption.APPEND + CREATE: atomic-enough for concurrent forked JVMs
            Files.write(file, Collections.singletonList(line), StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            log.warn("Write failed to {}: {}", file, e.getMessage());
        }
    }

    static List<String> readLines(Path file) {
        if (!Files.exists(file)) return Collections.emptyList();
        try {
            return Files.readAllLines(file, StandardCharsets.UTF_8);
        } catch (IOException e) {
            return Collections.emptyList();
        }
    }

    static void jsonField(StringBuilder sb, String key, String value, boolean first) {
        if (!first) sb.append(',');
        sb.append('"').append(escapeJson(key)).append("\":\"")
                .append(escapeJson(value != null ? value : "")).append('"');
    }

    static void appendJsonObject(StringBuilder sb, Map<String, Object> map) {
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : map.entrySet()) {
            if (!first) sb.append(',');
            sb.append('"').append(escapeJson(e.getKey())).append("\":");
            Object v = e.getValue();
            if (v == null) {
                sb.append("null");
            } else if (v instanceof Number) {
                if (v instanceof Double || v instanceof Float) {
                    double d = ((Number) v).doubleValue();
                    if (Double.isNaN(d) || Double.isInfinite(d)) {
                        sb.append("null");
                    } else {
                        sb.append(String.format("%.4f", d));
                    }
                } else {
                    sb.append(v);
                }
            } else if (v instanceof Boolean) {
                sb.append(v);
            } else {
                sb.append('"').append(escapeJson(String.valueOf(v))).append('"');
            }
            first = false;
        }
        sb.append('}');
    }

    static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    static String truncate(String s, int maxLen) {
        if (s == null) return null;
        return s.length() <= maxLen ? s : s.substring(0, maxLen) + "...";
    }

    private static String runGit(String... args) {
        try {
            String[] cmd = new String[args.length + 1];
            cmd[0] = "git";
            System.arraycopy(args, 0, cmd, 1, args.length);
            Process p = new ProcessBuilder(cmd)
                    .redirectErrorStream(true)
                    .start();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = r.readLine();
                p.waitFor();
                return line != null ? line.trim() : "unknown";
            }
        } catch (Exception e) {
            return "unknown";
        }
    }
}
