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

package org.eclipse.deeplearning4j.model.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.bytedeco.javacpp.Pointer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Orchestrates benchmark runs: reset -> configure -> compile -> execute -> validate -> report.
 *
 * Provides the main loop that test classes use. Tests supply:
 * - A list of models to reset/compile
 * - A decode function (model-specific generation logic)
 * - A validate function (model-specific quality checks)
 */
@Slf4j
public class BenchmarkRunner {
    private static final String OP_TIMING_ENABLED_PROPERTY = "vlm.benchmark.opTiming";
    private static final String OP_TIMING_DETAILED_PROPERTY = "vlm.benchmark.opTimingDetailed";
    private static final String OP_TIMING_TOP_N_PROPERTY = "vlm.benchmark.opTimingTopN";
    private static final String OP_TIMING_CSV_DIR_PROPERTY = "vlm.benchmark.opTimingCsvDir";
    private static final String OP_TIMING_BREAKDOWN_OPS_PROPERTY = "vlm.benchmark.opTimingBreakdownOps";
    private static final String OP_TIMING_HISTOGRAM_OPS_PROPERTY = "vlm.benchmark.opTimingHistogramOps";

    private static class PlanMetrics {
        private final String compactSummary;
        private final String detailLog;
        private final String primaryLabel;
        private final int primaryCapturedSegments;
        private final int primaryTotalSegments;
        private final int primaryHostOnlyOps;
        private final Boolean primaryCaptureValid;

        private PlanMetrics(String compactSummary, String detailLog, String primaryLabel,
                            int primaryCapturedSegments, int primaryTotalSegments,
                            int primaryHostOnlyOps, Boolean primaryCaptureValid) {
            this.compactSummary = compactSummary;
            this.detailLog = detailLog;
            this.primaryLabel = primaryLabel;
            this.primaryCapturedSegments = primaryCapturedSegments;
            this.primaryTotalSegments = primaryTotalSegments;
            this.primaryHostOnlyOps = primaryHostOnlyOps;
            this.primaryCaptureValid = primaryCaptureValid;
        }
    }

    /**
     * Function that performs model inference given a config.
     */
    @FunctionalInterface
    public interface DecodeFunction {
        GenerationResult decode(BenchmarkConfig config);
    }

    /**
     * Function that validates generation output.
     */
    @FunctionalInterface
    public interface ValidateFunction {
        void validate(BenchmarkConfig config, GenerationResult result);
    }

    /**
     * Function that resets and recompiles models for a config.
     */
    @FunctionalInterface
    public interface CompileFunction {
        void compile(BenchmarkConfig config);
    }

    /**
     * Run a single benchmark configuration.
     *
     * @param config      the configuration to test
     * @param modelNames  stable names for models, used in diagnostics
     * @param models      models to reset between configs
     * @param compileFn   function to compile models for this config
     * @param decodeFn    function to run inference
     * @param validateFn  function to validate results
     * @return the benchmark result
     */
    public static BenchmarkResult runSingle(BenchmarkConfig config, List<String> modelNames, List<SameDiff> models,
                                             CompileFunction compileFn,
                                             DecodeFunction decodeFn,
                                             ValidateFunction validateFn) {
        BenchmarkResult cr = new BenchmarkResult(config.getName());

        try {
            long t0 = System.currentTimeMillis();

            // 1. Reset state
            for (SameDiff model : models) {
                BenchmarkConfigApplier.resetModelState(model);
            }
            Nd4j.getExecutioner().commit();
            cr.setResetMs(System.currentTimeMillis() - t0);

            // 2. Apply environment flags
            BenchmarkConfigApplier.apply(config);

            // 3. Compile
            long compileStart = System.currentTimeMillis();
            compileFn.compile(config);
            cr.setCompileMs(System.currentTimeMillis() - compileStart);

            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

            // Enable DSP execution timing after compile (plan exists post-compile)
            if (config.isDspExecutionTiming()) {
                for (SameDiff model : models) {
                    DynamicShapePlanExecutor executor = model.getOrCreateSession().getDynamicShapePlanExecutor();
                    if (executor != null) {
                        executor.setExecutionTimingEnabled(true);
                        break;
                    }
                }
            }

            // 4. Decode
            beginDecodeOpTiming(nativeOps);
            GenerationResult result;
            long decodeStart = System.currentTimeMillis();
            try {
                result = decodeFn.decode(config);
            } finally {
                finishDecodeOpTiming(config, nativeOps);
            }
            cr.setDecodeMs(System.currentTimeMillis() - decodeStart);

            // Record decode metrics
            if (result == null) {
                throw new IllegalStateException(config.getName() + ": GenerationResult is null");
            }
            if (result.getText() == null) {
                throw new IllegalStateException(config.getName() + ": generated text is null");
            }
            cr.setTokenCount(result.getGeneratedTokenCount());
            cr.setTokPerSec(result.getTokensPerSecond());
            cr.setDecodeTokPerSec(result.getDecodeTokensPerSecond());
            cr.setSteadyTokPerSec(result.getSteadyStateTokensPerSecond());
            cr.setLateSteadyTokPerSec(result.getLateSteadyStateTokensPerSecond());
            cr.setFirstTokenMs(result.getFirstTokenLatencyMs());
            cr.setFinishReason(result.getFinishReason());
            cr.setGeneratedText(result.getText());
            cr.setWarmupStepCount(result.getWarmupStepCount());
            cr.setMaxWarmupStepMs(result.getMaxWarmupStepMs());
            cr.setTotalWarmupMs(result.getTotalWarmupMs());

            // Record Triton counters
            if (nativeOps.isTritonAvailable()) {
                cr.setTritonLaunches(nativeOps.getTritonKernelLaunchCount());
                cr.setTritonCacheHits(nativeOps.getTritonCacheHitCount());
            }

            PlanMetrics planMetrics = collectPlanMetrics(config, modelNames, models, nativeOps);
            cr.setPlanMetricsSummary(planMetrics.compactSummary);
            if (!planMetrics.detailLog.isEmpty()) {
                log.info("  DSP plan diagnostics:\n{}", planMetrics.detailLog);
            }
            assertPrimaryCaptureIfRequired(config, planMetrics);

            log.info("  {} -> {} tokens, {} (firstToken={}ms), finish={}, text: '{}'",
                    config.getName(), cr.getTokenCount(),
                    formatThroughput(cr),
                    cr.getFirstTokenMs(), cr.getFinishReason(),
                    result.getText().substring(0, Math.min(100, result.getText().length())));

            // 5. Validate
            long validateStart = System.currentTimeMillis();
            validateFn.validate(config, result);
            cr.setValidateMs(System.currentTimeMillis() - validateStart);

            cr.setPassed(true);
        } catch (Exception | AssertionError e) {
            cr.setPassed(false);
            cr.setFailureMessage(e.getClass().getSimpleName() + ": " + e.getMessage());
            log.error("Config {} failed", config.getName(), e);
        }

        return cr;
    }

    /**
     * Run a matrix of benchmark configurations.
     *
     * @param configs        configurations to test
     * @param modelNames     stable names for models, used in diagnostics
     * @param models         models to reset between configs
     * @param compileFn      function to compile models
     * @param decodeFn       function to run inference
     * @param validateFn     function to validate results
     * @param filterProperty system property name for filtering configs (e.g., "vlm.config")
     * @return list of results, one per config
     */
    public static List<BenchmarkResult> runMatrix(List<BenchmarkConfig> configs,
                                                   List<String> modelNames,
                                                   List<SameDiff> models,
                                                   CompileFunction compileFn,
                                                   DecodeFunction decodeFn,
                                                   ValidateFunction validateFn,
                                                   String filterProperty) {
        // Apply filter
        List<BenchmarkConfig> filtered = filterConfigs(configs, System.getProperty(filterProperty));

        log.info("Running {} configurations:", filtered.size());
        for (int i = 0; i < filtered.size(); i++) {
            log.info("  [{}] {}", i + 1, filtered.get(i));
        }

        List<BenchmarkResult> results = new ArrayList<>();
        int failures = 0;

        for (int i = 0; i < filtered.size(); i++) {
            BenchmarkConfig config = filtered.get(i);
            log.info("============================================================");
            log.info("[{}/{}] CONFIG: {}", i + 1, filtered.size(), config);
            log.info("============================================================");

            BenchmarkResult cr = runSingle(config, modelNames, models, compileFn, decodeFn, validateFn);
            results.add(cr);
            if (!cr.isPassed()) failures++;
            log.info("  {}", cr.summary());
        }

        return results;
    }

    /**
     * Print a summary report of benchmark results.
     */
    public static void printReport(List<BenchmarkResult> results) {
        int failures = (int) results.stream().filter(r -> !r.isPassed()).count();

        log.info("============================================================");
        log.info("CONFIGURATION MATRIX RESULTS: {}/{} passed", results.size() - failures, results.size());
        log.info("============================================================");

        StringBuilder report = new StringBuilder();
        for (BenchmarkResult cr : results) {
            report.append(cr.summary()).append("\n");
        }
        log.info("Full report:\n{}", report);

        // Summary statistics
        double maxTokPerSec = 0, minTokPerSec = Double.MAX_VALUE;
        long maxCompileMs = 0;
        String fastestConfig = "", slowestConfig = "";
        for (BenchmarkResult cr : results) {
            if (!cr.isPassed()) continue;
            double effectiveTokPerSec = effectiveTokPerSec(cr);
            if (effectiveTokPerSec > maxTokPerSec) { maxTokPerSec = effectiveTokPerSec; fastestConfig = cr.getConfigName(); }
            if (effectiveTokPerSec < minTokPerSec) { minTokPerSec = effectiveTokPerSec; slowestConfig = cr.getConfigName(); }
            if (cr.getCompileMs() > maxCompileMs) maxCompileMs = cr.getCompileMs();
        }
        if (maxTokPerSec > 0) {
            log.info("Performance range (late steady-state when available): {} tok/s ({}) to {} tok/s ({}), max compile={}ms",
                    String.format("%.2f", minTokPerSec), slowestConfig,
                    String.format("%.2f", maxTokPerSec), fastestConfig, maxCompileMs);
        }

        if (failures > 0) {
            throw new AssertionError(failures + " config(s) failed:\n" + report);
        }
    }

    /**
     * Filter configs by a pattern string (comma-separated names, supports * prefix matching).
     */
    public static List<BenchmarkConfig> filterConfigs(List<BenchmarkConfig> configs, String pattern) {
        if (pattern == null || pattern.isEmpty()) return configs;

        String[] filters = pattern.split(",");
        List<BenchmarkConfig> filtered = configs.stream().filter(c -> {
            for (String f : filters) {
                String ft = f.trim();
                if (ft.endsWith("*")) {
                    if (c.getName().startsWith(ft.substring(0, ft.length() - 1))) return true;
                } else {
                    if (c.getName().equals(ft)) return true;
                }
            }
            return false;
        }).collect(Collectors.toList());

        log.info("Config filter '{}' matched {} configs", pattern, filtered.size());
        if (filtered.isEmpty()) {
            throw new IllegalArgumentException("No configs matched filter: " + pattern);
        }
        return filtered;
    }

    private static PlanMetrics collectPlanMetrics(BenchmarkConfig config, List<String> modelNames,
                                                  List<SameDiff> models, NativeOps nativeOps) {
        String primaryCompact = "";
        StringBuilder details = new StringBuilder();
        String primaryLabel = "";
        int primaryCapturedSegments = -1;
        int primaryTotalSegments = -1;
        int primaryHostOnlyOps = -1;
        Boolean primaryCaptureValid = null;

        for (int i = 0; i < models.size(); i++) {
            SameDiff model = models.get(i);
            String label = resolveModelLabel(modelNames, i);

            InferenceSession session = model.getOrCreateSession();
            if (session == null) continue;

            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            if (executor == null) continue;

            Pointer handle = executor.getNativePlanHandle();
            if (handle == null || handle.isNull()) continue;

            int totalSegments = nativeOps.getPlanNumSegments(handle);
            if (totalSegments < 0) continue;

            int capturedSegments = nativeOps.getPlanNumCapturedGraphSegments(handle);
            int totalReplays = nativeOps.getPlanTotalGraphReplays(handle);
            int hostOnlyOps = nativeOps.getPlanNumHostOnlyOps(handle);
            String hostOnlyNames = safeTrim(nativeOps.getPlanHostOnlyOpNames(handle));
            String captureStats = sanitizeCaptureStats(nativeOps.getPlanCaptureStats(handle));

            boolean captureExpected = expectsGraphCapture(config);
            Boolean captureValid = null;
            if (captureExpected && capturedSegments > 0) {
                captureValid = nativeOps.validatePlanCapturedGraph(handle);
            }

            StringBuilder compactEntry = new StringBuilder()
                    .append(label)
                    .append("{")
                    .append(capturedSegments).append("/").append(totalSegments).append(" cap,")
                    .append(totalReplays).append(" replay,")
                    .append(hostOnlyOps).append(" host");
            if (captureValid != null) {
                compactEntry.append(",").append(captureValid ? "valid" : "INVALID");
            }
            if (!captureStats.isEmpty()) {
                compactEntry.append(",").append(captureStats);
            }
            compactEntry.append("}");
            if (i == 0) {
                primaryCompact = compactEntry.toString();
                primaryLabel = label;
                primaryCapturedSegments = capturedSegments;
                primaryTotalSegments = totalSegments;
                primaryHostOnlyOps = hostOnlyOps;
                primaryCaptureValid = captureValid;
            }

            details.append("  ")
                    .append(label)
                    .append(": segments=").append(totalSegments)
                    .append(", captured=").append(capturedSegments)
                    .append(", replays=").append(totalReplays)
                    .append(", hostOnly=").append(hostOnlyOps);
            if (captureValid != null) {
                details.append(", captureValid=").append(captureValid);
            }
            if (!captureStats.isEmpty()) {
                details.append(", stats=").append(captureStats);
            }
            details.append("\n");

            if (hostOnlyOps > 0 && !hostOnlyNames.isEmpty()) {
                details.append("    hostOnlyOps: ").append(hostOnlyNames).append("\n");
            }

            String replaySummary = safeTrim(PlanIntrospection.formatReplaySummary(handle));
            if (!replaySummary.isEmpty()) {
                for (String line : replaySummary.split("\\R")) {
                    details.append("    ").append(line).append("\n");
                }
            }
        }

        return new PlanMetrics(primaryCompact, trimTrailingNewline(details), primaryLabel,
                primaryCapturedSegments, primaryTotalSegments, primaryHostOnlyOps, primaryCaptureValid);
    }

    private static boolean expectsGraphCapture(BenchmarkConfig config) {
        // When shape freezing is disabled via system property, graph capture cannot happen
        // regardless of config flags (shapes must be frozen for CUDA graph capture)
        boolean freezeDisabled = "true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.DSP_NO_FREEZE));
        boolean directDisabled = "true".equalsIgnoreCase(System.getProperty(ND4JSystemProperties.DSP_NO_DIRECT));
        if (freezeDisabled || directDisabled) {
            return false;
        }
        return config.isTritonGraphCapture() || config.getExecutionMode() == GraphExecutionMode.CUDA_GRAPHS;
    }

    private static String resolveModelLabel(List<String> modelNames, int index) {
        if (modelNames != null && index < modelNames.size()) {
            String label = modelNames.get(index);
            if (label != null && !label.isEmpty()) {
                return label;
            }
        }
        return "model" + index;
    }

    private static String sanitizeCaptureStats(String captureStats) {
        String trimmed = safeTrim(captureStats);
        return trimmed.replace('|', ',');
    }

    private static String safeTrim(String value) {
        return value == null ? "" : value.trim();
    }

    private static String trimTrailingNewline(StringBuilder sb) {
        int length = sb.length();
        if (length > 0 && sb.charAt(length - 1) == '\n') {
            sb.setLength(length - 1);
        }
        return sb.toString();
    }

    private static void assertPrimaryCaptureIfRequired(BenchmarkConfig config, PlanMetrics planMetrics) {
        if (!expectsGraphCapture(config)) {
            return;
        }
        if (planMetrics.primaryTotalSegments <= 0) {
            throw new IllegalStateException("Expected graph capture on primary model but no primary plan segments were found");
        }
        if (planMetrics.primaryCapturedSegments <= 0) {
            throw new IllegalStateException(String.format(
                    "Expected graph capture on primary model '%s' but capturedSegments=%d/%d",
                    planMetrics.primaryLabel, planMetrics.primaryCapturedSegments, planMetrics.primaryTotalSegments));
        }
        if (planMetrics.primaryHostOnlyOps > 0) {
            throw new IllegalStateException(String.format(
                    "Primary model '%s' has %d host-only ops in captured graph",
                    planMetrics.primaryLabel, planMetrics.primaryHostOnlyOps));
        }
        if (planMetrics.primaryCaptureValid != null && !planMetrics.primaryCaptureValid) {
            throw new IllegalStateException(String.format(
                    "Captured graph validation failed for primary model '%s'",
                    planMetrics.primaryLabel));
        }
    }

    private static double effectiveTokPerSec(BenchmarkResult result) {
        if (result.getLateSteadyTokPerSec() > 0) {
            return result.getLateSteadyTokPerSec();
        }
        if (result.getSteadyTokPerSec() > 0) {
            return result.getSteadyTokPerSec();
        }
        if (result.getDecodeTokPerSec() > 0) {
            return result.getDecodeTokPerSec();
        }
        return result.getTokPerSec();
    }

    private static String formatThroughput(BenchmarkResult result) {
        StringBuilder sb = new StringBuilder(String.format("overall=%.2f tok/s", result.getTokPerSec()));
        if (result.getDecodeTokPerSec() > 0) {
            sb.append(String.format(", decode=%.2f tok/s", result.getDecodeTokPerSec()));
        }
        if (result.getSteadyTokPerSec() > 0) {
            sb.append(String.format(", steady=%.2f tok/s", result.getSteadyTokPerSec()));
        }
        if (result.getLateSteadyTokPerSec() > 0) {
            sb.append(String.format(", lateSteady=%.2f tok/s", result.getLateSteadyTokPerSec()));
        }
        return sb.toString();
    }

    private static void beginDecodeOpTiming(NativeOps nativeOps) {
        if (!isDecodeOpTimingEnabled()) {
            return;
        }
        nativeOps.setOpTimingEnabled(0, 0);
        nativeOps.resetOpTiming();
        nativeOps.setOpTimingEnabled(1, isDecodeOpTimingDetailed() ? 1 : 0);
    }

    private static void finishDecodeOpTiming(BenchmarkConfig config, NativeOps nativeOps) {
        if (!isDecodeOpTimingEnabled()) {
            return;
        }

        try {
            nativeOps.flushOpTiming();

            int topN = Integer.getInteger(OP_TIMING_TOP_N_PROPERTY, 12);
            if (topN > 0) {
                log.info("  Decode op timing hotspots for {}", config.getName());
                nativeOps.printOpTimingStats(topN);
            }

            for (String opName : getOpTimingListProperty(OP_TIMING_BREAKDOWN_OPS_PROPERTY)) {
                log.info("  Decode op timing breakdown for {} [{}]", config.getName(), opName);
                nativeOps.printOpTimingBreakdown(opName);
            }

            for (String opName : getOpTimingListProperty(OP_TIMING_HISTOGRAM_OPS_PROPERTY)) {
                log.info("  Decode op timing histogram for {} [{}]", config.getName(), opName);
                nativeOps.printOpTimingHistogram(opName);
            }

            String csvDir = System.getProperty(OP_TIMING_CSV_DIR_PROPERTY);
            if (csvDir != null && !csvDir.isBlank()) {
                Path outputDir = Paths.get(csvDir);
                Files.createDirectories(outputDir);
                Path csvPath = outputDir.resolve(sanitizeFileName(config.getName()) + ".csv");
                int exported = nativeOps.exportOpTimingCSV(csvPath.toString());
                if (exported != 0) {
                    log.info("  Decode op timing CSV: {}", csvPath);
                } else {
                    log.warn("  Decode op timing CSV export failed: {}", csvPath);
                }
            }
        } catch (IOException e) {
            log.warn("  Failed to write decode op timing CSV for {}", config.getName(), e);
        } finally {
            nativeOps.setOpTimingEnabled(0, 0);
            nativeOps.resetOpTiming();
        }
    }

    private static boolean isDecodeOpTimingEnabled() {
        return Boolean.parseBoolean(System.getProperty(OP_TIMING_ENABLED_PROPERTY, "false"));
    }

    private static boolean isDecodeOpTimingDetailed() {
        return Boolean.parseBoolean(System.getProperty(OP_TIMING_DETAILED_PROPERTY, "false"));
    }

    private static List<String> getOpTimingListProperty(String propertyName) {
        String value = System.getProperty(propertyName);
        if (value == null || value.isBlank()) {
            return List.of();
        }
        return List.of(value.split(",")).stream()
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
    }

    private static String sanitizeFileName(String value) {
        return value.replaceAll("[^A-Za-z0-9._-]+", "_");
    }
}
