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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

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
     * @param models      models to reset between configs
     * @param compileFn   function to compile models for this config
     * @param decodeFn    function to run inference
     * @param validateFn  function to validate results
     * @return the benchmark result
     */
    public static BenchmarkResult runSingle(BenchmarkConfig config, List<SameDiff> models,
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

            // 4. Decode
            long decodeStart = System.currentTimeMillis();
            GenerationResult result = decodeFn.decode(config);
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
            cr.setFirstTokenMs(result.getFirstTokenLatencyMs());
            cr.setFinishReason(result.getFinishReason());
            cr.setGeneratedText(result.getText());

            // Record Triton counters
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            if (nativeOps.isTritonAvailable()) {
                cr.setTritonLaunches(nativeOps.getTritonKernelLaunchCount());
                cr.setTritonCacheHits(nativeOps.getTritonCacheHitCount());
            }

            log.info("  {} -> {} tokens, {} tok/s (firstToken={}ms), finish={}, text: '{}'",
                    config.getName(), cr.getTokenCount(),
                    String.format("%.2f", cr.getTokPerSec()),
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
     * @param models         models to reset between configs
     * @param compileFn      function to compile models
     * @param decodeFn       function to run inference
     * @param validateFn     function to validate results
     * @param filterProperty system property name for filtering configs (e.g., "vlm.config")
     * @return list of results, one per config
     */
    public static List<BenchmarkResult> runMatrix(List<BenchmarkConfig> configs,
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

            BenchmarkResult cr = runSingle(config, models, compileFn, decodeFn, validateFn);
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
            if (cr.getTokPerSec() > maxTokPerSec) { maxTokPerSec = cr.getTokPerSec(); fastestConfig = cr.getConfigName(); }
            if (cr.getTokPerSec() < minTokPerSec) { minTokPerSec = cr.getTokPerSec(); slowestConfig = cr.getConfigName(); }
            if (cr.getCompileMs() > maxCompileMs) maxCompileMs = cr.getCompileMs();
        }
        if (maxTokPerSec > 0) {
            log.info("Performance range: {} tok/s ({}) to {} tok/s ({}), max compile={}ms",
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
}
