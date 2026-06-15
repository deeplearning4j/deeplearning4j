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

import org.eclipse.deeplearning4j.llm.generation.GenerationResult;

/**
 * Result of running a single benchmark configuration.
 *
 * Records timing, throughput, and Triton counters for a configuration run.
 */
public class BenchmarkResult {

    private final String configName;
    private boolean passed;
    private String failureMessage;

    // Timing
    private long resetMs;
    private long compileMs;
    private long decodeMs;
    private long validateMs;

    // Decode metrics
    private int tokenCount;
    private double tokPerSec;
    private double decodeTokPerSec;
    private double steadyTokPerSec;
    private double lateSteadyTokPerSec;
    private long firstTokenMs;
    private GenerationResult.FinishReason finishReason;

    // Triton counters
    private long tritonLaunches;
    private long tritonCacheHits;

    // Warmup classification
    private int warmupStepCount;
    private long maxWarmupStepMs;
    private long totalWarmupMs;

    // DSP / CUDA graph observability
    private String planMetricsSummary;

    // Full result
    private String generatedText;

    public BenchmarkResult(String configName) {
        this.configName = configName;
    }

    public String getConfigName() { return configName; }
    public boolean isPassed() { return passed; }
    public void setPassed(boolean passed) { this.passed = passed; }
    public String getFailureMessage() { return failureMessage; }
    public void setFailureMessage(String failureMessage) { this.failureMessage = failureMessage; }
    public long getResetMs() { return resetMs; }
    public void setResetMs(long resetMs) { this.resetMs = resetMs; }
    public long getCompileMs() { return compileMs; }
    public void setCompileMs(long compileMs) { this.compileMs = compileMs; }
    public long getDecodeMs() { return decodeMs; }
    public void setDecodeMs(long decodeMs) { this.decodeMs = decodeMs; }
    public long getValidateMs() { return validateMs; }
    public void setValidateMs(long validateMs) { this.validateMs = validateMs; }
    public int getTokenCount() { return tokenCount; }
    public void setTokenCount(int tokenCount) { this.tokenCount = tokenCount; }
    public double getTokPerSec() { return tokPerSec; }
    public void setTokPerSec(double tokPerSec) { this.tokPerSec = tokPerSec; }
    public double getDecodeTokPerSec() { return decodeTokPerSec; }
    public void setDecodeTokPerSec(double decodeTokPerSec) { this.decodeTokPerSec = decodeTokPerSec; }
    public double getSteadyTokPerSec() { return steadyTokPerSec; }
    public void setSteadyTokPerSec(double steadyTokPerSec) { this.steadyTokPerSec = steadyTokPerSec; }
    public double getLateSteadyTokPerSec() { return lateSteadyTokPerSec; }
    public void setLateSteadyTokPerSec(double lateSteadyTokPerSec) { this.lateSteadyTokPerSec = lateSteadyTokPerSec; }
    public long getFirstTokenMs() { return firstTokenMs; }
    public void setFirstTokenMs(long firstTokenMs) { this.firstTokenMs = firstTokenMs; }
    public GenerationResult.FinishReason getFinishReason() { return finishReason; }
    public void setFinishReason(GenerationResult.FinishReason finishReason) { this.finishReason = finishReason; }
    public long getTritonLaunches() { return tritonLaunches; }
    public void setTritonLaunches(long tritonLaunches) { this.tritonLaunches = tritonLaunches; }
    public long getTritonCacheHits() { return tritonCacheHits; }
    public void setTritonCacheHits(long tritonCacheHits) { this.tritonCacheHits = tritonCacheHits; }
    public int getWarmupStepCount() { return warmupStepCount; }
    public void setWarmupStepCount(int warmupStepCount) { this.warmupStepCount = warmupStepCount; }
    public long getMaxWarmupStepMs() { return maxWarmupStepMs; }
    public void setMaxWarmupStepMs(long maxWarmupStepMs) { this.maxWarmupStepMs = maxWarmupStepMs; }
    public long getTotalWarmupMs() { return totalWarmupMs; }
    public void setTotalWarmupMs(long totalWarmupMs) { this.totalWarmupMs = totalWarmupMs; }
    public String getPlanMetricsSummary() { return planMetricsSummary; }
    public void setPlanMetricsSummary(String planMetricsSummary) { this.planMetricsSummary = planMetricsSummary; }
    public String getGeneratedText() { return generatedText; }
    public void setGeneratedText(String generatedText) { this.generatedText = generatedText; }

    public String summary() {
        if (!passed) return String.format("[FAIL] %s: %s", configName, failureMessage);
        StringBuilder throughput = new StringBuilder(String.format("overall=%.2f tok/s", tokPerSec));
        if (decodeTokPerSec > 0) {
            throughput.append(String.format(", decode=%.2f tok/s", decodeTokPerSec));
        }
        if (steadyTokPerSec > 0) {
            throughput.append(String.format(", steady=%.2f tok/s", steadyTokPerSec));
        }
        if (lateSteadyTokPerSec > 0) {
            throughput.append(String.format(", lateSteady=%.2f tok/s", lateSteadyTokPerSec));
        }
        String warmupInfo = warmupStepCount > 0
                ? String.format(" | warmup: %d steps, total=%dms, max=%dms",
                    warmupStepCount, totalWarmupMs, maxWarmupStepMs)
                : "";
        String base = String.format("[PASS] %s: %d tokens, %s, firstToken=%dms | reset=%dms compile=%dms decode=%dms validate=%dms | triton: launches=%d hits=%d%s",
                configName, tokenCount, throughput, firstTokenMs,
                resetMs, compileMs, decodeMs, validateMs,
                tritonLaunches, tritonCacheHits, warmupInfo);
        if (planMetricsSummary == null || planMetricsSummary.isEmpty()) {
            return base;
        }
        return base + " | plan: " + planMetricsSummary;
    }
}
