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
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Applies benchmark configurations to the ND4J environment and SameDiff models.
 *
 * Extracts the config application, model reset, and compilation logic that was
 * previously embedded in test classes. This allows any model benchmark to reuse
 * the same configuration machinery.
 */
@Slf4j
public class BenchmarkConfigApplier {

    /**
     * Reset all cached state in a model between benchmark configurations.
     * NOTE: Does NOT invalidate Triton disk cache - that persists across runs.
     * Only clears DSP plan cache which must be recompiled for each config.
     */
    public static void resetModelState(SameDiff model) {
        model.resetSession();
        model.clearPlaceholderOverrides();
        model.clearPlaceholders(true);
        model.clearOpInputs();
        model.clearDynamicShapePlanCache();

        // NOTE: We do NOT call invalidateTritonCache() here because:
        // 1. Triton PTX kernels are cached to disk (~/.nd4j/triton_cache/)
        // 2. CUDA graphs are cached in-memory (CudaGraphScheduler._graphCache)
        // 3. Both should persist across benchmark configs for the same model
        // 4. invalidateTritonCache() would force re-compilation, defeating the cache
        
        // Only reset Triton counters for clean metrics
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        if (nativeOps.isTritonAvailable()) {
            nativeOps.resetTritonCounters();
        }
    }

    /**
     * Apply all environment flags from a benchmark config.
     */
    public static void apply(BenchmarkConfig config) {
        Environment env = Nd4j.getEnvironment();

        // Reset ALL Triton flags to defaults
        env.setDspBatchZero(false);
        env.setDspBatchZeroKernel(false);
        env.setDspBatchedGemm(false);
        env.setDspCastSinkMatmul(false);
        env.setTritonGraphCapture(false);
        env.setTritonSectionFusion(false);
        env.setTritonConsolidatedArgTable(false);
        env.setTritonArgDirtyTracking(false);
        env.setTritonSkipKernels(false);
        env.setTritonVerifyKernels(false);
        env.setTritonVerifyKeepNative(false);
        env.setTritonVerifyFullSnapshot(false);
        env.setTritonForceRecapture(false);
        env.setTritonIncludeTypes("");
        env.setTritonCaptureMinExec(1);
        env.setTritonCompileAll(false);
        env.setTritonExcludeOps("");
        env.setTritonCooperativeLaunch(false);
        env.setTritonVerbose(false);
        env.setTritonDumpSections(false);
        env.setTritonDumpArgs(false);
        env.setTritonDumpGraphDot(false);
        env.setTritonAllowFallbackCapture(false);

        // Apply the named profile first, then let explicit config fields override it.
        // This keeps profile defaults useful without silently discarding per-config tuning.
        applyTritonProfile(config.getTritonProfile());

        if (config.isTriton()) {
            env.setTritonIncludeTypes(config.getTritonIncludeTypes());
            env.setTritonSectionFusion(config.isTritonSectionFusion());
            env.setTritonGraphCapture(config.isTritonGraphCapture());
            env.setTritonConsolidatedArgTable(config.isTritonConsolidatedArgTable());
            env.setTritonArgDirtyTracking(config.isTritonArgDirtyTracking());
            env.setTritonCaptureMinExec(config.getCaptureMinExec());
            env.setTritonCooperativeLaunch(config.isTritonCooperativeLaunch());
            env.setTritonCompileAll(config.isTritonCompileAll());
            env.setTritonExcludeOps(config.getTritonExcludeOps());
            env.setTritonSkipKernels(config.isTritonSkipKernels());
            env.setTritonVerifyKernels(config.isTritonVerifyKernels());
            env.setTritonVerifyFullSnapshot(config.isTritonVerifyFullSnapshot());
            env.setTritonForceRecapture(config.isTritonForceRecapture());
            env.setTritonVerbose(config.isTritonVerbose());
            env.setTritonDumpSections(config.isTritonDumpSections());
            env.setTritonAllowFallbackCapture(config.isTritonAllowFallbackCapture());
            env.setTritonEnableFpFusion(config.isTritonEnableFpFusion());
            if (config.getTritonCoopTargetBlocks() > 0) env.setTritonCoopTargetBlocks(config.getTritonCoopTargetBlocks());
            if (config.getTritonNumWarps() > 0) env.setTritonNumWarps(config.getTritonNumWarps());
            if (config.getTritonNumStages() > 0) env.setTritonNumStages(config.getTritonNumStages());
            if (config.getTritonNumCTAs() > 0) env.setTritonNumCTAs(config.getTritonNumCTAs());
            if (config.getTritonMaxNreg() > 0) env.setTritonMaxNreg(config.getTritonMaxNreg());
            if (config.getTritonAttentionBlockN() > 0) env.setTritonAttentionBlockN(config.getTritonAttentionBlockN());
            if (config.getTritonMaxSubsegmentOps() > 0) env.setTritonMaxSubsegmentOps(config.getTritonMaxSubsegmentOps());
            if (config.getTritonMaxSubsegmentSections() > 0) env.setTritonMaxSubsegmentSections(config.getTritonMaxSubsegmentSections());
            if (config.getTritonBuildThreads() > 0) env.setTritonBuildThreads(config.getTritonBuildThreads());
            
            // Triton segment fusion optimization flags
            env.setTritonFuseIdentityShapes(config.isTritonFuseIdentityShapes());
            env.setTritonFuseCastChains(config.isTritonFuseCastChains());
            env.setTritonSpecializePermuteSeq1(config.isTritonSpecializePermuteSeq1());
            env.setTritonFuseAttentionNeighborhoods(config.isTritonFuseAttentionNeighborhoods());
            env.setTritonFusedMatmul(config.isTritonFusedMatmul());
            env.setTritonFusionScoring(config.isTritonFusionScoring());
            env.setTritonFusionMinScore(config.getTritonFusionMinScore());
        }

        // cuBLAS TF32: enables tensor cores for FP32 GEMMs on sm_80+ (Ampere+)
        env.setCublasTf32Enabled(config.isCublasTf32());

        // Triton TF32: enables tensor cores for Triton-compiled DotOps on sm_80+
        env.setTritonTf32Enabled(config.isTritonTf32());

        // DSP optimization flags
        env.setDspCastElimination(config.isDspCastElimination());
        env.setDspCastSinkMatmul(config.isDspCastSinkMatmul());
        env.setDspFp16Compute(config.isDspFp16Compute());

        // DSP batch optimizations
        env.setDspBatchZero(config.isDspBatchZero());
        env.setDspBatchZeroKernel(config.isDspBatchZeroKernel());
        env.setDspBatchedGemm(config.isDspBatchedGemm());

        // Enable DSP diagnostics for graph capture configs.
        // EXECUTE is NOT auto-enabled — it triggers per-step cudaStreamSynchronize
        // for argmax logging (~5ms penalty per decode step). Enable explicitly via
        // -Dnd4j.dsp.diagnostics=EXECUTE when needed for debugging.
        if (config.isTritonGraphCapture() || config.getExecutionMode() == GraphExecutionMode.CUDA_GRAPHS) {
            DspDiagnostics.enableCategories(
                    DspDiagnostics.COMPILE | DspDiagnostics.FALLBACK |
                    DspDiagnostics.BACKEND | DspDiagnostics.MEMORY);
            DspDiagnostics.setLevel(DspDiagnostics.LEVEL_FULL);
        }

        log.info("  Config applied: mode={} types='{}' fusion={} gc={} argTable={} dirty={} " +
                        "coopLaunch={} compileAll={} skipK={} verifyK={} fpFusion={} " +
                        "warps={} stages={} ctas={} maxNreg={} maxSubOps={} maxSubSections={} " +
                        "profile={} minExec={}",
                config.getExecutionMode(), config.getTritonIncludeTypes(), config.isTritonSectionFusion(),
                config.isTritonGraphCapture(), config.isTritonConsolidatedArgTable(),
                config.isTritonArgDirtyTracking(), config.isTritonCooperativeLaunch(),
                config.isTritonCompileAll(), config.isTritonSkipKernels(), config.isTritonVerifyKernels(),
                env.tritonEnableFpFusion(), env.tritonNumWarps(), env.tritonNumStages(),
                env.tritonNumCTAs(), env.tritonMaxNreg(), env.tritonMaxSubsegmentOps(),
                env.tritonMaxSubsegmentSections(), config.getTritonProfile(), config.getCaptureMinExec());

        // Verify critical flags were applied correctly
        if (config.isTriton()) {
            verify(config.getTritonIncludeTypes().equals(env.tritonIncludeTypes()),
                    "tritonIncludeTypes not applied");
            verify(config.isTritonSectionFusion() == env.tritonSectionFusion(),
                    "tritonSectionFusion not applied");
            verify(config.isTritonGraphCapture() == env.tritonGraphCapture(),
                    "tritonGraphCapture not applied");
            verify(config.isTritonConsolidatedArgTable() == env.tritonConsolidatedArgTable(),
                    "tritonConsolidatedArgTable not applied");
            verify(config.isTritonArgDirtyTracking() == env.tritonArgDirtyTracking(),
                    "tritonArgDirtyTracking not applied");
            verify(config.getCaptureMinExec() == env.tritonCaptureMinExec(),
                    "tritonCaptureMinExec not applied");
            verify(config.isTritonCompileAll() == env.tritonCompileAll(),
                    "tritonCompileAll not applied");
            verify(config.isTritonSkipKernels() == env.tritonSkipKernels(),
                    "tritonSkipKernels not applied");
            verify(config.isTritonVerifyKernels() == env.tritonVerifyKernels(),
                    "tritonVerifyKernels not applied");
            if (config.getTritonCoopTargetBlocks() > 0) {
                verify(config.getTritonCoopTargetBlocks() == env.tritonCoopTargetBlocks(),
                        "tritonCoopTargetBlocks not applied");
            }
            if (config.getTritonNumWarps() > 0) {
                verify(config.getTritonNumWarps() == env.tritonNumWarps(),
                        "tritonNumWarps not applied");
            }
            if (config.getTritonNumStages() > 0) {
                verify(config.getTritonNumStages() == env.tritonNumStages(),
                        "tritonNumStages not applied");
            }
            if (config.getTritonNumCTAs() > 0) {
                verify(config.getTritonNumCTAs() == env.tritonNumCTAs(),
                        "tritonNumCTAs not applied");
            }
            if (config.getTritonMaxNreg() > 0) {
                verify(config.getTritonMaxNreg() == env.tritonMaxNreg(),
                        "tritonMaxNreg not applied");
            }
            if (config.getTritonMaxSubsegmentOps() > 0) {
                verify(config.getTritonMaxSubsegmentOps() == env.tritonMaxSubsegmentOps(),
                        "tritonMaxSubsegmentOps not applied");
            }
            if (config.getTritonMaxSubsegmentSections() > 0) {
                verify(config.getTritonMaxSubsegmentSections() == env.tritonMaxSubsegmentSections(),
                        "tritonMaxSubsegmentSections not applied");
            }
            if (config.getTritonBuildThreads() > 0) {
                verify(config.getTritonBuildThreads() == env.tritonBuildThreads(),
                        "tritonBuildThreads not applied");
            }
            // Triton segment fusion flags verification
            verify(config.isTritonFuseIdentityShapes() == env.tritonFuseIdentityShapes(),
                    "tritonFuseIdentityShapes not applied");
            verify(config.isTritonFuseCastChains() == env.tritonFuseCastChains(),
                    "tritonFuseCastChains not applied");
            verify(config.isTritonSpecializePermuteSeq1() == env.tritonSpecializePermuteSeq1(),
                    "tritonSpecializePermuteSeq1 not applied");
            verify(config.isTritonFuseAttentionNeighborhoods() == env.tritonFuseAttentionNeighborhoods(),
                    "tritonFuseAttentionNeighborhoods not applied");
            verify(config.isTritonFusedMatmul() == env.tritonFusedMatmul(),
                    "tritonFusedMatmul not applied");
            verify(config.isTritonFusionScoring() == env.tritonFusionScoring(),
                    "tritonFusionScoring not applied");
            verify(config.getTritonFusionMinScore() == env.tritonFusionMinScore(),
                    "tritonFusionMinScore not applied");
        } else {
            verify("".equals(env.tritonIncludeTypes()),
                    "tritonIncludeTypes should be empty for non-Triton config");
            verify(!env.tritonGraphCapture(),
                    "tritonGraphCapture should be false for non-Triton config");
        }
    }

    /**
     * Apply a named Triton compiler profile.
     */
    public static void applyTritonProfile(String profile) {
        Environment env = Nd4j.getEnvironment();
        env.setTritonCacheEnabled(true);
        env.setTritonAlwaysCompile(false);
        env.setTritonDisableLineInfo(true);

        switch (profile) {
            case "DEBUG_FAST":
                env.setTritonBuildThreads(1);
                env.setTritonMaxSubsegmentOps(8);
                env.setTritonMaxSubsegmentSections(2);
                env.setTritonNumWarps(2);
                env.setTritonNumStages(2);
                env.setTritonNumCTAs(1);
                env.setTritonMaxNreg(64);
                env.setTritonEnableFpFusion(false);
                env.setTritonVerbose(true);
                env.setTritonDumpSections(true);
                env.setTritonDumpArgs(false);
                env.setTritonKernelDump(false);
                env.setTritonLogAllPatterns(false);
                env.setTritonCoopTargetBlocks(1);
                break;
            case "BALANCED":
                env.setTritonBuildThreads(4);
                env.setTritonMaxSubsegmentOps(0);
                env.setTritonMaxSubsegmentSections(0);
                env.setTritonNumWarps(8);
                env.setTritonNumStages(2);
                env.setTritonNumCTAs(1);
                env.setTritonMaxNreg(0);
                env.setTritonEnableFpFusion(true);
                env.setTritonVerbose(true);
                env.setTritonDumpSections(false);
                env.setTritonDumpArgs(false);
                env.setTritonKernelDump(false);
                env.setTritonLogAllPatterns(false);
                env.setTritonCoopTargetBlocks(0);
                env.setTritonCooperativeLaunch(false);
                break;
            default: // MAX_PERF / MAX_AUTOTUNE
                env.setTritonBuildThreads(4);
                env.setTritonMaxSubsegmentOps(0);
                env.setTritonMaxSubsegmentSections(0);
                env.setTritonNumWarps(8);
                env.setTritonNumStages(2);
                env.setTritonNumCTAs(1);
                env.setTritonMaxNreg(0);
                env.setTritonEnableFpFusion(true);
                env.setTritonVerbose(false);
                env.setTritonDumpSections(false);
                env.setTritonDumpArgs(false);
                env.setTritonKernelDump(false);
                env.setTritonLogAllPatterns(false);
                env.setTritonCoopTargetBlocks(0);
                env.setTritonCooperativeLaunch(false);
                break;
        }
    }

    /**
     * Compile a model for the given benchmark config.
     *
     * For Triton configs, uses MAX_AUTOTUNE compilation mode.
     * For non-Triton configs, compiles with the specified execution mode.
     *
     * @param model   the SameDiff model to compile
     * @param label   human-readable label for logging
     * @param outputs the output variable names
     * @param config  the benchmark config
     * @return the effective execution mode after compilation
     */
    public static GraphExecutionMode compileModel(SameDiff model, String label,
                                                   List<String> outputs, BenchmarkConfig config) {
        if (config.isTriton()) {
            return compileTritonModel(model, label, outputs);
        } else {
            if (config.getExecutionMode() == null) {
                throw new IllegalStateException(config.getName() + ": non-Triton config must have executionMode set");
            }
            return model.compileNativeDynamicShapePlan(outputs, config.getExecutionMode(), true);
        }
    }

    /**
     * Compile a pair of models (e.g., decoder + embed_tokens) for the given config.
     */
    public static void compileModels(SameDiff model1, String label1,
                                     SameDiff model2, String label2,
                                     BenchmarkConfig config) {
        List<String> outputs1 = model1.outputs() == null
                ? Collections.emptyList() : new ArrayList<>(model1.outputs());
        List<String> outputs2 = model2.outputs() == null
                ? Collections.emptyList() : new ArrayList<>(model2.outputs());

        verify(!outputs1.isEmpty(), config.getName() + ": " + label1 + " has no configured outputs");
        verify(!outputs2.isEmpty(), config.getName() + ": " + label2 + " has no configured outputs");

        long start = System.currentTimeMillis();
        if (config.isTriton()) {
            log.info("  Compiling with Triton MAX_AUTOTUNE (types={})", config.getTritonIncludeTypes());
            GraphExecutionMode mode1 = compileTritonModel(model1, label1, outputs1);
            GraphExecutionMode mode2 = compileTritonModel(model2, label2, outputs2);
            verify(mode1 == GraphExecutionMode.TRITON,
                    config.getName() + ": " + label1 + " did not compile to TRITON mode");
            verify(mode2 == GraphExecutionMode.TRITON,
                    config.getName() + ": " + label2 + " did not compile to TRITON mode");
        } else {
            if (config.getExecutionMode() == null) {
                throw new IllegalStateException(config.getName() + ": non-Triton config must have executionMode set");
            }
            log.info("  Compiling with mode {}", config.getExecutionMode());
            GraphExecutionMode mode1 = model1.compileNativeDynamicShapePlan(
                    outputs1, config.getExecutionMode(), true);
            GraphExecutionMode mode2 = model2.compileNativeDynamicShapePlan(
                    outputs2, config.getExecutionMode(), true);
            verify(mode1 != null, config.getName() + ": " + label1 + " compile returned null mode");
            verify(mode2 != null, config.getName() + ": " + label2 + " compile returned null mode");
            log.info("  Compiled: {}={} {}={}", label1, mode1, label2, mode2);
        }
        log.info("  Compile done [{}ms]", System.currentTimeMillis() - start);
    }

    private static void verify(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    private static GraphExecutionMode compileTritonModel(SameDiff model, String label, List<String> outputs) {
        model.setDspAutoCompileEnabled(false);
        model.setDspNativeAutoCompileEnabled(false);
        model.setDspFallbackToAutoIfTritonUnavailable(false);

        GraphExecutionMode effectiveMode = outputs.isEmpty()
                ? model.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE)
                : model.compileNativeDynamicShapePlan(outputs, DspCompilationMode.MAX_AUTOTUNE);
        if (effectiveMode != GraphExecutionMode.TRITON) {
            throw new IllegalStateException("MAX_AUTOTUNE for " + label +
                    " resolved to " + effectiveMode + " instead of TRITON");
        }
        return effectiveMode;
    }
}
