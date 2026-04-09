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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.*;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SmolDocling pipeline test with builder-based configuration matrix.
 *
 * Loads models ONCE, then loops through all meaningful combinations of
 * execution mode, Triton include types, fusion, graph capture, and arg table opts.
 *
 * Uses shared {@link BenchmarkRunner} infrastructure for the reset/configure/compile/decode/validate loop.
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestSmolDoclingOptimizedPipeline#testOptimizedDoclingPipeline \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
public class TestSmolDoclingOptimizedPipeline {

    private static String pdfPath;
    private static int specificPage = -1;
    private static int renderDpi = 150;

    // ─── PipelineContext: shared state loaded once ─────────────────────────

    private static class PipelineContext {
        SameDiff decoder;
        SameDiff embedTokens;
        Tokenizer tokenizer;
        INDArray inputsEmbeds;
        int[] promptTokenIds;
        long hiddenSize;
        // Draft model for speculative decoding (lazily loaded)
        SameDiff draftDecoder;
        int draftDeviceId = -1;
        long draftHiddenSize;
        // Pipeline setup timings
        long downloadMs;
        long importMs;
        long visionMs;
        long embedMs;
        int visionFrames;
        int decoderOps;
        int embedOps;
    }

    // ─── Configuration matrix: performance-focused configs ──────────────────
    //
    // BEST CURRENT DEFAULT: compileAll + COMPILE_ALL_TYPES + ATTENTION + GC + argOpt
    // + dirty tracking + batched GEMM + warps2/stages1.
    // Batch-zero still regresses decode step 2 badly in SmolDocling
    // (see TRITON_compileAll_best_ATTN_gc_argOpt_batchOps diagnostic config below).
    // CUDA_GRAPHS baseline (no Triton) -> 11.40 tok/s (40 tok/s steady)
    // SLOT_BY_SLOT baseline -> 5.62 tok/s
    //
    // NEVER compile MATMUL (cuBLAS 2.8x faster), NEVER include SPLIT/CONCAT without compileAll
    // Flash attention (+ATTENTION) gives +30% decode speed with CUDA graph capture
    // dspCastElimination is neutral with CUDA graphs
    // FP16: use nd4j.optimizer.fp16=true (pre-cast weights at load) NOT dspFp16Compute (runtime double-cast)

    private static final String FULL_TRITON_TYPES =
            "ELEMENTWISE,REDUCTION,NORMALIZATION,GATHER,STACK,ATTENTION";

    // Best-known compileAll types that achieved 100 tok/s decode (batchops-combined-test.log)
    // CRITICAL: Excludes NORMALIZATION/REDUCTION - Triton compilation is SLOWER than native fallback
    // rms_norm falls back to native CUDA kernel which is faster than Triton for these ops
    private static final String COMPILE_ALL_TYPES =
            "CONST_GEN,GATHER,CONCAT,SPLIT,STACK";

    private static final String COMPILE_ALL_TYPES_WITH_NORM =
            COMPILE_ALL_TYPES + ",NORMALIZATION";

    private static final String COMPILE_ALL_TYPES_WITH_NORM_NO_CONCAT =
            "CONST_GEN,GATHER,SPLIT,STACK,NORMALIZATION";

    private static final String COMPILE_ALL_TYPES_WITH_NORM_AND_MATMUL =
            COMPILE_ALL_TYPES_WITH_NORM + ",MATMUL";

    private static List<BenchmarkConfig> getAllConfigs() {
        boolean triton = Nd4j.getNativeOps().isTritonAvailable();
        List<BenchmarkConfig> configs = new ArrayList<>();

        // DEFAULT: Best measured steady-state config for SmolDocling right now.
        // This is the ONLY config that runs unless vlm.test.configs selects others.
        // Target: ≥90 tok/s steady-state decode with ≥50% token diversity.
        if (triton) {
            configs.add(BenchmarkConfig.optimal());

            // Audit variants for isolating the remaining Triton/cublas decode knobs.
            configs.add(BenchmarkConfig.create("OPTIMAL_NO_NORM")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true)
                    .tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            configs.add(BenchmarkConfig.create("OPTIMAL_NO_BATCHED_GEMM")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true)
                    .tritonTf32(true)
                    .dspBatchedGemm(false)
                    .maxTokens(250).minDiversityPct(30));

            configs.add(BenchmarkConfig.create("OPTIMAL_NO_NORM_NO_BATCHED_GEMM")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true)
                    .tritonTf32(true)
                    .dspBatchedGemm(false)
                    .maxTokens(250).minDiversityPct(30));

            // ─── cuBLAS workspace matrix ─────────────────────────────────────
            // Tests all combinations of workspace ON/OFF × stages 1/2 × tf32 ON/OFF.
            // workspace=ON prevents MemAlloc graph nodes but may cause algorithm divergence.
            // Run these via: --configs WORKSPACE_ON_stages1_tf32,WORKSPACE_OFF_stages1_tf32,...

            // Workspace ON + stages=1 + TF32 (current OPTIMAL)
            configs.add(BenchmarkConfig.create("WORKSPACE_ON_stages1_tf32")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).cublasCaptureWorkspace(true)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            // Workspace OFF + stages=1 + TF32
            configs.add(BenchmarkConfig.create("WORKSPACE_OFF_stages1_tf32")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).cublasCaptureWorkspace(false)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            // Workspace ON + stages=2 + no TF32 (committed best before workspace change)
            configs.add(BenchmarkConfig.create("WORKSPACE_ON_stages2_noTf32")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .cublasTf32(false).cublasCaptureWorkspace(true)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            // Workspace OFF + stages=2 + no TF32 (original committed config)
            configs.add(BenchmarkConfig.create("WORKSPACE_OFF_stages2_noTf32")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .cublasTf32(false).cublasCaptureWorkspace(false)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            // Workspace ON + stages=2 + TF32 (test if TF32 compensates for workspace divergence)
            configs.add(BenchmarkConfig.create("WORKSPACE_ON_stages2_tf32")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .cublasTf32(true).cublasCaptureWorkspace(true)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            // Workspace OFF + stages=2 + TF32
            configs.add(BenchmarkConfig.create("WORKSPACE_OFF_stages2_tf32")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .cublasTf32(true).cublasCaptureWorkspace(false)
                    .dspBatchedGemm(true)
                    .maxTokens(250).minDiversityPct(30));

            // blockN=64 variant: lower shared mem (36KB vs 71KB), better occupancy, more K-loop iterations
            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps4_stages1_tf32_blockN64")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .tritonAttentionBlockN(64)
                    .cublasTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(35));

            // Warps/stages tuning experiments for 100 tok/s target
            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps4_stages1")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionMinScore(4.0f)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps4_stages1_noFusionScoring")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps4_stages2")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionMinScore(4.0f)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps4_stages2_noFusionScoring")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_MATMUL_gc_argOpt_batchGemmOnly_warps4_stages2_noFusionScoring")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM_AND_MATMUL + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(2)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages2")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionMinScore(4.0f)
                    .tritonNumWarps(2).tritonNumStages(2)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages2_noFusionScoring")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(2).tritonNumStages(2)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages1")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages1_noFusionScoring")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages1_score4")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionMinScore(4.0f)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages1_score3")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionMinScore(3.0f)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_gc_argOpt_batchGemmOnly_warps2_stages1_noPermuteSeq1")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonSpecializePermuteSeq1(false)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_noConcat_gc_argOpt_batchGemmOnly_warps2_stages1")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM_NO_CONCAT + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_NORM_MATMUL_gc_argOpt_batchGemmOnly_warps2_stages1")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES_WITH_NORM_AND_MATMUL + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(100).minDiversityPct(0));

            // Diagnostic twin of the default path without CUDA graph replay.
            // This keeps the same Triton/default op mix so op timing can attribute
            // the replay-time kernels back to actual ops.
            configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_noGC_argOpt_batchGemmOnly_warps2_stages1")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonNumWarps(2).tritonNumStages(1)
                    .dspBatchedGemm(true)
                    .maxTokens(20).minDiversityPct(0));
        }

        // DIAGNOSTIC: SLOT_BY_SLOT baseline — no Triton, no graph capture, proves model works
        configs.add(BenchmarkConfig.create("DIAG_SLOT_BY_SLOT_baseline")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(10)
                .minDiversityPct(0));

        // DIAGNOSTIC: Triton WITHOUT graph capture — isolates Triton kernel correctness
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
        }

        // DIAGNOSTIC: Triton WITHOUT graph capture + VERIFY — compares each Triton section vs slot-by-slot
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_VERIFY")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonVerifyKernels(true)
                    .maxTokens(3).minDiversityPct(0));
        }

        // DIAGNOSTIC: Triton + GC but WITHOUT ATTENTION — isolates attention compilation
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_Triton_gc_noATTN")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES)
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(10).minDiversityPct(0));
        }

        // DIAGNOSTIC: Triton + GC + ATTENTION but WITHOUT argOpt — isolates arg table
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_Triton_gc_ATTN_noArgOpt")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .maxTokens(10).minDiversityPct(0));
        }

        // DIAGNOSTIC: Full config without batch ops
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_full_noBatchOps")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(10).minDiversityPct(0));
        }

        // BINARY SEARCH: Isolate which Triton op type causes wrong output
        // Triton no GC, ATTENTION only (no COMPILE_ALL_TYPES)
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_ATTN_only")
                    .tritonIncludeTypes("ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
        }
        // Triton no GC, NO attention (COMPILE_ALL_TYPES only)
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_noATTN")
                    .tritonIncludeTypes(COMPILE_ALL_TYPES)
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
        }
        // Triton no GC, GATHER only
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_GATHER_only")
                    .tritonIncludeTypes("GATHER")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
        }
        // Triton no GC, CONST_GEN only
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_CONSTGEN_only")
                    .tritonIncludeTypes("CONST_GEN")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
        }
        // Triton no GC, CONCAT+SPLIT+STACK only
        if (triton) {
            configs.add(BenchmarkConfig.create("DIAG_TRITON_noGC_CONCAT_SPLIT_STACK")
                    .tritonIncludeTypes("CONCAT,SPLIT,STACK")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(10).minDiversityPct(0));
        }

        // NON-TRITON DSP modes: isolate whether bug is Triton-specific or DSP mode related
        // CUDA_GRAPHS without Triton include types cannot capture — graph capture requires
        // Triton-compiled kernels. Use AUTO mode instead, which falls back to slot-by-slot.
        configs.add(BenchmarkConfig.create("DIAG_CUDA_GRAPHS_noTriton")
                .executionMode(GraphExecutionMode.AUTO)
                .maxTokens(10).minDiversityPct(0));
        configs.add(BenchmarkConfig.create("DIAG_AUTO_noTriton")
                .executionMode(GraphExecutionMode.AUTO)
                .maxTokens(10).minDiversityPct(0));

        // Build the extended matrix whenever the caller explicitly selects configs.
        // That lets the benchmark script run isolated named configs without forcing ALL.
        String filterProp = System.getProperty("vlm.test.configs");
        boolean includeAll = filterProp != null && !filterProp.trim().isEmpty();
        if (!includeAll) return configs;

        // ── Additional configs below only run with vlm.test.configs=ALL ──

        // Baselines
        configs.add(BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("CUDA_GRAPHS")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .maxTokens(50));

        if (!triton) return configs;

        // 2. compileAll: individual section types (bisect crashes)
        for (String singleType : new String[]{"GATHER", "STACK", "CONST_GEN", "CONCAT", "SPLIT"}) {
            configs.add(BenchmarkConfig.create("TRITON_compileAll_" + singleType)
                    .tritonIncludeTypes(singleType)
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .maxTokens(20));
        }

        configs.add(BenchmarkConfig.create("TRITON_compileAll_safe")
                .tritonIncludeTypes("GATHER,STACK")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_gc")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .maxTokens(20));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // Diagnostic: GC + verify to find replay divergence
        configs.add(BenchmarkConfig.create("TRITON_gc_argOpt_VERIFY")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .tritonVerifyKernels(true)
                .tritonVerifyFullSnapshot(true)
                .maxTokens(5)
                .minDiversityPct(0));

        // Diagnostic: GC + force-recapture (re-capture every step, tests freshness)
        configs.add(BenchmarkConfig.create("TRITON_gc_argOpt_FORCE_RECAP")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .tritonForceRecapture(true)
                .maxTokens(10)
                .minDiversityPct(0));

        // Isolation: consolidated arg table only (no dirty tracking)
        configs.add(BenchmarkConfig.create("TRITON_gc_consolidatedOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(false)
                .maxTokens(20)
                .minDiversityPct(0));

        // Isolation: dirty tracking only (no consolidated arg table)
        configs.add(BenchmarkConfig.create("TRITON_gc_dirtyTrackingOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(false)
                .tritonArgDirtyTracking(true)
                .maxTokens(20)
                .minDiversityPct(0));

        // 3. FULL types: attention is biggest win (+23%)
        configs.add(BenchmarkConfig.create("TRITON_FULL_fused")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_FULL_fused_gc")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_FULL_fused_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 4. compileAll + FULL types combined
        configs.add(BenchmarkConfig.create("TRITON_compileAll_FULL")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonExcludeOps("matmul,batched_gemm")
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_FULL_gc")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonExcludeOps("matmul,batched_gemm")
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 5. Combined high-performance GC configs
        configs.add(BenchmarkConfig.create("TRITON_compileAll_FULL_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonExcludeOps("matmul,batched_gemm")
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_MAX_PERF_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonProfile("MAX_PERF")
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 6. DSP optimization flags (standalone)
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_castElim")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .maxTokens(50));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_fp16compute")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspFp16Compute(true)
                .maxTokens(50));

        // 7. DSP optimization flags + GC variants
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_castElim_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_fp16_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspFp16Compute(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 8. MAX_PERF profile (standalone, no GC)
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_MAX_PERF")
                .tritonIncludeTypes(COMPILE_ALL_TYPES)
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonProfile("MAX_PERF")
                .maxTokens(50));

        // 9. Ultimate combined: ATTN + castElim + fp16compute + GC + argOpt
        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_fp16_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .dspFp16Compute(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 10. ATTN + castElim only (no fp16compute), to isolate contributions
        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_gc_argOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true)
                .maxTokens(100)
                .minDiversityPct(0));

        // 11. Batch-zero + batched GEMM node reduction configs
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchZero")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .maxTokens(100).minDiversityPct(0));

        // Regression repro: enabling the full batch-ops bundle makes decode step 2 much slower.
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchOps")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .dspBatchedGemm(true)
                .dspCastSinkMatmul(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("CUDA_GRAPHS_batchOps")
                .executionMode(GraphExecutionMode.CUDA_GRAPHS)
                .dspBatchZero(true).dspBatchZeroKernel(true)
                .dspBatchedGemm(true)
                .maxTokens(50));

        // Isolation config: batched GEMM only (no batch-zero) to isolate correctness
        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps2")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(2)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps2_stages1")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps2_stages1_castElim")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .dspBatchedGemm(true)
                .dspCastElimination(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps2_stages1_castSink")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .dspBatchedGemm(true)
                .dspCastSinkMatmul(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps2_stages1_ctas2")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(2).tritonNumStages(1).tritonNumCTAs(2)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps1_stages1")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(1).tritonNumStages(1)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_compileAll_best_ATTN_gc_argOpt_batchGemmOnly_warps4_stages1")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonNumWarps(4).tritonNumStages(1)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        // Experimental combinations built from the best non-regressing knobs measured so far.
        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_batchGemmOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_gc_noArgOpt")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_gc_noArgOpt_batchGemmOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_gc_noArgOpt_MAX_PERF")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonProfile("MAX_PERF")
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_castElim_gc_noArgOpt_batchGemmOnly_MAX_PERF")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .dspCastElimination(true)
                .tritonProfile("MAX_PERF")
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        // Launch-tuning sweep: these only became meaningful once explicit overrides
        // stopped getting clobbered by the Triton profile defaults.
        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps4")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(4)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps2")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps4_stages1")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(4).tritonNumStages(1)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps4_batchGemmOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(4)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps4_stages1_batchGemmOnly")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(4).tritonNumStages(1)
                .dspBatchedGemm(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps1")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(1)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps2_stages1")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps2_ctas2")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2).tritonNumCTAs(2)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gc_noArgOpt_warps2_noFpFusion")
                .tritonIncludeTypes(COMPILE_ALL_TYPES + ",ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2).tritonEnableFpFusion(false)
                .maxTokens(100).minDiversityPct(0));

        // Scope-tuning sweep: compile only the fused attention path and let the rest
        // fall back to the native kernels if that reduces Triton section overhead.
        configs.add(BenchmarkConfig.create("TRITON_ATTN_only_gc")
                .tritonIncludeTypes("ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_only_gc_warps2_stages1")
                .tritonIncludeTypes("ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_ATTN_gather_stack_gc_warps2_stages1")
                .tritonIncludeTypes("GATHER,STACK,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_TRUE_FULL_gc")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .maxTokens(100).minDiversityPct(0));

        configs.add(BenchmarkConfig.create("TRITON_TRUE_FULL_gc_warps2_stages1")
                .tritonIncludeTypes(FULL_TRITON_TYPES)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonNumWarps(2).tritonNumStages(1)
                .maxTokens(100).minDiversityPct(0));

        return configs;
    }

    // ─── Setup ─────────────────────────────────────────────────────────────

    @BeforeAll
    public static void setup() {
        // Maven surefire sets undefined ${...} properties to empty string, not null.
        // Check for both null and empty to ensure defaults apply.
        String optEnabled = System.getProperty("nd4j.optimizer.enabled");
        if (optEnabled == null || optEnabled.isEmpty()) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        // Pre-cast FP32 weight constants to FP16 at model load time via optimizer.
        // This halves weight memory bandwidth and avoids the runtime per-matmul double-cast
        // that dspFp16Compute uses. MmulHelper's mixed-type path handles HALF×FLOAT
        // automatically, casting only the FP32 activation (1 cast vs 2).
        String fp16Prop = System.getProperty("nd4j.optimizer.fp16");
        if (fp16Prop == null || fp16Prop.isEmpty()) {
            System.setProperty("nd4j.optimizer.fp16", "true");
        }
        System.setProperty("nd4j.optimizer.logApplied", "true");

        // Debug flags passed via run-benchmark.sh or -D Maven properties

        pdfPath = System.getProperty("vlm.test.pdf.path");
        String pageStr = System.getProperty("vlm.test.pdf.page");
        if (pageStr != null && !pageStr.isEmpty()) {
            specificPage = Integer.parseInt(pageStr);
        }
        String dpiStr = System.getProperty("vlm.test.pdf.dpi");
        if (dpiStr != null && !dpiStr.isEmpty()) {
            renderDpi = Integer.parseInt(dpiStr);
        }
    }

    // ─── Main test: loads models once, sweeps all configs ──────────────────

    @Test
    @DisplayName("Optimized SmolDocling: Configuration matrix sweep")
    public void testOptimizedDoclingPipeline() throws Exception {
        long setupPhaseStart = phaseStart("PIPELINE_SETUP", benchmarkInputSummary());
        PipelineContext ctx;
        try {
            ctx = loadModelsAndPrepareEmbeddings();
        } catch (Throwable t) {
            throw phaseFailure("PIPELINE_SETUP", benchmarkInputSummary(), t);
        }
        phaseSuccess("PIPELINE_SETUP", setupPhaseStart,
                benchmarkInputSummary() + " " + summarizeTensor("inputsEmbeds", ctx.inputsEmbeds));

        // Assert pipeline setup produced valid state
        assertNotNull(ctx.decoder, "Decoder model must be loaded");
        assertNotNull(ctx.embedTokens, "EmbedTokens model must be loaded");
        assertNotNull(ctx.tokenizer, "Tokenizer must be loaded");
        assertNotNull(ctx.inputsEmbeds, "Input embeddings must be prepared");
        assertFalse(ctx.inputsEmbeds.wasClosed(), "Input embeddings must not be closed");
        assertTrue(ctx.hiddenSize > 0, "Hidden size must be positive, got: " + ctx.hiddenSize);
        assertTrue(ctx.promptTokenIds.length > 0, "Prompt token IDs must not be empty");
        assertTrue(ctx.decoderOps > 0, "Decoder should have ops");

        log.info("Pipeline setup complete: download={}ms import={}ms vision={}ms embed={}ms",
                ctx.downloadMs, ctx.importMs, ctx.visionMs, ctx.embedMs);
        log.info("  decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}, frames={}",
                ctx.decoderOps, ctx.embedOps, ctx.hiddenSize, ctx.promptTokenIds.length, ctx.visionFrames);

        List<BenchmarkConfig> configs = getAllConfigs();

        // Filter configs by name if vlm.test.configs is set (comma-separated).
        // "ALL" loads every config (handled in getAllConfigs). Specific names filter the list.
        String filterProp = System.getProperty("vlm.test.configs");
        if (filterProp != null && !filterProp.isEmpty() && !"ALL".equalsIgnoreCase(filterProp)) {
            Set<String> keep = Set.of(filterProp.split(","));
            configs.removeIf(c -> !keep.contains(c.getName()));
            log.info("Filtered to {} configs via vlm.test.configs: {}", configs.size(), keep);
        }

        // Override maxTokens for all configs if vlm.test.maxTokens is set
        String maxTokensOverride = System.getProperty("vlm.test.maxTokens");
        if (maxTokensOverride != null && !maxTokensOverride.isEmpty()) {
            int mt = Integer.parseInt(maxTokensOverride);
            configs.forEach(c -> c.maxTokens(mt));
            log.info("Override maxTokens={} for all {} configs", mt, configs.size());
        }

        List<SameDiff> models = List.of(ctx.decoder, ctx.embedTokens);

        // Compile function: delegates to BenchmarkConfigApplier
        BenchmarkRunner.CompileFunction compileFn = config -> {
            String configSummary = summarizeConfig(config);
            long phaseNs = phaseStart("CONFIG_COMPILE", configSummary);
            try {
                BenchmarkConfigApplier.compileModels(
                        ctx.decoder, "decoder", ctx.embedTokens, "embed_tokens", config);
                logDspState("POST_COMPILE " + config.getName(), ctx.decoder);
                phaseSuccess("CONFIG_COMPILE", phaseNs, configSummary);
            } catch (Throwable t) {
                logDspState("COMPILE_FAILURE " + config.getName(), ctx.decoder);
                throw phaseFailure("CONFIG_COMPILE", configSummary, t);
            }
        };

        // Decode function: wraps StaticKvCacheDecodeLoop
        BenchmarkRunner.DecodeFunction decodeFn = config -> {
            String configSummary = summarizeConfig(config);
            long phaseNs = phaseStart("CONFIG_DECODE",
                    configSummary + " "
                            + summarizeTokens("promptTokenIds", ctx.promptTokenIds) + " "
                            + summarizeTensor("inputsEmbeds", ctx.inputsEmbeds));
            String specTokensProp = System.getProperty("vlm.speculative.tokens", "0");
            int specTokens = (specTokensProp == null || specTokensProp.isEmpty()) ? 0 : Integer.parseInt(specTokensProp);
            boolean useDraft = config.isUseDraftModel()
                    || "true".equalsIgnoreCase(System.getProperty("vlm.speculative.draft"));
            if (useDraft && specTokens == 0) {
                specTokens = config.getDraftModelK() > 0 ? config.getDraftModelK() : 5;
            }

            // Build draft model speculator if requested
            Speculator draftSpeculator = null;
            if (useDraft && specTokens > 0) {
                draftSpeculator = buildDraftModelSpeculator(ctx, specTokens);
            }

            // Auto-discover I/O names from the decoder model graph
            ModelIOConfig decoderIOConfig = ModelIOConfig.discover(ctx.decoder);

            StaticKvCacheDecodeLoop.StaticKvCacheDecodeLoopBuilder loopBuilder = StaticKvCacheDecodeLoop.builder()
                    .decoder(ctx.decoder)
                    .embedTokens(ctx.embedTokens)
                    .tokenizer(ctx.tokenizer)
                    .ioConfig(decoderIOConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(config.getMaxTokens())
                    .maxSpeculativeTokens(specTokens)
                    .hiddenSize(ctx.hiddenSize);
            if (draftSpeculator != null) {
                loopBuilder.speculator(draftSpeculator);
            }
            try {
                logDspState("PRE_DECODE " + config.getName(), ctx.decoder);
                GenerationResult result = loopBuilder.build().decode(ctx.inputsEmbeds, ctx.promptTokenIds);
                logDspState("POST_DECODE " + config.getName(), ctx.decoder);
                phaseSuccess("CONFIG_DECODE", phaseNs, summarizeResult(result));
                return result;
            } catch (Throwable t) {
                logDspState("DECODE_FAILURE " + config.getName(), ctx.decoder);
                throw phaseFailure("CONFIG_DECODE", configSummary, t);
            }
        };

        // Validate function: structural tags + diversity checks
        BenchmarkRunner.ValidateFunction validateFn = (config, result) -> {
            long phaseNs = phaseStart("FINAL_VALIDATE",
                    config.getName() + " " + summarizeResult(result));
            try {
                validateResult(config, result);
                phaseSuccess("FINAL_VALIDATE", phaseNs, config.getName());
            } catch (Throwable t) {
                throw phaseFailure("FINAL_VALIDATE",
                        config.getName() + " " + summarizeResult(result), t);
            }
        };

        // Run the matrix
        List<BenchmarkResult> results = BenchmarkRunner.runMatrix(
                configs, List.of("decoder", "embed_tokens"), models,
                compileFn, decodeFn, validateFn, "vlm.config");

        ctx.tokenizer.close();
        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);

        // Print report (throws if any config failed)
        StringBuilder pipelineInfo = new StringBuilder();
        pipelineInfo.append(String.format("Pipeline setup: download=%dms import=%dms vision=%dms embed=%dms\n\n",
                ctx.downloadMs, ctx.importMs, ctx.visionMs, ctx.embedMs));
        log.info("{}", pipelineInfo);
        BenchmarkRunner.printReport(results);
    }

    // ─── validateResult ────────────────────────────────────────────────────

    private double effectiveThroughput(GenerationResult result) {
        if (result.getLateSteadyStateTokensPerSecond() > 0) {
            return result.getLateSteadyStateTokensPerSecond();
        }
        if (result.getSteadyStateTokensPerSecond() > 0) {
            return result.getSteadyStateTokensPerSecond();
        }
        if (result.getDecodeTokensPerSecond() > 0) {
            return result.getDecodeTokensPerSecond();
        }
        return result.getTokensPerSecond();
    }

    private String effectiveThroughputLabel(GenerationResult result) {
        if (result.getLateSteadyStateTokensPerSecond() > 0) {
            return "late steady-state";
        }
        if (result.getSteadyStateTokensPerSecond() > 0) {
            return "steady-state";
        }
        if (result.getDecodeTokensPerSecond() > 0) {
            return "decode-only";
        }
        return "overall";
    }

    private void validateResult(BenchmarkConfig config, GenerationResult result) {
        String name = config.getName();

        // Basic generation assertions
        assertNotNull(result.getText(), name + ": generated text is null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                name + ": should have generated at least one token");
        assertNotNull(result.getTokenIds(), name + ": token IDs array is null");
        assertEquals(result.getGeneratedTokenCount(), result.getTokenIds().length,
                name + ": token count mismatch with tokenIds array length");
        assertTrue(result.getGenerationTimeMs() > 0,
                name + ": generation time must be positive");
        assertTrue(result.getTokensPerSecond() > 0,
                name + ": tokens/sec must be positive");
        assertNotNull(result.getFinishReason(),
                name + ": finish reason is null");

        String trimmed = result.getText().trim();

        // Structural tag check
        if (config.isExpectStructuralTags() && result.getGeneratedTokenCount() >= 10) {
            boolean hasDocTags = trimmed.contains("<") && trimmed.contains(">");
            if (hasDocTags) {
                Set<String> tagTypes = extractTagTypes(trimmed);
                assertFalse(tagTypes.isEmpty(),
                        name + ": found angle brackets but extracted no tag types");
                boolean hasStructuralTags = tagTypes.stream().anyMatch(t ->
                        t.equals("doctag") || t.equals("page") || t.equals("text") ||
                                t.equals("section_header") || t.equals("otsl") || t.equals("table"));
                assertTrue(hasStructuralTags,
                        name + ": expected structural DocTags in " + result.getGeneratedTokenCount() +
                                " tokens. Tags found: " + tagTypes +
                                ". Text: " + trimmed.substring(0, Math.min(200, trimmed.length())));
            }
        }

        // Degeneracy check
        if (result.getGeneratedTokenCount() >= 10) {
            int[] tokenIds = result.getTokenIds();
            Set<Integer> uniqueTokens = new HashSet<>();
            for (int id : tokenIds) uniqueTokens.add(id);
            double uniqueRatio = (double) uniqueTokens.size() / tokenIds.length;
            log.info("  Token diversity: {}/{} unique ({}%)",
                    uniqueTokens.size(), tokenIds.length, String.format("%.1f", uniqueRatio * 100));
            assertTrue(uniqueRatio > config.getMinDiversityPct() / 100.0,
                    name + ": degenerate output: " + uniqueTokens.size() + "/" + tokenIds.length +
                            " unique (min " + config.getMinDiversityPct() + "%)");
        }

        // Throughput check
        if (result.getGeneratedTokenCount() >= 5) {
            assertTrue(result.getTokensPerSecond() > 0.1,
                    name + ": throughput too low: " +
                            String.format("%.2f", result.getTokensPerSecond()) + " tok/s");
        }
        if ("OPTIMAL".equals(name) && result.getGeneratedTokenCount() >= 20) {
            double effectiveThroughput = effectiveThroughput(result);
            String throughputLabel = effectiveThroughputLabel(result);
            assertTrue(effectiveThroughput >= 100.0,
                    name + ": native benchmark target missed: "
                            + throughputLabel + "=" + String.format("%.2f", effectiveThroughput)
                            + " tok/s (target 100.00 tok/s)");
        }
    }

    // ─── KvScatter isolation test ──────────────────────────────────────────

    @Test
    @DisplayName("Test KvScatter op in isolation")
    public void testKvScatterIsolated() {
        int batch = 1, heads = 8, maxKvLen = 100, dim = 64;
        long cachePos = 5;

        INDArray present = Nd4j.randn(DataType.FLOAT, batch, heads, maxKvLen + 1, dim);
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

        org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter scatter =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(staticBuf, present, cachePos);

        INDArray[] result = Nd4j.getExecutioner().exec(scatter);
        assertNotNull(result, "KvScatter result is null");
        assertTrue(result.length > 0, "KvScatter result is empty");

        INDArray expectedEntry = present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(maxKvLen), NDArrayIndex.all()).dup();
        INDArray actualEntry = result[0].get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();

        double maxDiff = expectedEntry.sub(actualEntry).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5, "KV scatter should copy present's last pos to static's cachePos, maxDiff=" + maxDiff);

        present.close();
        staticBuf.close();
    }

    // ─── loadModelsAndPrepareEmbeddings: one-time pipeline setup ──────────

    private PipelineContext loadModelsAndPrepareEmbeddings() throws Exception {
        PipelineContext ctx = new PipelineContext();
        Nd4j.getEnvironment().setTritonBuildThreads(4);

        long phaseNs;

        // Download
        phaseNs = phaseStart("DOWNLOAD_MODELS", benchmarkInputSummary());
        VLMModelDownloader.DownloadResult visionResult;
        VLMModelDownloader.DownloadResult decoderResult;
        VLMModelDownloader.DownloadResult embedTokensResult;
        VLMModelDownloader.DownloadResult tokenizerResult;
        try {
            long t0 = System.currentTimeMillis();
            log.info("Downloading models...");
            visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
            decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
            embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
            tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
            VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
            ctx.downloadMs = System.currentTimeMillis() - t0;
            log.info("Download done [{}ms]", ctx.downloadMs);
        } catch (Throwable t) {
            throw phaseFailure("DOWNLOAD_MODELS", benchmarkInputSummary(), t);
        }
        phaseSuccess("DOWNLOAD_MODELS", phaseNs,
                "decoder=" + safeFileName(decoderResult.getModelFile())
                        + " vision=" + safeFileName(visionResult.getModelFile())
                        + " embed=" + safeFileName(embedTokensResult.getModelFile())
                        + " tokenizer=" + safeFileName(tokenizerResult.getModelFile()));

        // Tokenizer
        phaseNs = phaseStart("TOKENIZER_LOAD", "tokenizer=" + safeFileName(tokenizerResult.getModelFile()));
        try {
            ctx.tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
            assertNotNull(ctx.tokenizer, "Tokenizer failed to load");
            assertTrue(ctx.tokenizer.getVocabSize() > 0, "Tokenizer vocab size must be positive");
            log.info("Tokenizer loaded: vocab_size={}", ctx.tokenizer.getVocabSize());
        } catch (Throwable t) {
            throw phaseFailure("TOKENIZER_LOAD",
                    "tokenizer=" + safeFileName(tokenizerResult.getModelFile()), t);
        }
        phaseSuccess("TOKENIZER_LOAD", phaseNs, "vocabSize=" + ctx.tokenizer.getVocabSize());

        // Import ONNX
        phaseNs = phaseStart("IMPORT_MODELS", "decoder=" + safeFileName(decoderResult.getModelFile()));
        SameDiff visionEncoder;
        try {
            long importStart = System.currentTimeMillis();
            log.info("Importing ONNX models...");
            boolean forceReoptimize = Boolean.getBoolean("vlm.model.cache.disable");
            if (forceReoptimize) {
                OnnxModelCache.invalidateCache(decoderResult.getModelFile().getAbsolutePath());
            }
            SameDiff[] models = OnnxModelCache.importAllWithCache(
                    visionResult.getModelFile().getAbsolutePath(),
                    decoderResult.getModelFile().getAbsolutePath(),
                    embedTokensResult.getModelFile().getAbsolutePath()
            );
            visionEncoder = models[0];
            ctx.decoder = models[1];
            ctx.embedTokens = models[2];
            ctx.importMs = System.currentTimeMillis() - importStart;

            assertNotNull(visionEncoder, "Vision encoder import failed");
            assertNotNull(ctx.decoder, "Decoder import failed");
            assertNotNull(ctx.embedTokens, "EmbedTokens import failed");
            ctx.decoderOps = ctx.decoder.getOps().size();
            ctx.embedOps = ctx.embedTokens.getOps().size();
            assertTrue(ctx.decoderOps > 0, "Decoder has no ops");
            assertTrue(ctx.embedOps > 0, "EmbedTokens has no ops");
            assertTrue(visionEncoder.getOps().size() > 0, "Vision encoder has no ops");

            log.info("ONNX import done [{}ms]: vision={} ops, decoder={} ops, embed={} ops",
                    ctx.importMs, visionEncoder.getOps().size(), ctx.decoderOps, ctx.embedOps);
        } catch (Throwable t) {
            throw phaseFailure("IMPORT_MODELS",
                    "decoder=" + safeFileName(decoderResult.getModelFile()), t);
        }
        phaseSuccess("IMPORT_MODELS", phaseNs,
                "visionOps=" + visionEncoder.getOps().size()
                        + " decoderOps=" + ctx.decoderOps
                        + " embedOps=" + ctx.embedOps);

        // Log op-type distribution for the decoder to verify optimizer ran
        Map<String, Integer> opCounts = new java.util.TreeMap<>();
        for (var entry : ctx.decoder.getOps().entrySet()) {
            var op = entry.getValue().getOp();
            String opName = op != null ? op.opName() : "null";
            opCounts.merge(opName, 1, Integer::sum);
        }
        log.info("Decoder op distribution ({} total):", ctx.decoderOps);
        opCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(25)
                .forEach(e -> log.info("  {} x {}", e.getValue(), e.getKey()));
        int rmsNormCount = opCounts.getOrDefault("rms_norm", 0);
        log.info("  rms_norm ops: {} (expected ~61 if optimizer ran)", rmsNormCount);
        if (rmsNormCount == 0) {
            log.warn("WARNING: No rms_norm ops found in decoder! GraphOptimizer may not have run. " +
                     "Check nd4j.optimizer.enabled=true and delete stale SDZ caches if needed.");
        }

        // Image preprocessing
        int targetSize = 512;
        phaseNs = phaseStart("IMAGE_PREPROCESS", benchmarkInputSummary());
        BufferedImage pdfImage;
        ImageTiler.SplitImageResult splitResult;
        INDArray imageInput;
        try {
            pdfImage = loadImageFromPdfOrGenerate();
            assertNotNull(pdfImage, "Failed to load/generate test image");
            assertTrue(pdfImage.getWidth() > 0 && pdfImage.getHeight() > 0, "Test image has zero dimensions");

            BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
            splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, 9);
            ctx.visionFrames = splitResult.getTotalFrames();
            assertTrue(ctx.visionFrames > 0, "No vision frames produced");

            PreprocessorConfig ppConfig = new PreprocessorConfig();
            ppConfig.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
            ppConfig.setDoRescale(true);
            ppConfig.setRescaleFactor(1.0 / 255.0);
            ppConfig.setDoNormalize(true);
            ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
            ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
            VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
            imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);
            preprocessor.shutdown();
            assertNotNull(imageInput, "Image preprocessing returned null");
            assertFalse(imageInput.wasClosed(), "Image tensor closed after preprocessing");
        } catch (Throwable t) {
            throw phaseFailure("IMAGE_PREPROCESS", benchmarkInputSummary(), t);
        }
        phaseSuccess("IMAGE_PREPROCESS", phaseNs,
                "image=" + pdfImage.getWidth() + "x" + pdfImage.getHeight()
                        + " frames=" + ctx.visionFrames + " "
                        + summarizeTensor("imageInput", imageInput));

        // Vision encoder - process each frame sequentially
        phaseNs = phaseStart("VISION_ENCODE",
                "frames=" + ctx.visionFrames + " " + summarizeTensor("imageInput", imageInput));
        INDArray visionEmbeddings;
        try {
            long visionStart = System.currentTimeMillis();
            log.info("Running vision encoder on {} frames...", ctx.visionFrames);
            List<String> visionInputNames = visionEncoder.inputs();
            String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
            assertFalse(visionInputNames.isEmpty(), "Vision encoder has no inputs");
            assertTrue(visionOutputNames.length > 0, "Vision encoder has no outputs");

            List<INDArray> frameEmbeddings = new ArrayList<>();
            for (int frameIdx = 0; frameIdx < ctx.visionFrames; frameIdx++) {
                INDArray frameSlice = imageInput.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();

                Map<String, INDArray> visionInputMap = new HashMap<>();
                for (String inputName : visionInputNames) {
                    if (inputName.equals("pixel_values")) {
                        visionInputMap.put(inputName, singleFrame);
                    } else if (inputName.equals("pixel_attention_mask")) {
                        ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                        visionInputMap.put(inputName,
                                ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                    }
                }

                Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
                assertNotNull(visionOutputs, "Vision encoder output null for frame " + frameIdx);
                assertFalse(visionOutputs.isEmpty(), "Vision encoder output empty for frame " + frameIdx);

                VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
                assertNotNull(selected, "Vision encoder selected output null for frame " + frameIdx);
                assertNotNull(selected.tensor, "Vision encoder selected tensor null for frame " + frameIdx);
                assertTrue(selected.tensor.rank() >= 2, "Vision output rank < 2 for frame " + frameIdx);

                INDArray out = selected.tensor.dup();
                assertFalse(out.wasClosed(), "Vision output dup closed for frame " + frameIdx);
                frameEmbeddings.add(out);

                for (var entry : visionOutputs.entrySet()) {
                    INDArray arr = entry.getValue();
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
                singleFrame.close();
            }

            // Clean up vision encoder
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();

            assertEquals(ctx.visionFrames, frameEmbeddings.size(),
                    "Frame embedding count mismatch: expected " + ctx.visionFrames);

            visionEmbeddings = frameEmbeddings.size() == 1
                    ? frameEmbeddings.get(0).dup()
                    : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
            for (INDArray fe : frameEmbeddings) {
                if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
            }
            imageInput.close();
            ctx.visionMs = System.currentTimeMillis() - visionStart;

            assertFalse(visionEmbeddings.wasClosed(), "Concatenated vision embeddings closed");
            assertTrue(visionEmbeddings.rank() == 3, "Vision embeddings should be rank 3, got " + visionEmbeddings.rank());
            log.info("Vision encoder done [{}ms]: shape={}", ctx.visionMs,
                    Arrays.toString(visionEmbeddings.shape()));
        } catch (Throwable t) {
            throw phaseFailure("VISION_ENCODE", "frames=" + ctx.visionFrames, t);
        }
        phaseSuccess("VISION_ENCODE", phaseNs,
                "frames=" + ctx.visionFrames + " " + summarizeTensor("visionEmbeddings", visionEmbeddings));

        freeModelConstants(visionEncoder, "vision encoder");

        // Build prompt + embeddings
        phaseNs = phaseStart("PROMPT_EMBED",
                "visionFrames=" + ctx.visionFrames + " " + summarizeTensor("visionEmbeddings", visionEmbeddings));
        try {
            long embedStart = System.currentTimeMillis();
            int imageTokenId = ImagePromptBuilder.resolveImageTokenId(ctx.tokenizer);
            assertTrue(imageTokenId >= 0, "Image token ID should be non-negative");

            int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / ctx.visionFrames;
            assertTrue(imageSeqLenPerFrame > 0, "Image seq len per frame must be positive");

            String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                    splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
            String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
            ctx.promptTokenIds = ctx.tokenizer.encode(chatPrompt, false).getIds();
            assertTrue(ctx.promptTokenIds.length > 0, "Prompt encoding produced no tokens");

            INDArray promptIdsTensor = Nd4j.createFromArray(ctx.promptTokenIds)
                    .reshape(1, ctx.promptTokenIds.length).castTo(DataType.LONG);
            String embedInputName = ctx.embedTokens.inputs().isEmpty() ? "input_ids" : ctx.embedTokens.inputs().get(0);
            String[] embedOutputNames = ctx.embedTokens.outputs().toArray(new String[0]);
            Map<String, INDArray> embedOutputs = ctx.embedTokens.output(
                    Map.of(embedInputName, promptIdsTensor), embedOutputNames);
            assertNotNull(embedOutputs, "EmbedTokens output is null");
            assertFalse(embedOutputs.isEmpty(), "EmbedTokens produced no output");

            INDArray textEmbeddings = null;
            for (var entry : embedOutputs.entrySet()) {
                textEmbeddings = entry.getValue().dup();
            }
            assertNotNull(textEmbeddings, "embed_tokens produced no output");
            assertFalse(textEmbeddings.wasClosed(), "Text embeddings closed after dup");

            ctx.hiddenSize = visionEmbeddings.shape()[2];
            assertEquals(ctx.hiddenSize, textEmbeddings.shape()[2],
                    "Hidden size mismatch: vision=" + ctx.hiddenSize + " text=" + textEmbeddings.shape()[2]);
            assertTrue(ctx.hiddenSize > 0, "Hidden size must be positive");

            ctx.inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings, visionEmbeddings, ctx.promptTokenIds, imageTokenId);
            assertNotNull(ctx.inputsEmbeds, "Merged embeddings are null");
            assertFalse(ctx.inputsEmbeds.wasClosed(), "Merged embeddings are closed");
            assertTrue(ctx.inputsEmbeds.rank() == 3,
                    "Merged embeddings should be rank 3, got " + ctx.inputsEmbeds.rank());

            if (textEmbeddings.closeable() && !textEmbeddings.wasClosed()) textEmbeddings.close();
            ctx.embedMs = System.currentTimeMillis() - embedStart;
            log.info("Embeddings merged [{}ms]: shape={}", ctx.embedMs,
                    Arrays.toString(ctx.inputsEmbeds.shape()));
        } catch (Throwable t) {
            throw phaseFailure("PROMPT_EMBED",
                    "visionFrames=" + ctx.visionFrames + " " + summarizeTensor("visionEmbeddings", visionEmbeddings), t);
        }
        phaseSuccess("PROMPT_EMBED", phaseNs,
                summarizeTokens("promptTokenIds", ctx.promptTokenIds) + " "
                        + summarizeTensor("inputsEmbeds", ctx.inputsEmbeds));

        return ctx;
    }

    // ─── Utility helpers ──────────────────────────────────────────────────

    private boolean isPhaseLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.phaseLogging", "true"));
    }

    private boolean isTensorFingerprintLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.tensorFingerprints", "false"));
    }

    private boolean isDspStateLoggingEnabled() {
        return Boolean.parseBoolean(System.getProperty("vlm.benchmark.dspStateLogging", "true"));
    }

    private int tensorFingerprintSamples() {
        return Integer.getInteger("vlm.benchmark.tensorSampleValues", 8);
    }

    private long phaseStart(String phase, String detail) {
        if (isPhaseLoggingEnabled()) {
            log.info("[PHASE] START {} {}", phase, detail);
        }
        return System.nanoTime();
    }

    private void phaseSuccess(String phase, long startNs, String detail) {
        if (isPhaseLoggingEnabled()) {
            long elapsedMs = (System.nanoTime() - startNs) / 1_000_000;
            log.info("[PHASE] OK {} {}ms {}", phase, elapsedMs, detail);
        }
    }

    private IllegalStateException phaseFailure(String phase, String detail, Throwable cause) {
        log.error("[PHASE] FAIL {} {}: {}", phase, detail, cause.getMessage(), cause);
        return new IllegalStateException("Benchmark phase " + phase + " failed: " + detail, cause);
    }

    private String benchmarkInputSummary() {
        return "pdf=" + (pdfPath != null && !pdfPath.isEmpty() ? pdfPath : "<generated>")
                + " page=" + (specificPage >= 0 ? specificPage : 0)
                + " dpi=" + renderDpi;
    }

    private String summarizeConfig(BenchmarkConfig config) {
        return "config=" + config.getName()
                + " maxTokens=" + config.getMaxTokens()
                + " triton=" + config.isTriton()
                + " compileAll=" + config.isTritonCompileAll()
                + " graphCapture=" + config.isTritonGraphCapture()
                + " noFallbackCapture=" + !config.isTritonAllowFallbackCapture()
                + " batchedGemm=" + config.isDspBatchedGemm();
    }

    private String summarizeTensor(String label, INDArray arr) {
        if (arr == null) {
            return label + "{null}";
        }

        StringBuilder sb = new StringBuilder(label).append("{shape=");
        try {
            sb.append(Arrays.toString(arr.shape()))
                    .append(",dtype=").append(arr.dataType())
                    .append(",length=").append(arr.length())
                    .append(",closed=").append(arr.wasClosed());

            if (isTensorFingerprintLoggingEnabled() && !arr.wasClosed() && arr.length() > 0) {
                INDArray flat = arr.reshape(arr.length());
                long len = flat.length();
                long stride = Math.max(1L, len / Math.max(1, tensorFingerprintSamples()));
                int sampled = 0;
                double sampleMin = Double.POSITIVE_INFINITY;
                double sampleMax = Double.NEGATIVE_INFINITY;
                double sampleSum = 0.0;
                double checksum = 0.0;
                boolean sampleHasNaN = false;
                for (long idx = 0; idx < len && sampled < tensorFingerprintSamples(); idx += stride) {
                    double value = flat.getDouble(idx);
                    sampleMin = Math.min(sampleMin, value);
                    sampleMax = Math.max(sampleMax, value);
                    sampleSum += value;
                    checksum += value * (idx + 1);
                    sampleHasNaN |= Double.isNaN(value);
                    sampled++;
                }
                if (sampled == 0) {
                    double value = flat.getDouble(0);
                    sampleMin = value;
                    sampleMax = value;
                    sampleSum = value;
                    checksum = value;
                    sampleHasNaN = Double.isNaN(value);
                    sampled = 1;
                }
                sb.append(",sampled=").append(sampled)
                        .append(",stride=").append(stride)
                        .append(",sampleMin=").append(String.format("%.6f", sampleMin))
                        .append(",sampleMax=").append(String.format("%.6f", sampleMax))
                        .append(",sampleMean=").append(String.format("%.6f", sampleSum / sampled))
                        .append(",sampleChecksum=").append(String.format("%.6f", checksum))
                        .append(",sampleHasNaN=").append(sampleHasNaN);
            }
        } catch (Throwable t) {
            sb.append("?,fingerprintError=").append(t.getClass().getSimpleName())
                    .append(":").append(t.getMessage());
        }

        return sb.append("}").toString();
    }

    private String summarizeTokens(String label, int[] tokens) {
        if (tokens == null) {
            return label + "{null}";
        }

        int preview = Math.min(tokens.length, 8);
        int tailStart = Math.max(0, tokens.length - 8);
        return label + "{count=" + tokens.length
                + ",head=" + Arrays.toString(Arrays.copyOfRange(tokens, 0, preview))
                + ",tail=" + Arrays.toString(Arrays.copyOfRange(tokens, tailStart, tokens.length))
                + "}";
    }

    private String summarizeResult(GenerationResult result) {
        if (result == null) {
            return "result{null}";
        }

        return "result{tokens=" + result.getGeneratedTokenCount()
                + ",finish=" + result.getFinishReason()
                + ",throughputLabel=" + effectiveThroughputLabel(result)
                + ",throughput=" + String.format("%.2f", effectiveThroughput(result))
                + ",text='" + safeSnippet(result.getText(), 160) + "'"
                + "," + summarizeTokens("tokenIds", result.getTokenIds())
                + "}";
    }

    private String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "<null>";
        }

        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        if (normalized.length() <= maxChars) {
            return normalized;
        }
        return normalized.substring(0, maxChars) + "...";
    }

    private String safeFileName(File file) {
        return file == null ? "<null>" : file.getName();
    }

    private void logDspState(String phase, SameDiff model) {
        if (!isDspStateLoggingEnabled() || model == null) {
            return;
        }

        try {
            DspDebugger debugger = DspDebugger.attach(model);
            DspDebugger.PlanReport planReport = debugger.analyzePlan();
            DspDebugger.GraphReplayReport replayReport = debugger.analyzeGraphReplay();

            if (planReport.errorMessage != null || replayReport.errorMessage != null) {
                log.info("[DSP] {} plan={} replay={}",
                        phase, planReport.errorMessage, replayReport.errorMessage);
                return;
            }

            log.info("[DSP] {} planPhase={} pointersStable={} fullyReplaying={} frozenExec={} segments={} replaying={} captureFailures={} stuck={} riskyOps={} unfrozenOps={}",
                    phase,
                    replayReport.planPhase,
                    replayReport.pointersStable,
                    replayReport.isFullyReplaying(),
                    replayReport.frozenExecutionCount,
                    replayReport.numSegments,
                    replayReport.getReplayingSegments().size(),
                    replayReport.getCaptureFailures().size(),
                    replayReport.getStuckSegments().size(),
                    planReport.getRiskyOps().size(),
                    planReport.getUnfrozenOps().size());
        } catch (Throwable t) {
            log.warn("[DSP] {} state unavailable: {}", phase, t.getMessage());
        }
    }

    private Set<String> extractTagTypes(String text) {
        Set<String> tagTypes = new HashSet<>();
        int idx = 0;
        while (idx < text.length()) {
            int open = text.indexOf('<', idx);
            if (open < 0) break;
            int close = text.indexOf('>', open);
            if (close < 0) break;
            String tag = text.substring(open + 1, close);
            if (tag.startsWith("/")) tag = tag.substring(1);
            int space = tag.indexOf(' ');
            if (space > 0) tag = tag.substring(0, space);
            if (!tag.isEmpty()) tagTypes.add(tag);
            idx = close + 1;
        }
        return tagTypes;
    }

    private BufferedImage loadImageFromPdfOrGenerate() throws IOException {
        if (pdfPath != null && new File(pdfPath).exists()) {
            try (PDDocument document = PDDocument.load(new File(pdfPath))) {
                PDFRenderer renderer = new PDFRenderer(document);
                return renderer.renderImageWithDPI(specificPage >= 0 ? specificPage : 0, renderDpi, ImageType.RGB);
            }
        }
        BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 512, 512);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("This is a test page for the", 50, 220);
        g.drawString("SmolDocling VLM pipeline.", 50, 260);
        g.drawString("Section 2: Content", 50, 340);
        g.drawString("Lorem ipsum dolor sit amet,", 50, 400);
        g.drawString("consectetur adipiscing elit.", 50, 440);
        g.dispose();
        return img;
    }

    private void freeModelConstants(SameDiff model, String label) {
        int closedArrays = 0;
        long closedBytes = 0;
        for (org.nd4j.autodiff.samediff.ArrayHolder holder :
                new org.nd4j.autodiff.samediff.ArrayHolder[]{model.getConstantArrays(), model.getVariablesArrays()}) {
            for (String name : new ArrayList<>(holder.arrayNames())) {
                INDArray arr = holder.removeArray(name);
                if (arr != null && !arr.wasClosed()) {
                    closedBytes += arr.length() * arr.dataType().width();
                    arr.data().setConstant(false);
                    arr.close();
                    closedArrays++;
                }
            }
        }
        Nd4j.getExecutioner().commit();
        log.info("  Freed {} {} arrays ({}MB)", closedArrays, label, closedBytes / (1024 * 1024));
    }

    // ─── Draft model speculation support ──────────────────────────────────

    /**
     * Build a DraftModelSpeculator using SmolLM2-135M as the draft model.
     * Lazy-loads the draft model into PipelineContext on first call.
     */
    private Speculator buildDraftModelSpeculator(PipelineContext ctx, int maxSpecTokens) {
        try {
            if (ctx.draftDecoder == null) {
                loadDraftModel(ctx);
            }

            // Extract embedding table from draft model
            INDArray draftEmbeddingTable = null;
            for (org.nd4j.autodiff.samediff.SDVariable var : ctx.draftDecoder.variables()) {
                if (var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT
                        || var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                    INDArray arr = var.getArr();
                    if (arr != null && arr.rank() == 2) {
                        if (draftEmbeddingTable == null || arr.length() > draftEmbeddingTable.length()) {
                            draftEmbeddingTable = arr;
                        }
                    }
                }
            }

            if (draftEmbeddingTable == null) {
                throw new RuntimeException("Could not extract embedding table from draft model");
            }

            long draftHidden = ctx.draftHiddenSize > 0 ? ctx.draftHiddenSize : draftEmbeddingTable.size(1);

            // Auto-discover I/O names from the draft model graph
            ModelIOConfig draftIOConfig = ModelIOConfig.discover(ctx.draftDecoder);

            // Embed function: direct table lookup
            final INDArray embedTable = draftEmbeddingTable;
            final long hidden = draftHidden;
            java.util.function.Function<int[], INDArray> embedFn = tokenIds -> {
                INDArray emb = Nd4j.zeros(DataType.FLOAT, 1, tokenIds.length, hidden);
                for (int i = 0; i < tokenIds.length; i++) {
                    emb.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                            .assign(embedTable.getRow(tokenIds[i]));
                }
                return emb;
            };

            // Decode function: greedy argmax from logits
            java.util.function.Function<INDArray, Integer> decodeFn = logits -> {
                INDArray lastLogits;
                if (logits.rank() == 3) {
                    lastLogits = logits.get(NDArrayIndex.point(0),
                            NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all());
                } else if (logits.rank() == 2) {
                    lastLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.all());
                } else {
                    lastLogits = logits;
                }
                return Nd4j.argMax(lastLogits).getInt(0);
            };

            log.info("  Draft model speculator: hidden={}, logits={}, kvLayers={}",
                    draftHidden, draftIOConfig.getLogitsOutputName(),
                    draftIOConfig.getKvCacheNames() != null ? draftIOConfig.getKvCacheNames().keyNames.size() : 0);

            long draftVocabSize = draftEmbeddingTable.size(0);
            return new DraftModelSpeculator(
                    "draft-smollm2-135m",
                    ctx.draftDecoder,
                    embedFn,
                    decodeFn,
                    draftIOConfig,
                    draftHidden,
                    draftVocabSize,
                    maxSpecTokens,
                    256,
                    ctx.draftDeviceId);
        } catch (Exception e) {
            log.error("Failed to build draft model speculator, falling back to ngram", e);
            return null;
        }
    }

    /**
     * Load SmolLM2-135M ONNX model as the draft decoder.
     */
    private void loadDraftModel(PipelineContext ctx) throws Exception {
        log.info("  Loading SmolLM2-135M draft model...");
        long startMs = System.currentTimeMillis();

        // Download ONNX model
        VLMModelDownloader.DownloadResult draftResult = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLLM2_135M_DECODER);
        File draftOnnx = draftResult.getModelFile();
        log.info("  Draft model downloaded: {} ({}MB)", draftOnnx.getName(),
                draftOnnx.length() / (1024 * 1024));

        // Load draft model on device 1 (if available) to reduce memory pressure on
        // device 0 where the target model's CUDA graph needs memory for replay.
        // The draft model is small (~515MB) and runs without CUDA graph acceleration.
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        int draftDevice = numDevices > 1 ? 1 : 0;
        int origDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        if (draftDevice != origDevice) {
            log.info("  Loading draft model on device {} (freeing device {} for target model graph)",
                    draftDevice, origDevice);
            DeviceMemoryManager.getInstance().switchDevice(draftDevice, "loadDraftModel", "draft-load");
        }

        // Import via SDZ cache (same path as main models)
        ctx.draftDecoder = OnnxModelCache.importWithCache(draftOnnx.getAbsolutePath());

        // Restore original device
        if (draftDevice != origDevice) {
            DeviceMemoryManager.getInstance().switchDevice(origDevice, "loadDraftModel", "restore");
        }

        ctx.draftDeviceId = draftDevice;
        long elapsed = System.currentTimeMillis() - startMs;
        log.info("  Draft model loaded in {}ms on device {}, ops={}", elapsed, draftDevice,
                ctx.draftDecoder.ops().length);

        // Try to read hidden size from config
        try {
            VLMModelDownloader.DownloadResult configResult = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLLM2_135M_CONFIG);
            org.eclipse.deeplearning4j.llm.config.ModelConfig modelConfig =
                    org.eclipse.deeplearning4j.llm.config.ModelConfig.fromFile(configResult.getModelFile());
            ctx.draftHiddenSize = modelConfig.getHiddenSize();
            log.info("  Draft model config: hidden_size={}", ctx.draftHiddenSize);
        } catch (Exception e) {
            log.warn("  Could not load draft model config, will infer hidden size from embedding table", e);
        }
    }
}
