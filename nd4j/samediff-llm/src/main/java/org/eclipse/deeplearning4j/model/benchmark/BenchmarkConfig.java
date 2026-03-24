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

import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;

/**
 * Unified benchmark configuration for model performance testing.
 *
 * Supports all execution modes (SLOT_BY_SLOT, CUDA_GRAPHS, Triton, etc.)
 * and Triton compiler settings. Use the fluent builder pattern:
 * <pre>
 *   BenchmarkConfig.create("MY_CONFIG")
 *       .executionMode(GraphExecutionMode.CUDA_GRAPHS)
 *       .maxTokens(50)
 * </pre>
 */
public class BenchmarkConfig {

    // Identity
    String name;

    // Execution mode — null means Triton
    GraphExecutionMode executionMode;

    // Triton settings
    String tritonIncludeTypes = "";
    boolean tritonSectionFusion;
    boolean tritonGraphCapture;
    boolean tritonConsolidatedArgTable;
    boolean tritonArgDirtyTracking;
    boolean tritonCooperativeLaunch;
    int tritonCoopTargetBlocks = -1;
    boolean tritonCompileAll;
    String tritonExcludeOps = "";
    boolean tritonSkipKernels;
    boolean tritonVerifyKernels;
    boolean tritonVerifyFullSnapshot;
    boolean tritonForceRecapture;
    boolean tritonVerbose;
    boolean tritonDumpSections;
    int tritonNumWarps = -1;
    int tritonNumStages = -1;
    int tritonNumCTAs = -1;
    int tritonMaxNreg = -1;
    int tritonAttentionBlockN = -1;
    boolean tritonEnableFpFusion = true;
    int tritonMaxSubsegmentOps = -1;
    int tritonMaxSubsegmentSections = -1;
    boolean tritonAllowFallbackCapture;
    int tritonBuildThreads = -1;
    String tritonProfile = "BALANCED";

    // Triton segment fusion optimization flags (temporary for testing)
    boolean tritonFuseIdentityShapes = true;
    boolean tritonFuseCastChains = true;
    boolean tritonFuseTrivialGather = true;
    boolean tritonSpecializePermuteSeq1 = true;
    boolean tritonEliminateConcatSplitPairs = true;
    boolean tritonFusedMatmul = false;  // HIGH RISK - disabled by default
    
    // Triton fusion scoring
    boolean tritonFusionScoring = true;
    float tritonFusionMinScore = 5.0f;

    // cuBLAS flags
    boolean cublasTf32;
    boolean tritonTf32;  // TF32 precision for Triton-compiled DotOps (default OFF for accuracy)
    boolean cublasCaptureWorkspace = true;  // default ON — prevents MemAlloc graph nodes

    // DSP flags
    boolean dspCastElimination;
    boolean dspCastSinkMatmul;
    boolean dspFp16Compute;
    boolean dspBatchZero;
    boolean dspBatchZeroKernel;
    boolean dspBatchedGemm;

    // Draft model speculation
    boolean useDraftModel;
    int draftModelK = 5;  // number of speculative tokens per step

    // Generation
    int captureMinExec = 1;
    int maxTokens = 20;

    // Validation
    double minDiversityPct = 5.0;
    String[] expectedSubstrings;
    boolean expectStructuralTags = true;

    // DSP accuracy validation
    boolean validationMode;
    ValidationConfig validationConfig;

    public static BenchmarkConfig create(String name) {
        BenchmarkConfig c = new BenchmarkConfig();
        c.name = name;
        return c;
    }

    /**
     * Returns the best-known benchmark configuration for LLM decode.
     * This is the single source of truth for optimal inference settings.
     * Matches: warps4_stages1_tf32_default (~86 tok/s on RTX 4090).
     *
     * @see org.nd4j.linalg.factory.Environment#applyOptimalLLMConfig()
     */
    public static BenchmarkConfig optimal() {
        return create("OPTIMAL")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true)
                .tritonCompileAll(true)
                .tritonGraphCapture(true).tritonAllowFallbackCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonFusionScoring(false)
                .tritonNumWarps(4).tritonNumStages(1)
                .cublasTf32(true)
                .tritonTf32(true)
                .dspBatchedGemm(true)
                .maxTokens(250).minDiversityPct(30);
    }

    // Fluent setters
    public BenchmarkConfig executionMode(GraphExecutionMode m) { this.executionMode = m; return this; }
    public BenchmarkConfig tritonIncludeTypes(String t) { this.tritonIncludeTypes = t; return this; }
    public BenchmarkConfig tritonSectionFusion(boolean b) { this.tritonSectionFusion = b; return this; }
    public BenchmarkConfig tritonGraphCapture(boolean b) { this.tritonGraphCapture = b; return this; }
    public BenchmarkConfig tritonConsolidatedArgTable(boolean b) { this.tritonConsolidatedArgTable = b; return this; }
    public BenchmarkConfig tritonArgDirtyTracking(boolean b) { this.tritonArgDirtyTracking = b; return this; }
    public BenchmarkConfig tritonCooperativeLaunch(boolean b) { this.tritonCooperativeLaunch = b; return this; }
    public BenchmarkConfig tritonCoopTargetBlocks(int n) { this.tritonCoopTargetBlocks = n; return this; }
    public BenchmarkConfig tritonCompileAll(boolean b) { this.tritonCompileAll = b; return this; }
    public BenchmarkConfig tritonExcludeOps(String s) { this.tritonExcludeOps = s; return this; }
    public BenchmarkConfig tritonSkipKernels(boolean b) { this.tritonSkipKernels = b; return this; }
    public BenchmarkConfig tritonVerifyKernels(boolean b) { this.tritonVerifyKernels = b; return this; }
    public BenchmarkConfig tritonVerifyFullSnapshot(boolean b) { this.tritonVerifyFullSnapshot = b; return this; }
    public BenchmarkConfig tritonForceRecapture(boolean b) { this.tritonForceRecapture = b; return this; }
    public BenchmarkConfig tritonVerbose(boolean b) { this.tritonVerbose = b; return this; }
    public BenchmarkConfig tritonDumpSections(boolean b) { this.tritonDumpSections = b; return this; }
    public BenchmarkConfig tritonNumWarps(int n) { this.tritonNumWarps = n; return this; }
    public BenchmarkConfig tritonNumStages(int n) { this.tritonNumStages = n; return this; }
    public BenchmarkConfig tritonNumCTAs(int n) { this.tritonNumCTAs = n; return this; }
    public BenchmarkConfig tritonMaxNreg(int n) { this.tritonMaxNreg = n; return this; }
    public BenchmarkConfig tritonAttentionBlockN(int n) { this.tritonAttentionBlockN = n; return this; }
    public BenchmarkConfig tritonEnableFpFusion(boolean b) { this.tritonEnableFpFusion = b; return this; }
    public BenchmarkConfig tritonMaxSubsegmentOps(int n) { this.tritonMaxSubsegmentOps = n; return this; }
    public BenchmarkConfig tritonMaxSubsegmentSections(int n) { this.tritonMaxSubsegmentSections = n; return this; }
    public BenchmarkConfig tritonAllowFallbackCapture(boolean b) { this.tritonAllowFallbackCapture = b; return this; }
    public BenchmarkConfig tritonBuildThreads(int n) { this.tritonBuildThreads = n; return this; }
    public BenchmarkConfig tritonProfile(String p) { this.tritonProfile = p; return this; }
    
    // Triton segment fusion flags
    public BenchmarkConfig tritonFuseIdentityShapes(boolean b) { this.tritonFuseIdentityShapes = b; return this; }
    public BenchmarkConfig tritonFuseCastChains(boolean b) { this.tritonFuseCastChains = b; return this; }
    public BenchmarkConfig tritonFuseTrivialGather(boolean b) { this.tritonFuseTrivialGather = b; return this; }
    public BenchmarkConfig tritonSpecializePermuteSeq1(boolean b) { this.tritonSpecializePermuteSeq1 = b; return this; }
    public BenchmarkConfig tritonEliminateConcatSplitPairs(boolean b) { this.tritonEliminateConcatSplitPairs = b; return this; }
    public BenchmarkConfig tritonFusedMatmul(boolean b) { this.tritonFusedMatmul = b; return this; }
    public BenchmarkConfig tritonFusionScoring(boolean b) { this.tritonFusionScoring = b; return this; }
    public BenchmarkConfig tritonFusionMinScore(float f) { this.tritonFusionMinScore = f; return this; }
    
    public BenchmarkConfig captureMinExec(int n) { this.captureMinExec = n; return this; }
    public BenchmarkConfig maxTokens(int n) { this.maxTokens = n; return this; }
    public BenchmarkConfig minDiversityPct(double d) { this.minDiversityPct = d; return this; }
    public BenchmarkConfig expectedSubstrings(String... s) { this.expectedSubstrings = s; return this; }
    public BenchmarkConfig expectStructuralTags(boolean b) { this.expectStructuralTags = b; return this; }
    public BenchmarkConfig cublasTf32(boolean b) { this.cublasTf32 = b; return this; }
    public BenchmarkConfig tritonTf32(boolean b) { this.tritonTf32 = b; return this; }
    public BenchmarkConfig cublasCaptureWorkspace(boolean b) { this.cublasCaptureWorkspace = b; return this; }
    public BenchmarkConfig dspCastElimination(boolean b) { this.dspCastElimination = b; return this; }
    public BenchmarkConfig dspCastSinkMatmul(boolean b) { this.dspCastSinkMatmul = b; return this; }
    public BenchmarkConfig dspFp16Compute(boolean b) { this.dspFp16Compute = b; return this; }
    public BenchmarkConfig dspBatchZero(boolean b) { this.dspBatchZero = b; return this; }
    public BenchmarkConfig dspBatchZeroKernel(boolean b) { this.dspBatchZeroKernel = b; return this; }
    public BenchmarkConfig dspBatchedGemm(boolean b) { this.dspBatchedGemm = b; return this; }
    public BenchmarkConfig useDraftModel(boolean b) { this.useDraftModel = b; return this; }
    public BenchmarkConfig draftModelK(int n) { this.draftModelK = n; return this; }
    public BenchmarkConfig validationMode(boolean b) { this.validationMode = b; return this; }
    public BenchmarkConfig validationConfig(ValidationConfig c) { this.validationConfig = c; return this; }

    // Getters
    public String getName() { return name; }
    public GraphExecutionMode getExecutionMode() { return executionMode; }
    public String getTritonIncludeTypes() { return tritonIncludeTypes; }
    public boolean isTritonSectionFusion() { return tritonSectionFusion; }
    public boolean isTritonGraphCapture() { return tritonGraphCapture; }
    public boolean isTritonConsolidatedArgTable() { return tritonConsolidatedArgTable; }
    public boolean isTritonArgDirtyTracking() { return tritonArgDirtyTracking; }
    public boolean isTritonCooperativeLaunch() { return tritonCooperativeLaunch; }
    public int getTritonCoopTargetBlocks() { return tritonCoopTargetBlocks; }
    public boolean isTritonCompileAll() { return tritonCompileAll; }
    public String getTritonExcludeOps() { return tritonExcludeOps; }
    public boolean isTritonSkipKernels() { return tritonSkipKernels; }
    public boolean isTritonVerifyKernels() { return tritonVerifyKernels; }
    public boolean isTritonVerifyFullSnapshot() { return tritonVerifyFullSnapshot; }
    public boolean isTritonForceRecapture() { return tritonForceRecapture; }
    public boolean isTritonVerbose() { return tritonVerbose; }
    public boolean isTritonDumpSections() { return tritonDumpSections; }
    public int getTritonNumWarps() { return tritonNumWarps; }
    public int getTritonNumStages() { return tritonNumStages; }
    public int getTritonNumCTAs() { return tritonNumCTAs; }
    public int getTritonMaxNreg() { return tritonMaxNreg; }
    public int getTritonAttentionBlockN() { return tritonAttentionBlockN; }
    public boolean isTritonEnableFpFusion() { return tritonEnableFpFusion; }
    public int getTritonMaxSubsegmentOps() { return tritonMaxSubsegmentOps; }
    public int getTritonMaxSubsegmentSections() { return tritonMaxSubsegmentSections; }
    public boolean isTritonAllowFallbackCapture() { return tritonAllowFallbackCapture; }
    public int getTritonBuildThreads() { return tritonBuildThreads; }
    public String getTritonProfile() { return tritonProfile; }
    
    // Triton segment fusion getters
    public boolean isTritonFuseIdentityShapes() { return tritonFuseIdentityShapes; }
    public boolean isTritonFuseCastChains() { return tritonFuseCastChains; }
    public boolean isTritonFuseTrivialGather() { return tritonFuseTrivialGather; }
    public boolean isTritonSpecializePermuteSeq1() { return tritonSpecializePermuteSeq1; }
    public boolean isTritonEliminateConcatSplitPairs() { return tritonEliminateConcatSplitPairs; }
    public boolean isTritonFusedMatmul() { return tritonFusedMatmul; }
    public boolean isTritonFusionScoring() { return tritonFusionScoring; }
    public float getTritonFusionMinScore() { return tritonFusionMinScore; }
    
    public int getCaptureMinExec() { return captureMinExec; }
    public int getMaxTokens() { return maxTokens; }
    public double getMinDiversityPct() { return minDiversityPct; }
    public String[] getExpectedSubstrings() { return expectedSubstrings; }
    public boolean isExpectStructuralTags() { return expectStructuralTags; }
    public boolean isCublasTf32() { return cublasTf32; }
    public boolean isTritonTf32() { return tritonTf32; }
    public boolean isCublasCaptureWorkspace() { return cublasCaptureWorkspace; }
    public boolean isDspCastElimination() { return dspCastElimination; }
    public boolean isDspCastSinkMatmul() { return dspCastSinkMatmul; }
    public boolean isDspFp16Compute() { return dspFp16Compute; }
    public boolean isDspBatchZero() { return dspBatchZero; }
    public boolean isDspBatchZeroKernel() { return dspBatchZeroKernel; }
    public boolean isDspBatchedGemm() { return dspBatchedGemm; }
    public boolean isUseDraftModel() { return useDraftModel; }
    public int getDraftModelK() { return draftModelK; }
    public boolean isValidationMode() { return validationMode; }
    public ValidationConfig getValidationConfig() { return validationConfig; }

    public boolean isTriton() {
        return !tritonIncludeTypes.isEmpty();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(name);
        if (executionMode != null) sb.append(" mode=").append(executionMode);
        if (isTriton()) sb.append(" types=").append(tritonIncludeTypes);
        if (tritonSectionFusion) sb.append(" fusion");
        if (tritonGraphCapture) sb.append(" gc");
        if (tritonConsolidatedArgTable) sb.append(" argTable");
        if (tritonArgDirtyTracking) sb.append(" dirtyTrack");
        if (tritonCooperativeLaunch) sb.append(" coopLaunch");
        if (tritonCompileAll) sb.append(" compileAll");
        if (!tritonExcludeOps.isEmpty()) sb.append(" exclude=").append(tritonExcludeOps);
        if (!tritonEnableFpFusion) sb.append(" noFpFusion");
        if (tritonNumWarps > 0) sb.append(" warps=").append(tritonNumWarps);
        if (tritonNumStages > 0) sb.append(" stages=").append(tritonNumStages);
        if (tritonMaxNreg > 0) sb.append(" maxNreg=").append(tritonMaxNreg);
        if (tritonAttentionBlockN > 0) sb.append(" attnBlockN=").append(tritonAttentionBlockN);
        if (tritonMaxSubsegmentOps > 0) sb.append(" maxSubOps=").append(tritonMaxSubsegmentOps);
        if (tritonMaxSubsegmentSections > 0) sb.append(" maxSubSections=").append(tritonMaxSubsegmentSections);
        if (tritonAllowFallbackCapture) sb.append(" fallbackCapture");
        if (dspCastElimination) sb.append(" castElim");
        if (dspCastSinkMatmul) sb.append(" castSinkMatmul");
        if (dspFp16Compute) sb.append(" fp16compute");
        if (dspBatchZero) sb.append(" batchZero");
        if (dspBatchZeroKernel) sb.append(" batchZeroKernel");
        if (dspBatchedGemm) sb.append(" batchedGemm");
        if (cublasTf32) sb.append(" cublasTf32");
        if (tritonTf32) sb.append(" tritonTf32");
        if (!cublasCaptureWorkspace) sb.append(" noWorkspace");
        if (useDraftModel) sb.append(" draftModel(K=").append(draftModelK).append(")");
        sb.append(" tokens=").append(maxTokens);
        return sb.toString();
    }
}
