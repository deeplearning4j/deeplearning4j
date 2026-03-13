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
    boolean tritonEnableFpFusion = true;
    int tritonMaxSubsegmentOps = -1;
    int tritonMaxSubsegmentSections = -1;
    boolean tritonAllowFallbackCapture;
    int tritonBuildThreads = -1;
    String tritonProfile = "BALANCED";

    // DSP flags
    boolean dspCastElimination;
    boolean dspCastSinkMatmul;
    boolean dspFp16Compute;
    boolean dspBatchZero;
    boolean dspBatchZeroKernel;
    boolean dspBatchedGemm;

    // Generation
    int captureMinExec = 1;
    int maxTokens = 20;

    // Validation
    double minDiversityPct = 5.0;
    String[] expectedSubstrings;
    boolean expectStructuralTags = true;

    public static BenchmarkConfig create(String name) {
        BenchmarkConfig c = new BenchmarkConfig();
        c.name = name;
        return c;
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
    public BenchmarkConfig tritonEnableFpFusion(boolean b) { this.tritonEnableFpFusion = b; return this; }
    public BenchmarkConfig tritonMaxSubsegmentOps(int n) { this.tritonMaxSubsegmentOps = n; return this; }
    public BenchmarkConfig tritonMaxSubsegmentSections(int n) { this.tritonMaxSubsegmentSections = n; return this; }
    public BenchmarkConfig tritonAllowFallbackCapture(boolean b) { this.tritonAllowFallbackCapture = b; return this; }
    public BenchmarkConfig tritonBuildThreads(int n) { this.tritonBuildThreads = n; return this; }
    public BenchmarkConfig tritonProfile(String p) { this.tritonProfile = p; return this; }
    public BenchmarkConfig captureMinExec(int n) { this.captureMinExec = n; return this; }
    public BenchmarkConfig maxTokens(int n) { this.maxTokens = n; return this; }
    public BenchmarkConfig minDiversityPct(double d) { this.minDiversityPct = d; return this; }
    public BenchmarkConfig expectedSubstrings(String... s) { this.expectedSubstrings = s; return this; }
    public BenchmarkConfig expectStructuralTags(boolean b) { this.expectStructuralTags = b; return this; }
    public BenchmarkConfig dspCastElimination(boolean b) { this.dspCastElimination = b; return this; }
    public BenchmarkConfig dspCastSinkMatmul(boolean b) { this.dspCastSinkMatmul = b; return this; }
    public BenchmarkConfig dspFp16Compute(boolean b) { this.dspFp16Compute = b; return this; }
    public BenchmarkConfig dspBatchZero(boolean b) { this.dspBatchZero = b; return this; }
    public BenchmarkConfig dspBatchZeroKernel(boolean b) { this.dspBatchZeroKernel = b; return this; }
    public BenchmarkConfig dspBatchedGemm(boolean b) { this.dspBatchedGemm = b; return this; }

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
    public boolean isTritonEnableFpFusion() { return tritonEnableFpFusion; }
    public int getTritonMaxSubsegmentOps() { return tritonMaxSubsegmentOps; }
    public int getTritonMaxSubsegmentSections() { return tritonMaxSubsegmentSections; }
    public boolean isTritonAllowFallbackCapture() { return tritonAllowFallbackCapture; }
    public int getTritonBuildThreads() { return tritonBuildThreads; }
    public String getTritonProfile() { return tritonProfile; }
    public int getCaptureMinExec() { return captureMinExec; }
    public int getMaxTokens() { return maxTokens; }
    public double getMinDiversityPct() { return minDiversityPct; }
    public String[] getExpectedSubstrings() { return expectedSubstrings; }
    public boolean isExpectStructuralTags() { return expectStructuralTags; }
    public boolean isDspCastElimination() { return dspCastElimination; }
    public boolean isDspCastSinkMatmul() { return dspCastSinkMatmul; }
    public boolean isDspFp16Compute() { return dspFp16Compute; }
    public boolean isDspBatchZero() { return dspBatchZero; }
    public boolean isDspBatchZeroKernel() { return dspBatchZeroKernel; }
    public boolean isDspBatchedGemm() { return dspBatchedGemm; }

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
        if (tritonMaxSubsegmentOps > 0) sb.append(" maxSubOps=").append(tritonMaxSubsegmentOps);
        if (tritonMaxSubsegmentSections > 0) sb.append(" maxSubSections=").append(tritonMaxSubsegmentSections);
        if (tritonAllowFallbackCapture) sb.append(" fallbackCapture");
        if (dspCastElimination) sb.append(" castElim");
        if (dspCastSinkMatmul) sb.append(" castSinkMatmul");
        if (dspFp16Compute) sb.append(" fp16compute");
        if (dspBatchZero) sb.append(" batchZero");
        if (dspBatchZeroKernel) sb.append(" batchZeroKernel");
        if (dspBatchedGemm) sb.append(" batchedGemm");
        sb.append(" tokens=").append(maxTokens);
        return sb.toString();
    }
}
