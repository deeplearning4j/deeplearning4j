/*
 *  ******************************************************************************
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
package org.nd4j.linalg.factory.config;

/**
 * Triton GPU compilation configuration: build threads, caching, cooperative
 * launch, kernel tuning, CUDA graph integration, segment fusion, debugging.
 *
 * <p>All methods have default implementations that return safe defaults,
 * so CPU-only backends need not implement them.</p>
 */
public interface TritonEnvironmentConfig {

    // Build settings
    default int tritonBuildThreads() { return 8; }
    default void setTritonBuildThreads(int threads) {}
    default boolean tritonCacheEnabled() { return true; }
    default void setTritonCacheEnabled(boolean enabled) {}
    default boolean tritonCooperativeLaunch() { return false; }
    default void setTritonCooperativeLaunch(boolean enabled) {}
    default int tritonCoopTargetBlocks() { return 0; }
    default void setTritonCoopTargetBlocks(int blocks) {}
    default int tritonMaxSubsegmentOps() { return 0; }
    default void setTritonMaxSubsegmentOps(int ops) {}
    default int tritonMaxSubsegmentSections() { return 0; }
    default void setTritonMaxSubsegmentSections(int sections) {}
    default boolean tritonVerbose() { return false; }
    default void setTritonVerbose(boolean verbose) {}
    default boolean tritonDumpSections() { return false; }
    default void setTritonDumpSections(boolean dumpSections) {}
    default boolean tritonDumpArgs() { return false; }
    default void setTritonDumpArgs(boolean dumpArgs) {}
    default boolean tritonLogAllPatterns() { return false; }
    default void setTritonLogAllPatterns(boolean logAllPatterns) {}
    default boolean tritonAlwaysCompile() { return false; }
    default void setTritonAlwaysCompile(boolean alwaysCompile) {}
    default boolean tritonInvalidateOnPlanFree() { return false; }
    default void setTritonInvalidateOnPlanFree(boolean invalidate) {}
    default boolean tritonKernelDump() { return false; }
    default void setTritonKernelDump(boolean kernelDump) {}
    default boolean tritonKernelOverride() { return false; }
    default void setTritonKernelOverride(boolean kernelOverride) {}

    // Kernel tuning
    default int tritonNumWarps() { return 0; }
    default void setTritonNumWarps(int warps) {}
    default int tritonNumStages() { return 0; }
    default void setTritonNumStages(int stages) {}
    default int tritonNumCTAs() { return 1; }
    default void setTritonNumCTAs(int ctas) {}
    default int tritonMaxNreg() { return 0; }
    default void setTritonMaxNreg(int maxNreg) {}
    default int tritonAttentionBlockN() { return 0; }
    default void setTritonAttentionBlockN(int blockN) {}
    default boolean tritonEnableFpFusion() { return true; }
    default void setTritonEnableFpFusion(boolean enableFpFusion) {}
    default boolean tritonDisableLineInfo() { return false; }
    default void setTritonDisableLineInfo(boolean disableLineInfo) {}

    // Directories
    default String tritonCacheDir() { return ""; }
    default void setTritonCacheDir(String cacheDir) {}
    default String tritonDumpDir() { return ""; }
    default void setTritonDumpDir(String dumpDir) {}
    default String tritonOverrideDir() { return ""; }
    default void setTritonOverrideDir(String overrideDir) {}
    default String tritonOverrideArch() { return ""; }
    default void setTritonOverrideArch(String overrideArch) {}

    // CUDA graph integration
    default boolean tritonAllowFallbackCapture() { return true; }
    default void setTritonAllowFallbackCapture(boolean allow) {}
    default boolean tritonGraphCapture() { return true; }
    default void setTritonGraphCapture(boolean enable) {}
    default boolean tritonDumpGraphDot() { return false; }
    default void setTritonDumpGraphDot(boolean dump) {}
    default boolean tritonMergedCaptureThroughViews() { return false; }
    default void setTritonMergedCaptureThroughViews(boolean v) {}

    // Compilation scope
    default boolean tritonCompileAll() { return false; }
    default void setTritonCompileAll(boolean v) {}
    default String tritonExcludeOps() { return ""; }
    default void setTritonExcludeOps(String ops) {}
    default String tritonIncludeTypes() { return ""; }
    default void setTritonIncludeTypes(String types) {}

    // Segment fusion
    default boolean tritonFuseIdentityShapes() { return true; }
    default void setTritonFuseIdentityShapes(boolean v) {}
    default boolean tritonFuseCastChains() { return true; }
    default void setTritonFuseCastChains(boolean v) {}
    default boolean tritonSpecializePermuteSeq1() { return true; }
    default void setTritonSpecializePermuteSeq1(boolean v) {}
    default boolean tritonFusedMatmul() { return false; }
    default void setTritonFusedMatmul(boolean v) {}
    default boolean tritonFuseAttentionNeighborhoods() { return true; }
    default void setTritonFuseAttentionNeighborhoods(boolean v) {}

    // Debugging
    default boolean tritonSkipKernels() { return false; }
    default void setTritonSkipKernels(boolean skip) {}
    default boolean tritonVerifyKernels() { return false; }
    default void setTritonVerifyKernels(boolean verify) {}
    default boolean tritonVerifyKeepNative() { return false; }
    default void setTritonVerifyKeepNative(boolean v) {}
    default int tritonMaxSubKernelIndex() { return -1; }
    default void setTritonMaxSubKernelIndex(int idx) {}
    default boolean tritonVerifyFullSnapshot() { return false; }
    default void setTritonVerifyFullSnapshot(boolean v) {}
    default boolean tritonForceRecapture() { return false; }
    default void setTritonForceRecapture(boolean v) {}
    default int tritonCaptureMinExec() { return 2; }
    default void setTritonCaptureMinExec(int v) {}

    // Optimization flags
    default boolean tritonTf32Enabled() { return false; }
    default void setTritonTf32Enabled(boolean enabled) {}
    default boolean tritonConsolidatedArgTable() { return false; }
    default void setTritonConsolidatedArgTable(boolean enabled) {}
    default boolean tritonArgDirtyTracking() { return false; }
    default void setTritonArgDirtyTracking(boolean enabled) {}
    default boolean tritonSectionFusion() { return true; }
    default void setTritonSectionFusion(boolean enabled) {}
    default boolean tritonFusionScoring() { return true; }
    default void setTritonFusionScoring(boolean enabled) {}
    default float tritonFusionMinScore() { return 5.0f; }
    default void setTritonFusionMinScore(float score) {}
}
