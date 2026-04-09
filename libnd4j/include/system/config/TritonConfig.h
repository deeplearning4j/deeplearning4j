/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_TRITON_CONFIG_H
#define LIBND4J_TRITON_CONFIG_H

#include <system/common.h>

#include <atomic>
#include <string>

namespace sd {
namespace config {

/**
 * Triton GPU compilation configuration: build threads, caching, cooperative
 * launch, kernel tuning (warps/stages/CTAs/registers), CUDA graph integration,
 * segment fusion, debugging flags, section fusion scoring, and op exclusions.
 */
class SD_LIB_EXPORT TritonConfig {
 private:
  // Build settings
  std::atomic<int> _buildThreads{8};
  std::atomic<bool> _cacheEnabled{true};
  std::atomic<bool> _cooperativeLaunch{false};
  std::atomic<int> _coopTargetBlocks{0};
  std::atomic<int> _maxSubsegmentOps{0};
  std::atomic<int> _maxSubsegmentSections{0};
  std::atomic<bool> _verbose{false};
  std::atomic<bool> _dumpSections{false};
  std::atomic<bool> _dumpArgs{false};
  std::atomic<bool> _logAllPatterns{false};
  std::atomic<bool> _alwaysCompile{false};
  std::atomic<bool> _kernelDump{false};
  std::atomic<bool> _kernelOverride{false};
  std::atomic<bool> _invalidateOnPlanFree{false};

  // Kernel tuning
  std::atomic<int> _numWarps{0};
  std::atomic<int> _numStages{0};
  std::atomic<int> _numCTAs{1};
  std::atomic<int> _maxNreg{0};
  std::atomic<int> _attentionBlockN{0};
  std::atomic<bool> _enableFpFusion{true};
  std::atomic<bool> _disableLineInfo{false};

  // Directories
  std::string _cacheDir;
  std::string _dumpDir;
  std::string _overrideDir;
  std::string _overrideArch;

  // CUDA graph integration
  std::atomic<bool> _allowFallbackCapture{true};
  std::atomic<bool> _graphCapture{true};
  std::atomic<bool> _dumpGraphDot{false};
  std::atomic<bool> _graphCtxPush{false};
  std::atomic<bool> _graphReinstantiate{false};
  std::atomic<bool> _graphAutoFree{false};
  std::atomic<bool> _graphDotVerbose{false};

  // Compilation scope
  std::atomic<bool> _compileAll{false};
  std::string _excludeOps;
  std::string _includeTypes;

  // Segment fusion
  std::atomic<bool> _fuseIdentityShapes{true};
  std::atomic<bool> _fuseCastChains{true};
  std::atomic<bool> _specializePermuteSeq1{true};
  std::atomic<bool> _fusedMatmul{false};
  std::atomic<bool> _fuseAttentionNeighborhoods{true};

  // Debugging
  std::atomic<bool> _skipKernels{false};
  std::atomic<bool> _verifyKernels{false};
  std::atomic<bool> _verifyKeepNative{false};
  std::atomic<int> _maxSubKernelIndex{-1};
  std::atomic<bool> _verifyFullSnapshot{false};
  std::atomic<bool> _forceRecapture{false};
  std::atomic<bool> _warmupOnly{false};
  std::atomic<int> _captureMinExec{2};

  // Optimization flags shared between Triton and cuBLAS
  std::atomic<bool> _tf32Enabled{false};
  std::atomic<bool> _consolidatedArgTable{false};
  std::atomic<bool> _argDirtyTracking{false};
  std::atomic<bool> _sectionFusion{true};
  std::atomic<bool> _fusionScoring{true};
  std::atomic<float> _fusionMinScore{5.0f};

 public:
  TritonConfig();

  // --- Build settings ---
  int buildThreads() { return _buildThreads.load(); }
  void setBuildThreads(int threads);
  bool cacheEnabled() { return _cacheEnabled.load(); }
  void setCacheEnabled(bool v) { _cacheEnabled.store(v); }
  bool cooperativeLaunch() { return _cooperativeLaunch.load(); }
  void setCooperativeLaunch(bool v) { _cooperativeLaunch.store(v); }
  int coopTargetBlocks() { return _coopTargetBlocks.load(); }
  void setCoopTargetBlocks(int blocks);
  int maxSubsegmentOps() { return _maxSubsegmentOps.load(); }
  void setMaxSubsegmentOps(int ops);
  int maxSubsegmentSections() { return _maxSubsegmentSections.load(); }
  void setMaxSubsegmentSections(int sections);
  bool isVerbose() { return _verbose.load(); }
  void setVerbose(bool v) { _verbose.store(v); }
  bool dumpSections() { return _dumpSections.load(); }
  void setDumpSections(bool v) { _dumpSections.store(v); }
  bool dumpArgs() { return _dumpArgs.load(); }
  void setDumpArgs(bool v) { _dumpArgs.store(v); }
  bool logAllPatterns() { return _logAllPatterns.load(); }
  void setLogAllPatterns(bool v) { _logAllPatterns.store(v); }
  bool alwaysCompile() { return _alwaysCompile.load(); }
  void setAlwaysCompile(bool v) { _alwaysCompile.store(v); }
  bool invalidateOnPlanFree() { return _invalidateOnPlanFree.load(); }
  void setInvalidateOnPlanFree(bool v) { _invalidateOnPlanFree.store(v); }
  bool kernelDump() { return _kernelDump.load(); }
  void setKernelDump(bool v) { _kernelDump.store(v); }
  bool kernelOverride() { return _kernelOverride.load(); }
  void setKernelOverride(bool v) { _kernelOverride.store(v); }

  // --- Kernel tuning ---
  int numWarps() { return _numWarps.load(); }
  void setNumWarps(int warps);
  int numStages() { return _numStages.load(); }
  void setNumStages(int stages);
  int numCTAs() { return _numCTAs.load(); }
  void setNumCTAs(int ctas);
  int maxNreg() { return _maxNreg.load(); }
  void setMaxNreg(int v);
  int attentionBlockN() { return _attentionBlockN.load(); }
  void setAttentionBlockN(int v);
  bool enableFpFusion() { return _enableFpFusion.load(); }
  void setEnableFpFusion(bool v) { _enableFpFusion.store(v); }
  bool disableLineInfo() { return _disableLineInfo.load(); }
  void setDisableLineInfo(bool v) { _disableLineInfo.store(v); }

  // --- Directories ---
  std::string cacheDir() const { return _cacheDir; }
  void setCacheDir(const std::string& v) { _cacheDir = v; }
  std::string dumpDir() const { return _dumpDir; }
  void setDumpDir(const std::string& v) { _dumpDir = v; }
  std::string overrideDir() const { return _overrideDir; }
  void setOverrideDir(const std::string& v) { _overrideDir = v; }
  std::string overrideArch() const { return _overrideArch; }
  void setOverrideArch(const std::string& v) { _overrideArch = v; }

  // --- CUDA graph integration ---
  bool allowFallbackCapture() { return _allowFallbackCapture.load(); }
  void setAllowFallbackCapture(bool v) { _allowFallbackCapture.store(v); }
  bool graphCapture() { return _graphCapture.load(); }
  void setGraphCapture(bool v) { _graphCapture.store(v); }
  bool dumpGraphDot() { return _dumpGraphDot.load(); }
  void setDumpGraphDot(bool v) { _dumpGraphDot.store(v); }
  bool graphCtxPush() { return _graphCtxPush.load(); }
  void setGraphCtxPush(bool v) { _graphCtxPush.store(v); }
  bool graphReinstantiate() { return _graphReinstantiate.load(); }
  void setGraphReinstantiate(bool v) { _graphReinstantiate.store(v); }
  bool graphAutoFree() { return _graphAutoFree.load(); }
  void setGraphAutoFree(bool v) { _graphAutoFree.store(v); }
  bool graphDotVerbose() { return _graphDotVerbose.load(); }
  void setGraphDotVerbose(bool v) { _graphDotVerbose.store(v); }

  // --- Compilation scope ---
  bool compileAll() { return _compileAll.load(); }
  void setCompileAll(bool v) { _compileAll.store(v); }
  std::string excludeOps() const { return _excludeOps; }
  void setExcludeOps(const std::string& v) { _excludeOps = v; }
  bool isExcludedOp(const std::string& opName) const;
  std::string includeTypes() const { return _includeTypes; }
  void setIncludeTypes(const std::string& v) { _includeTypes = v; }

  // --- Segment fusion ---
  bool fuseIdentityShapes() { return _fuseIdentityShapes.load(); }
  void setFuseIdentityShapes(bool v) { _fuseIdentityShapes.store(v); }
  bool fuseCastChains() { return _fuseCastChains.load(); }
  void setFuseCastChains(bool v) { _fuseCastChains.store(v); }
  bool specializePermuteSeq1() { return _specializePermuteSeq1.load(); }
  void setSpecializePermuteSeq1(bool v) { _specializePermuteSeq1.store(v); }
  bool fusedMatmul() { return _fusedMatmul.load(); }
  void setFusedMatmul(bool v) { _fusedMatmul.store(v); }
  bool fuseAttentionNeighborhoods() { return _fuseAttentionNeighborhoods.load(); }
  void setFuseAttentionNeighborhoods(bool v) { _fuseAttentionNeighborhoods.store(v); }

  // --- Debugging ---
  bool skipKernels() { return _skipKernels.load(); }
  void setSkipKernels(bool v) { _skipKernels.store(v); }
  bool verifyKernels() { return _verifyKernels.load(); }
  void setVerifyKernels(bool v) { _verifyKernels.store(v); }
  bool verifyKeepNative() { return _verifyKeepNative.load(); }
  void setVerifyKeepNative(bool v) { _verifyKeepNative.store(v); }
  int maxSubKernelIndex() { return _maxSubKernelIndex.load(); }
  void setMaxSubKernelIndex(int v) { _maxSubKernelIndex.store(v); }
  bool verifyFullSnapshot() { return _verifyFullSnapshot.load(); }
  void setVerifyFullSnapshot(bool v) { _verifyFullSnapshot.store(v); }
  bool forceRecapture() { return _forceRecapture.load(); }
  void setForceRecapture(bool v) { _forceRecapture.store(v); }
  bool warmupOnly() { return _warmupOnly.load(); }
  void setWarmupOnly(bool v) { _warmupOnly.store(v); }
  int captureMinExec() { return _captureMinExec.load(); }
  void setCaptureMinExec(int v) { _captureMinExec.store(v); }

  // --- Optimization flags ---
  bool tf32Enabled() { return _tf32Enabled.load(); }
  void setTf32Enabled(bool v) { _tf32Enabled.store(v); }
  bool consolidatedArgTable() { return _consolidatedArgTable.load(); }
  void setConsolidatedArgTable(bool v) { _consolidatedArgTable.store(v); }
  bool argDirtyTracking() { return _argDirtyTracking.load(); }
  void setArgDirtyTracking(bool v) { _argDirtyTracking.store(v); }
  bool sectionFusion() { return _sectionFusion.load(); }
  void setSectionFusion(bool v) { _sectionFusion.store(v); }
  bool fusionScoring() { return _fusionScoring.load(); }
  void setFusionScoring(bool v) { _fusionScoring.store(v); }
  float fusionMinScore() { return _fusionMinScore.load(); }
  void setFusionMinScore(float v) { _fusionMinScore.store(v); }

  /**
   * Initialize from environment variables.
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_TRITON_CONFIG_H
