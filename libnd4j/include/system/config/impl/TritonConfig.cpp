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

#include <system/config/TritonConfig.h>
#include <system/config/EnvHelper.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>

namespace sd {
namespace config {

TritonConfig::TritonConfig() {
  // Defaults set via member initializers.
}

void TritonConfig::setBuildThreads(int threads) {
  if (threads > 0 && threads <= 16) _buildThreads.store(threads);
}

void TritonConfig::setCoopTargetBlocks(int blocks) {
  _coopTargetBlocks.store(blocks < 0 ? 0 : blocks);
}

void TritonConfig::setMaxSubsegmentOps(int ops) {
  _maxSubsegmentOps.store(ops < 0 ? 0 : ops);
}

void TritonConfig::setMaxSubsegmentSections(int sections) {
  _maxSubsegmentSections.store(sections < 0 ? 0 : sections);
}

void TritonConfig::setNumWarps(int warps) {
  _numWarps.store(warps < 0 ? 0 : warps);
}

void TritonConfig::setNumStages(int stages) {
  _numStages.store(stages < 0 ? 0 : stages);
}

void TritonConfig::setNumCTAs(int ctas) {
  _numCTAs.store(ctas < 1 ? 1 : ctas);
}

void TritonConfig::setMaxNreg(int v) {
  _maxNreg.store(v < 0 ? 0 : v);
}

void TritonConfig::setAttentionBlockN(int v) {
  _attentionBlockN.store(v < 0 ? 0 : v);
}

bool TritonConfig::isExcludedOp(const std::string& opName) const {
  if (_excludeOps.empty()) return false;
  std::string lower;
  lower.reserve(opName.size());
  for (char c : opName) lower += static_cast<char>(std::tolower(c));

  size_t start = 0;
  while (start < _excludeOps.size()) {
    size_t end = _excludeOps.find(',', start);
    if (end == std::string::npos) end = _excludeOps.size();
    size_t s = start, e = end;
    while (s < e && std::isspace(_excludeOps[s])) s++;
    while (e > s && std::isspace(_excludeOps[e - 1])) e--;
    if (e > s) {
      std::string token;
      token.reserve(e - s);
      for (size_t i = s; i < e; i++) token += static_cast<char>(std::tolower(_excludeOps[i]));
      if (token == lower) return true;
    }
    start = end + 1;
  }
  return false;
}

void TritonConfig::initFromEnvironment() {
  // Build settings
  {
    int v = readIntEnv("ND4J_TRITON_BUILD_THREADS", -1);
    if (v > 0) setBuildThreads(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_CACHE_ENABLE");
    if (v >= 0) setCacheEnabled(v == 1);
  }
  {
    int64_t v = readInt64Env("ND4J_TRITON_MODULE_RESIDENCY_BUDGET_BYTES", -1);
    if (v >= 0) setModuleResidencyBudgetBytes(v);
  }
  {
    int64_t v = readInt64Env("ND4J_TRITON_MODULE_RESIDENCY_WARN_BYTES", -1);
    if (v >= 0) setModuleResidencyWarnBytes(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_BATCH_PRELOAD_MODULES");
    if (v >= 0) setBatchPreloadModules(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_COOPERATIVE_LAUNCH");
    if (v >= 0) setCooperativeLaunch(v == 1);
  }
  {
    int v = readIntEnv("ND4J_TRITON_COOP_TARGET_BLOCKS", -1);
    if (v >= 0) setCoopTargetBlocks(v);
  }
  {
    int v = readIntEnv("ND4J_TRITON_MAX_SUBSEGMENT_OPS", -1);
    if (v >= 0) setMaxSubsegmentOps(v);
  }
  {
    int v = readIntEnv("ND4J_TRITON_MAX_SUBSEGMENT_SECTIONS", -1);
    if (v >= 0) setMaxSubsegmentSections(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_VERBOSE");
    if (v >= 0) setVerbose(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_DUMP_SECTIONS");
    if (v >= 0) setDumpSections(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_DUMP_ARGS");
    if (v >= 0) setDumpArgs(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_LOG_ALL_PATTERNS");
    if (v >= 0) setLogAllPatterns(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_ALWAYS_COMPILE");
    if (v >= 0) setAlwaysCompile(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_INVALIDATE_ON_PLAN_FREE");
    if (v >= 0) setInvalidateOnPlanFree(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_KERNEL_DUMP");
    if (v >= 0) setKernelDump(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_KERNEL_OVERRIDE");
    if (v >= 0) setKernelOverride(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_ENABLE_FP_FUSION");
    if (v >= 0) setEnableFpFusion(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_DISABLE_LINE_INFO");
    if (v >= 0) setDisableLineInfo(v == 1);
  }

  // Kernel tuning
  {
    int v = readIntEnv("ND4J_TRITON_NUM_WARPS", -1);
    if (v >= 0) setNumWarps(v);
  }
  {
    int v = readIntEnv("ND4J_TRITON_NUM_STAGES", -1);
    if (v >= 0) setNumStages(v);
  }
  {
    int v = readIntEnv("ND4J_TRITON_NUM_CTAS", -1);
    if (v >= 0) setNumCTAs(v);
  }
  {
    int v = readIntEnv("ND4J_TRITON_MAXNREG", -1);
    if (v >= 0) setMaxNreg(v);
  }
  {
    int v = readIntEnv("ND4J_TRITON_ATTENTION_BLOCK_N", -1);
    if (v >= 0) setAttentionBlockN(v);
  }

  // Directories
  {
    std::string v = readStringEnv("ND4J_TRITON_CACHE_DIR");
    if (!v.empty()) setCacheDir(v);
  }
  {
    std::string v = readStringEnv("ND4J_TRITON_DUMP_DIR");
    if (!v.empty()) setDumpDir(v);
  }
  {
    std::string v = readStringEnv("ND4J_TRITON_OVERRIDE_DIR");
    if (!v.empty()) setOverrideDir(v);
  }
  {
    std::string v = readStringEnv("ND4J_TRITON_OVERRIDE_ARCH");
    if (!v.empty()) setOverrideArch(v);
  }

  // Default dump/cache dirs to ~/.kompile/cache/triton/ when not explicitly set.
  // This prevents polluting /tmp with hundreds of MLIR diagnostic files.
  if (_dumpDir.empty() || _cacheDir.empty()) {
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE"); // Windows fallback
    if (home) {
      std::string defaultDir = std::string(home) + "/.kompile/cache/triton/";
      std::error_code ec;
      std::filesystem::create_directories(defaultDir, ec);
      if (!ec) {
        if (_dumpDir.empty()) setDumpDir(defaultDir);
        if (_cacheDir.empty()) setCacheDir(defaultDir);
      }
    }
  }

  // CUDA graph integration
  {
    int v = readBoolEnvTriState("ND4J_TRITON_ALLOW_FALLBACK_CAPTURE");
    if (v >= 0) setAllowFallbackCapture(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_GRAPH_CAPTURE");
    if (v >= 0) setGraphCapture(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_DUMP_GRAPH_DOT");
    if (v >= 0) setDumpGraphDot(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_GRAPH_CTX_PUSH");
    if (v >= 0) setGraphCtxPush(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_GRAPH_REINSTANTIATE");
    if (v >= 0) setGraphReinstantiate(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_GRAPH_AUTOFREE");
    if (v >= 0) setGraphAutoFree(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_GRAPH_DOT_VERBOSE");
    if (v >= 0) setGraphDotVerbose(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_MERGED_CAPTURE_THROUGH_VIEWS");
    if (v >= 0) setMergedCaptureThroughViews(v == 1);
  }

  // Compilation scope
  {
    int v = readBoolEnvTriState("ND4J_TRITON_COMPILE_ALL");
    if (v >= 0) setCompileAll(v == 1);
  }
  {
    std::string v = readStringEnv("ND4J_TRITON_EXCLUDE_OPS");
    if (!v.empty()) setExcludeOps(v);
  }
  {
    std::string v = readStringEnv("ND4J_TRITON_INCLUDE_TYPES");
    if (!v.empty()) setIncludeTypes(v);
  }

  // Segment fusion
  {
    int v = readBoolEnvTriState("ND4J_TRITON_FUSE_IDENTITY_SHAPES");
    if (v >= 0) setFuseIdentityShapes(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_FUSE_CAST_CHAINS");
    if (v >= 0) setFuseCastChains(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_SPECIALIZE_PERMUTE_SEQ1");
    if (v >= 0) setSpecializePermuteSeq1(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_FUSED_MATMUL");
    if (v >= 0) setFusedMatmul(v == 1);
  }

  // Debugging
  {
    int v = readBoolEnvTriState("ND4J_TRITON_SKIP_KERNELS");
    if (v >= 0) setSkipKernels(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_VERIFY_KERNELS");
    if (v >= 0) setVerifyKernels(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_VERIFY_KEEP_NATIVE");
    if (v >= 0) setVerifyKeepNative(v == 1);
  }
  {
    int v = readIntEnv("ND4J_TRITON_MAX_SUB_KERNEL_INDEX", -2);
    if (v >= -1) setMaxSubKernelIndex(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_VERIFY_FULL_SNAPSHOT");
    if (v >= 0) setVerifyFullSnapshot(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_FORCE_RECAPTURE");
    if (v >= 0) setForceRecapture(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_WARMUP_ONLY");
    if (v >= 0) setWarmupOnly(v == 1);
  }
  {
    int v = readIntEnv("ND4J_TRITON_CAPTURE_MIN_EXEC", -1);
    if (v >= 0) setCaptureMinExec(v);
  }

  // Optimization flags
  {
    int v = readBoolEnvTriState("ND4J_TRITON_TF32");
    if (v >= 0) setTf32Enabled(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_CONSOLIDATED_ARG_TABLE");
    if (v >= 0) setConsolidatedArgTable(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_ARG_DIRTY_TRACKING");
    if (v >= 0) setArgDirtyTracking(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_SECTION_FUSION");
    if (v >= 0) setSectionFusion(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_TRITON_FUSION_SCORING");
    if (v >= 0) setFusionScoring(v == 1);
  }
  {
    float v = readFloatEnv("ND4J_TRITON_FUSION_MIN_SCORE", -1.0f);
    if (v >= 0.0f) setFusionMinScore(v);
  }
}

}  // namespace config
}  // namespace sd
