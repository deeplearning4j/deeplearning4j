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

#include <system/config/DspConfig.h>
#include <system/config/EnvHelper.h>

namespace sd {
namespace config {

DspConfig::DspConfig() {
  // Defaults set via member initializers.
}

void DspConfig::initFromEnvironment() {
  // Batch operations
  {
    int v = readBoolEnvTriState("ND4J_DSP_BATCH_ZERO");
    if (v >= 0) setBatchZero(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_BATCH_ZERO_VERBOSE");
    if (v >= 0) setBatchZeroVerbose(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_BATCH_ZERO_GAP_ONLY");
    if (v >= 0) setBatchZeroGapOnly(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_BATCH_ZERO_KERNEL");
    if (v >= 0) setBatchZeroKernel(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_BATCHED_GEMM");
    if (v >= 0) setBatchedGemm(v == 1);
  }

  // Optimization flags
  {
    int v = readBoolEnvTriState("ND4J_DSP_CAST_ELIMINATION");
    if (v >= 0) setCastElimination(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_MATMUL_SEGMENTATION");
    if (v >= 0) setMatmulSegmentation(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_FP16_COMPUTE");
    if (v >= 0) setFp16Compute(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_CUBLAS_TF32");
    if (v >= 0) setCublasTf32Enabled(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_CUBLAS_CAPTURE_WORKSPACE");
    if (v >= 0) setCublasCaptureWorkspace(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_CAST_SINK_MATMUL");
    if (v >= 0) setCastSinkMatmul(v == 1);
  }

  // Symbolic shapes
  {
    int v = readBoolEnvTriState("ND4J_DSP_SYMBOLIC_SHAPES");
    if (v >= 0) setSymbolicShapes(v == 1);
  }
  // ND4J_DSP_SYMBOLIC_SHAPE_WARMUP removed — warmup is a compile-time constant (2)
  // in DspConfig::kSymbolicShapeWarmup. Runtime tuning was never used.

  // Frozen-shape transition
  {
    int v = readBoolEnvTriState("ND4J_DSP_FREEZE_MERGE_SEGMENTS");
    if (v >= 0) setFreezeMergeSegments(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_FREEZE_RECOMPILE");
    if (v >= 0) setFreezeRecompile(v == 1);
  }

  // Capture buffer pool
  {
    int v = readBoolEnvTriState("ND4J_DSP_CAPTURE_POOL_ENABLED");
    if (v >= 0) setCapturePoolEnabled(v == 1);
  }
  {
    int64_t v = readInt64Env("ND4J_DSP_CAPTURE_POOL_MAX_BYTES", -1);
    if (v > 0) setCapturePoolMaxBytes(v);
  }

  // OOM retry
  {
    int v = readIntEnv("ND4J_DSP_CAPTURE_OOM_MAX_RETRIES", -1);
    if (v >= 0) setCaptureOomMaxRetries(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_CAPTURE_OOM_RETRY_INTERVAL", -1);
    if (v >= 1) setCaptureOomRetryInterval(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_CUBLAS_WORKSPACE_MB", -1);
    if (v > 0) setCublasWorkspaceMb(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_GRAPH_METADATA_SAFETY_MB", -1);
    if (v >= 0) setGraphMetadataSafetyMb(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_PROACTIVE_EVICT");
    if (v >= 0) setProactiveEvictBeforeCapture(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_LRU_EVICTION");
    if (v >= 0) setLruEviction(v == 1);
  }

  // Diagnostics
  {
    std::string v = readStringEnv("ND4J_DSP_DIAGNOSTICS");
    if (!v.empty()) setDiagnosticsCategories(v);
  }
  {
    std::string v = readStringEnv("ND4J_DSP_DIAGNOSTICS_LEVEL");
    if (!v.empty()) setDiagnosticsLevel(v);
  }
  {
    std::string v = readStringEnv("ND4J_DSP_DIAGNOSTICS_FILE");
    if (!v.empty()) setDiagnosticsFile(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_TRACE");
    if (v >= 0) setDiagnosticsTrace(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_EXECUTION_TIMING");
    if (v >= 0) setDiagnosticsTiming(v == 1);
  }
  {
    int v = readBoolEnvTriState("ND4J_DSP_NATIVE_DUMP_OUTPUTS");
    if (v >= 0) setDiagnosticsNativeDump(v == 1);
  }

  // Replay graph cache
  {
    std::string v = readStringEnv("ND4J_REPLAY_CACHE_DIR");
    if (!v.empty()) setReplayCacheDir(v);
  }
  {
    int v = readBoolEnvTriState("ND4J_REPLAY_CACHE_ENABLED");
    if (v >= 0) setReplayCacheEnabled(v == 1);
  }
  {
    int v = readIntEnv("ND4J_DSP_TRACE_SLOT", -1);
    if (v >= 0) setTraceSlot(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_TRACE_EXT_INPUT", -1);
    if (v >= 0) setTraceExtInput(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_DIAG_EXEC_LIMIT", 0);
    if (v > 0) setDiagExecLimit(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_DIAG_DETAIL_LIMIT", 20);
    if (v > 0) setDiagDetailLimit(v);
  }

  // Shape-keyed plan cache
  {
    float v = readFloatEnv("ND4J_DSP_PLAN_CACHE_BUDGET_FRACTION", -1.0f);
    if (v >= 0.0f && v <= 1.0f) setPlanCacheBudgetFraction(v);
  }
  {
    int v = readIntEnv("ND4J_DSP_PLAN_CACHE_MAX_PLANS", -1);
    if (v > 0) setPlanCacheMaxPlans(v);
  }
}

}  // namespace config
}  // namespace sd
