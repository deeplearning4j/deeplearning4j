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

#ifndef LIBND4J_DSP_CONFIG_H
#define LIBND4J_DSP_CONFIG_H

#include <system/common.h>

#include <atomic>

namespace sd {
namespace config {

/**
 * DSP (DynamicShapePlan) configuration: batch-zero, batched GEMM,
 * cast elimination, matmul segmentation, FP16 compute, cuBLAS settings,
 * symbolic shapes, frozen-shape transitions, capture pool, and OOM retry.
 */
class SD_LIB_EXPORT DspConfig {
 private:
  // Batch operations
  std::atomic<bool> _batchZero{false};
  std::atomic<bool> _batchZeroVerbose{false};
  std::atomic<bool> _batchZeroGapOnly{true};
  std::atomic<bool> _batchZeroKernel{false};
  std::atomic<bool> _batchedGemm{false};

  // Pool trim
  std::atomic<int> _trimInterval{5};

  // Optimization flags
  std::atomic<bool> _castElimination{true};
  std::atomic<bool> _matmulSegmentation{true};
  std::atomic<bool> _fp16Compute{false};
  std::atomic<bool> _cublasTf32Enabled{false};
  std::atomic<bool> _cublasCaptureWorkspace{true};
  std::atomic<bool> _castSinkMatmul{false};

  // Symbolic shape ranges
  std::atomic<bool> _symbolicShapes{true};
  std::atomic<int> _symbolicShapeWarmup{2};

  // Frozen-shape transition
  std::atomic<bool> _freezeMergeSegments{true};
  std::atomic<bool> _freezeRecompile{false};

  // Capture buffer pool
  std::atomic<bool> _capturePoolEnabled{true};
  std::atomic<long long> _capturePoolMaxBytes{1073741824LL};

  // OOM retry
  std::atomic<int> _captureOomMaxRetries{3};
  std::atomic<int> _captureOomRetryInterval{4};
  std::atomic<int> _cublasWorkspaceMb{256};
  std::atomic<int> _graphMetadataSafetyMb{16};
  std::atomic<bool> _proactiveEvictBeforeCapture{true};
  std::atomic<bool> _lruEviction{true};

 public:
  DspConfig();

  // --- Batch operations ---
  bool batchZero() { return _batchZero.load(); }
  void setBatchZero(bool v) { _batchZero.store(v); }
  bool batchZeroVerbose() { return _batchZeroVerbose.load(); }
  void setBatchZeroVerbose(bool v) { _batchZeroVerbose.store(v); }
  bool batchZeroGapOnly() { return _batchZeroGapOnly.load(); }
  void setBatchZeroGapOnly(bool v) { _batchZeroGapOnly.store(v); }
  bool batchZeroKernel() { return _batchZeroKernel.load(); }
  void setBatchZeroKernel(bool v) { _batchZeroKernel.store(v); }
  bool batchedGemm() { return _batchedGemm.load(); }
  void setBatchedGemm(bool v) { _batchedGemm.store(v); }

  // --- Pool trim ---
  int trimInterval() { return _trimInterval.load(); }
  void setTrimInterval(int v) { _trimInterval.store(v); }

  // --- Optimization flags ---
  bool castElimination() { return _castElimination.load(); }
  void setCastElimination(bool v) { _castElimination.store(v); }
  bool matmulSegmentation() { return _matmulSegmentation.load(); }
  void setMatmulSegmentation(bool v) { _matmulSegmentation.store(v); }
  bool fp16Compute() { return _fp16Compute.load(); }
  void setFp16Compute(bool v) { _fp16Compute.store(v); }
  bool cublasTf32Enabled() { return _cublasTf32Enabled.load(); }
  void setCublasTf32Enabled(bool v) { _cublasTf32Enabled.store(v); }
  bool cublasCaptureWorkspace() { return _cublasCaptureWorkspace.load(); }
  void setCublasCaptureWorkspace(bool v) { _cublasCaptureWorkspace.store(v); }
  bool castSinkMatmul() { return _castSinkMatmul.load(); }
  void setCastSinkMatmul(bool v) { _castSinkMatmul.store(v); }

  // --- Symbolic shapes ---
  bool symbolicShapes() { return _symbolicShapes.load(); }
  void setSymbolicShapes(bool v) { _symbolicShapes.store(v); }
  int symbolicShapeWarmup() { return _symbolicShapeWarmup.load(); }
  void setSymbolicShapeWarmup(int v) { _symbolicShapeWarmup.store(v); }

  // --- Frozen-shape transition ---
  bool freezeMergeSegments() { return _freezeMergeSegments.load(); }
  void setFreezeMergeSegments(bool v) { _freezeMergeSegments.store(v); }
  bool freezeRecompile() { return _freezeRecompile.load(); }
  void setFreezeRecompile(bool v) { _freezeRecompile.store(v); }

  // --- Capture buffer pool ---
  bool capturePoolEnabled() { return _capturePoolEnabled.load(); }
  void setCapturePoolEnabled(bool v) { _capturePoolEnabled.store(v); }
  long long capturePoolMaxBytes() { return _capturePoolMaxBytes.load(); }
  void setCapturePoolMaxBytes(long long v) { _capturePoolMaxBytes.store(v); }

  // --- OOM retry ---
  int captureOomMaxRetries() { return _captureOomMaxRetries.load(); }
  void setCaptureOomMaxRetries(int v) { _captureOomMaxRetries.store(v); }
  int captureOomRetryInterval() { return _captureOomRetryInterval.load(); }
  void setCaptureOomRetryInterval(int v) { _captureOomRetryInterval.store(v); }
  int cublasWorkspaceMb() { return _cublasWorkspaceMb.load(); }
  void setCublasWorkspaceMb(int v) { _cublasWorkspaceMb.store(v); }
  int graphMetadataSafetyMb() { return _graphMetadataSafetyMb.load(); }
  void setGraphMetadataSafetyMb(int v) { _graphMetadataSafetyMb.store(v); }
  bool proactiveEvictBeforeCapture() { return _proactiveEvictBeforeCapture.load(); }
  void setProactiveEvictBeforeCapture(bool v) { _proactiveEvictBeforeCapture.store(v); }
  bool lruEviction() { return _lruEviction.load(); }
  void setLruEviction(bool v) { _lruEviction.store(v); }

  /**
   * Initialize from environment variables.
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_DSP_CONFIG_H
