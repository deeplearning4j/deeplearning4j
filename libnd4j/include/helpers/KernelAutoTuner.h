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

#ifndef SD_KERNEL_AUTO_TUNER_H
#define SD_KERNEL_AUTO_TUNER_H

#include <system/BackendNamespace.h>

#include <execution/Engine.h>
#include <graph/Context.h>
#include <helpers/KernelPerformanceRegistry.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/common.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

namespace sd {
namespace ops {

// Forward declaration
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
class DeclarableOp;
SD_BACKEND_OPS_INLINE_NAMESPACE_END

namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

/**
 * Result of a benchmark run
 */
struct SD_LIB_EXPORT BenchmarkResult {
  samediff::Engine engine;
  double meanTimeNanos = 0.0;
  double minTimeNanos = 0.0;
  double maxTimeNanos = 0.0;
  double stdDevNanos = 0.0;
  int sampleCount = 0;
  bool success = false;
  std::string errorMessage;

  bool operator<(const BenchmarkResult& other) const { return meanTimeNanos < other.meanTimeNanos; }
};

/**
 * Interface for executing a kernel (allows benchmarking helpers or native ops)
 */
class SD_LIB_EXPORT KernelExecutor {
 public:
  virtual ~KernelExecutor() = default;
  virtual Status execute(graph::Context& context) = 0;
  virtual samediff::Engine getEngine() const = 0;
  virtual std::string getName() const = 0;
};

/**
 * Executor that wraps a PlatformHelper
 */
class SD_LIB_EXPORT PlatformHelperExecutor : public KernelExecutor {
 private:
  PlatformHelper* _helper;

 public:
  explicit PlatformHelperExecutor(PlatformHelper* helper) : _helper(helper) {}

  Status execute(graph::Context& context) override {
    return _helper->invokeHelper(context);
  }

  samediff::Engine getEngine() const override { return _helper->engine(); }

  std::string getName() const override { return _helper->name(); }
};

/**
 * Executor that wraps a native DeclarableOp
 */
class SD_LIB_EXPORT NativeOpExecutor : public KernelExecutor {
 private:
  DeclarableOp* _op;

 public:
  explicit NativeOpExecutor(DeclarableOp* op) : _op(op) {}

  Status execute(graph::Context& context) override;

  samediff::Engine getEngine() const override { return samediff::ENGINE_CPU; }

  std::string getName() const override;
};

/**
 * RAII guard to prevent recursive auto-tuning
 */
class SD_LIB_EXPORT TuningGuard {
 private:
  // Function-local thread_local avoids MSVC C2492 with dllexport
  static bool& isTuningRef() {
    static thread_local bool _isTuning = false;
    return _isTuning;
  }

 public:
  TuningGuard() { isTuningRef() = true; }
  ~TuningGuard() { isTuningRef() = false; }

  static bool isTuning() { return isTuningRef(); }

  // Prevent copying
  TuningGuard(const TuningGuard&) = delete;
  TuningGuard& operator=(const TuningGuard&) = delete;
};

/**
 * Auto-tuner for kernel selection.
 * Benchmarks available kernels and selects the fastest.
 */
class SD_LIB_EXPORT KernelAutoTuner {
 private:
  KernelSelectionConfig _config;
  std::atomic<int64_t> _totalBenchmarks{0};
  std::atomic<int64_t> _totalBenchmarkTimeNanos{0};

 public:
  KernelAutoTuner() : _config(KernelSelectionConfig::global()) {}
  explicit KernelAutoTuner(const KernelSelectionConfig& config) : _config(config) {}

  /**
   * Benchmark a single kernel executor
   */
  BenchmarkResult benchmark(graph::Context& context, KernelExecutor* executor);

  /**
   * Benchmark a single kernel executor with specific parameters
   */
  BenchmarkResult benchmark(graph::Context& context, KernelExecutor* executor, int warmupRuns,
                            int benchmarkRuns);

  /**
   * Benchmark all provided executors and return sorted results (fastest first)
   */
  std::vector<BenchmarkResult> benchmarkAll(graph::Context& context,
                                             const std::vector<KernelExecutor*>& executors);

  /**
   * Select the best kernel from available options
   * Uses cached performance data if available, otherwise benchmarks
   */
  KernelExecutor* selectBest(graph::Context& context, const std::vector<KernelExecutor*>& executors,
                              LongType opHash);

  /**
   * Force re-benchmark for an operation (clears cached data)
   */
  void forceBenchmark(LongType opHash);

  /**
   * Get configuration
   */
  KernelSelectionConfig& getConfig() { return _config; }
  const KernelSelectionConfig& getConfig() const { return _config; }
  void setConfig(const KernelSelectionConfig& config) { _config = config; }

  /**
   * Get statistics
   */
  int64_t getTotalBenchmarks() const { return _totalBenchmarks.load(); }
  int64_t getTotalBenchmarkTimeNanos() const { return _totalBenchmarkTimeNanos.load(); }

  /**
   * Singleton instance
   */
  static KernelAutoTuner& getInstance();

 private:
  /**
   * Clone context for safe benchmarking
   */
  std::unique_ptr<graph::Context> cloneContext(graph::Context& original);
};

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_KERNEL_AUTO_TUNER_H
