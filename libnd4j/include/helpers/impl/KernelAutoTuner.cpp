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

#include <helpers/KernelAutoTuner.h>
#include <ops/declarable/DeclarableOp.h>

#include <algorithm>
#include <chrono>
#include <mutex>

namespace sd {
namespace ops {
namespace platforms {

// Thread-local tuning guard - now function-local in header (isTuningRef())

Status NativeOpExecutor::execute(graph::Context& context) {
  if (_op != nullptr) {
    return _op->execute(&context);
  }
  return Status::BAD_INPUT;
}

std::string NativeOpExecutor::getName() const {
  if (_op != nullptr) {
    return _op->getOpName()->c_str();
  }
  return "unknown_native";
}

KernelAutoTuner& KernelAutoTuner::getInstance() {
  static KernelAutoTuner* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new KernelAutoTuner();
  });
  return *instance;
}

BenchmarkResult KernelAutoTuner::benchmark(graph::Context& context, KernelExecutor* executor) {
  return benchmark(context, executor, _config.warmupRuns, _config.benchmarkRuns);
}

BenchmarkResult KernelAutoTuner::benchmark(graph::Context& context, KernelExecutor* executor,
                                            int warmupRuns, int benchmarkRuns) {
  BenchmarkResult result;
  result.engine = executor->getEngine();

  if (executor == nullptr) {
    result.success = false;
    result.errorMessage = "Null executor";
    return result;
  }

  // Prevent recursive tuning
  if (TuningGuard::isTuning()) {
    result.success = false;
    result.errorMessage = "Recursive tuning detected";
    return result;
  }

  TuningGuard guard;

  try {
    // Warmup runs
    for (int i = 0; i < warmupRuns; i++) {
      auto clonedCtx = cloneContext(context);
      if (clonedCtx) {
        executor->execute(*clonedCtx);
      } else {
        executor->execute(context);
      }
    }

    // Benchmark runs
    std::vector<double> times;
    times.reserve(benchmarkRuns);

    auto startTotal = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < benchmarkRuns; i++) {
      auto clonedCtx = cloneContext(context);

      auto start = std::chrono::high_resolution_clock::now();

      Status status;
      if (clonedCtx) {
        status = executor->execute(*clonedCtx);
      } else {
        status = executor->execute(context);
      }

      auto end = std::chrono::high_resolution_clock::now();

      if (status != Status::OK) {
        result.success = false;
        result.errorMessage = "Execution failed";
        return result;
      }

      double nanos = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
      times.push_back(nanos);

      // Check time limit
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now() - startTotal)
                         .count();
      if (elapsed > _config.maxBenchmarkTimeMs) {
        break;
      }
    }

    if (times.empty()) {
      result.success = false;
      result.errorMessage = "No benchmark samples collected";
      return result;
    }

    // Calculate statistics
    result.sampleCount = static_cast<int>(times.size());
    result.minTimeNanos = *std::min_element(times.begin(), times.end());
    result.maxTimeNanos = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double t : times) {
      sum += t;
    }
    result.meanTimeNanos = sum / times.size();

    if (times.size() > 1) {
      double variance = 0.0;
      for (double t : times) {
        double diff = t - result.meanTimeNanos;
        variance += diff * diff;
      }
      result.stdDevNanos = std::sqrt(variance / (times.size() - 1));
    }

    result.success = true;

    // Update statistics
    _totalBenchmarks++;
    _totalBenchmarkTimeNanos += static_cast<int64_t>(sum);

  } catch (const std::exception& e) {
    result.success = false;
    result.errorMessage = e.what();
  } catch (...) {
    result.success = false;
    result.errorMessage = "Unknown exception";
  }

  return result;
}

std::vector<BenchmarkResult> KernelAutoTuner::benchmarkAll(
    graph::Context& context, const std::vector<KernelExecutor*>& executors) {
  std::vector<BenchmarkResult> results;
  results.reserve(executors.size());

  for (auto* executor : executors) {
    results.push_back(benchmark(context, executor));
  }

  // Sort by mean time (fastest first)
  std::sort(results.begin(), results.end());

  return results;
}

KernelExecutor* KernelAutoTuner::selectBest(graph::Context& context,
                                             const std::vector<KernelExecutor*>& executors,
                                             LongType opHash) {
  if (executors.empty()) {
    return nullptr;
  }

  if (executors.size() == 1) {
    return executors[0];
  }

  // Check if we have reliable cached data
  auto& registry = KernelPerformanceRegistry::getInstance();

  // Create signature for current context
  std::vector<const LongType*> shapes;
  std::vector<int> ranks;
  DataType dtype = DataType::FLOAT32;

  for (int i = 0; i < context.width(); i++) {
    auto arr = context.getNDArray(i);
    if (arr != nullptr) {
      shapes.push_back(arr->shapeInfo() + 1);  // Skip rank at position 0
      ranks.push_back(arr->rankOf());
      if (i == 0) {
        dtype = arr->dataType();
      }
    }
  }

  auto sig = KernelPerformanceRegistry::createSignature(opHash, shapes, ranks, dtype);

  // Check if we already have reliable data
  if (registry.hasReliableData(sig)) {
    samediff::Engine bestEngine = registry.getBestEngine(sig);

    // Find executor for best engine
    for (auto* executor : executors) {
      if (executor->getEngine() == bestEngine) {
        return executor;
      }
    }
  }

  // Need to benchmark
  auto results = benchmarkAll(context, executors);

  // Record results
  for (const auto& result : results) {
    if (result.success) {
      registry.recordPerformance(sig, result.engine, result.meanTimeNanos);
    }
  }

  // Return fastest successful executor
  for (const auto& result : results) {
    if (result.success) {
      for (auto* executor : executors) {
        if (executor->getEngine() == result.engine) {
          return executor;
        }
      }
    }
  }

  // Fallback to first executor
  return executors[0];
}

void KernelAutoTuner::forceBenchmark(LongType opHash) {
  KernelPerformanceRegistry::getInstance().clearForOp(opHash);
}

std::unique_ptr<graph::Context> KernelAutoTuner::cloneContext(graph::Context& original) {
  // For now, return nullptr to use original context
  // TODO: Implement proper context cloning for safe benchmarking
  return nullptr;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
