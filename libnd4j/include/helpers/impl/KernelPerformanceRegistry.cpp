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

#include <helpers/KernelPerformanceRegistry.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace sd {
namespace ops {

KernelPerformanceRegistry& KernelPerformanceRegistry::getInstance() {
  static KernelPerformanceRegistry instance;
  return instance;
}

KernelPerformanceRegistry::~KernelPerformanceRegistry() {
  if (_dirty && !_cachePath.empty()) {
    saveToFile(_cachePath);
  }
}

int KernelPerformanceRegistry::bucketDimension(LongType dim) {
  if (dim <= 0) return 0;
  if (dim <= 16) return 16;
  if (dim <= 32) return 32;
  if (dim <= 64) return 64;
  if (dim <= 128) return 128;
  if (dim <= 256) return 256;
  if (dim <= 512) return 512;
  if (dim <= 1024) return 1024;
  if (dim <= 2048) return 2048;
  if (dim <= 4096) return 4096;
  if (dim <= 8192) return 8192;
  if (dim <= 16384) return 16384;
  // For larger dimensions, use powers of 2
  int bucket = 16384;
  while (bucket < dim && bucket < 1073741824) {
    bucket *= 2;
  }
  return bucket;
}

KernelSignature KernelPerformanceRegistry::createSignature(LongType opHash,
                                                            const std::vector<const LongType*>& shapes,
                                                            const std::vector<int>& ranks,
                                                            DataType dtype) {
  KernelSignature sig;
  sig.opHash = opHash;
  sig.dataType = dtype;

  // Create bucketed shape representation
  // Use first input's shape (most common case)
  if (!shapes.empty() && ranks.size() > 0 && ranks[0] > 0 && shapes[0] != nullptr) {
    for (int i = 0; i < ranks[0]; i++) {
      sig.shapeBucket.push_back(bucketDimension(shapes[0][i]));
    }
  }

  return sig;
}

void KernelPerformanceRegistry::recordPerformance(const KernelSignature& sig, samediff::Engine engine,
                                                   double timeNanos) {
  std::lock_guard<std::mutex> lock(_mutex);
  _cache[sig][engine].update(timeNanos);
  _dirty = true;
}

samediff::Engine KernelPerformanceRegistry::getBestEngine(const KernelSignature& sig) const {
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _cache.find(sig);
  if (it == _cache.end()) {
    return samediff::ENGINE_CPU;
  }

  const auto& engineMap = it->second;
  if (engineMap.empty()) {
    return samediff::ENGINE_CPU;
  }

  samediff::Engine bestEngine = samediff::ENGINE_CPU;
  double bestTime = std::numeric_limits<double>::max();

  for (const auto& pair : engineMap) {
    if (pair.second.isReliable() && pair.second.meanTimeNanos < bestTime) {
      bestTime = pair.second.meanTimeNanos;
      bestEngine = pair.first;
    }
  }

  return bestEngine;
}

const KernelPerformanceEntry* KernelPerformanceRegistry::getPerformance(const KernelSignature& sig,
                                                                         samediff::Engine engine) const {
  std::lock_guard<std::mutex> lock(_mutex);

  auto sigIt = _cache.find(sig);
  if (sigIt == _cache.end()) {
    return nullptr;
  }

  auto engIt = sigIt->second.find(engine);
  if (engIt == sigIt->second.end()) {
    return nullptr;
  }

  return &engIt->second;
}

bool KernelPerformanceRegistry::hasReliableData(const KernelSignature& sig) const {
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _cache.find(sig);
  if (it == _cache.end()) {
    return false;
  }

  for (const auto& pair : it->second) {
    if (pair.second.isReliable()) {
      return true;
    }
  }

  return false;
}

std::unordered_map<samediff::Engine, KernelPerformanceEntry> KernelPerformanceRegistry::getAllPerformance(
    const KernelSignature& sig) const {
  std::lock_guard<std::mutex> lock(_mutex);

  auto it = _cache.find(sig);
  if (it == _cache.end()) {
    return {};
  }

  return it->second;
}

void KernelPerformanceRegistry::clear() {
  std::lock_guard<std::mutex> lock(_mutex);
  _cache.clear();
  _dirty = true;
}

void KernelPerformanceRegistry::clearForOp(LongType opHash) {
  std::lock_guard<std::mutex> lock(_mutex);

  for (auto it = _cache.begin(); it != _cache.end();) {
    if (it->first.opHash == opHash) {
      it = _cache.erase(it);
    } else {
      ++it;
    }
  }
  _dirty = true;
}

bool KernelPerformanceRegistry::saveToFile(const std::string& path) {
  std::lock_guard<std::mutex> lock(_mutex);

  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Simple binary format: count, then entries
  uint32_t sigCount = static_cast<uint32_t>(_cache.size());
  file.write(reinterpret_cast<const char*>(&sigCount), sizeof(sigCount));

  for (const auto& sigPair : _cache) {
    const auto& sig = sigPair.first;
    const auto& engines = sigPair.second;

    // Write signature
    file.write(reinterpret_cast<const char*>(&sig.opHash), sizeof(sig.opHash));
    uint32_t rankCount = static_cast<uint32_t>(sig.shapeBucket.size());
    file.write(reinterpret_cast<const char*>(&rankCount), sizeof(rankCount));
    for (int dim : sig.shapeBucket) {
      file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    int dtypeInt = static_cast<int>(sig.dataType);
    file.write(reinterpret_cast<const char*>(&dtypeInt), sizeof(dtypeInt));

    // Write engine entries
    uint32_t engineCount = static_cast<uint32_t>(engines.size());
    file.write(reinterpret_cast<const char*>(&engineCount), sizeof(engineCount));

    for (const auto& engPair : engines) {
      int engineInt = static_cast<int>(engPair.first);
      file.write(reinterpret_cast<const char*>(&engineInt), sizeof(engineInt));
      file.write(reinterpret_cast<const char*>(&engPair.second.meanTimeNanos), sizeof(double));
      file.write(reinterpret_cast<const char*>(&engPair.second.varianceNanos), sizeof(double));
      file.write(reinterpret_cast<const char*>(&engPair.second.sampleCount), sizeof(int));
      file.write(reinterpret_cast<const char*>(&engPair.second.lastUpdated), sizeof(int64_t));
    }
  }

  _dirty = false;
  return true;
}

bool KernelPerformanceRegistry::loadFromFile(const std::string& path) {
  std::lock_guard<std::mutex> lock(_mutex);

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  _cache.clear();

  uint32_t sigCount;
  file.read(reinterpret_cast<char*>(&sigCount), sizeof(sigCount));
  if (!file.good()) return false;

  for (uint32_t i = 0; i < sigCount; i++) {
    KernelSignature sig;

    file.read(reinterpret_cast<char*>(&sig.opHash), sizeof(sig.opHash));

    uint32_t rankCount;
    file.read(reinterpret_cast<char*>(&rankCount), sizeof(rankCount));
    sig.shapeBucket.resize(rankCount);
    for (uint32_t j = 0; j < rankCount; j++) {
      file.read(reinterpret_cast<char*>(&sig.shapeBucket[j]), sizeof(int));
    }

    int dtypeInt;
    file.read(reinterpret_cast<char*>(&dtypeInt), sizeof(dtypeInt));
    sig.dataType = static_cast<DataType>(dtypeInt);

    uint32_t engineCount;
    file.read(reinterpret_cast<char*>(&engineCount), sizeof(engineCount));

    for (uint32_t j = 0; j < engineCount; j++) {
      int engineInt;
      file.read(reinterpret_cast<char*>(&engineInt), sizeof(engineInt));

      KernelPerformanceEntry entry;
      file.read(reinterpret_cast<char*>(&entry.meanTimeNanos), sizeof(double));
      file.read(reinterpret_cast<char*>(&entry.varianceNanos), sizeof(double));
      file.read(reinterpret_cast<char*>(&entry.sampleCount), sizeof(int));
      file.read(reinterpret_cast<char*>(&entry.lastUpdated), sizeof(int64_t));

      _cache[sig][static_cast<samediff::Engine>(engineInt)] = entry;
    }

    if (!file.good()) return false;
  }

  return true;
}

size_t KernelPerformanceRegistry::getCacheSize() const {
  std::lock_guard<std::mutex> lock(_mutex);
  size_t count = 0;
  for (const auto& pair : _cache) {
    count += pair.second.size();
  }
  return count;
}

size_t KernelPerformanceRegistry::getOperationCount() const {
  std::lock_guard<std::mutex> lock(_mutex);

  std::unordered_set<LongType> ops;
  for (const auto& pair : _cache) {
    ops.insert(pair.first.opHash);
  }
  return ops.size();
}

}  // namespace ops
}  // namespace sd
