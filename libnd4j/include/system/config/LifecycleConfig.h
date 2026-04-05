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

#ifndef LIBND4J_LIFECYCLE_CONFIG_H
#define LIBND4J_LIFECYCLE_CONFIG_H

#include <system/common.h>

#include <atomic>
#include <cstddef>

namespace sd {
namespace config {

/**
 * NDArray/DataBuffer lifecycle tracking configuration: master switch,
 * view tracking, deletion tracking, stack depth, report intervals,
 * snapshot files, and per-tracker enable flags.
 */
class SD_LIB_EXPORT LifecycleConfig {
 private:
  std::atomic<bool> _lifecycleTracking{false};
  std::atomic<bool> _trackViews{true};
  std::atomic<bool> _trackDeletions{true};
  std::atomic<int> _stackDepth{32};
  std::atomic<int> _reportInterval{300};
  std::atomic<size_t> _maxDeletionHistory{10000};
  std::atomic<bool> _snapshotFiles{false};
  std::atomic<bool> _trackOperations{false};

  // Per-tracker enable flags
  std::atomic<bool> _ndArrayTracking{false};
  std::atomic<bool> _dataBufferTracking{false};
  std::atomic<bool> _tadCacheTracking{false};
  std::atomic<bool> _shapeCacheTracking{false};
  std::atomic<bool> _opContextTracking{false};

 public:
  LifecycleConfig();

  // --- Master switch ---
  bool isLifecycleTracking() { return _lifecycleTracking.load(); }
  void setLifecycleTracking(bool v) { _lifecycleTracking.store(v); }

  // --- View/deletion tracking ---
  bool isTrackViews() { return _trackViews.load(); }
  void setTrackViews(bool v) { _trackViews.store(v); }
  bool isTrackDeletions() { return _trackDeletions.load(); }
  void setTrackDeletions(bool v) { _trackDeletions.store(v); }

  // --- Stack depth & reporting ---
  int getStackDepth() { return _stackDepth.load(); }
  void setStackDepth(int v);
  int getReportInterval() { return _reportInterval.load(); }
  void setReportInterval(int v);
  size_t getMaxDeletionHistory() { return _maxDeletionHistory.load(); }
  void setMaxDeletionHistory(size_t v) { _maxDeletionHistory.store(v); }

  // --- Snapshots & operations ---
  bool isSnapshotFiles() { return _snapshotFiles.load(); }
  void setSnapshotFiles(bool v) { _snapshotFiles.store(v); }
  bool isTrackOperations() { return _trackOperations.load(); }
  void setTrackOperations(bool v) { _trackOperations.store(v); }

  // --- Per-tracker enable flags ---
  // Note: the actual tracker enable/disable is done via Environment setters
  // that also call the tracker singleton. These just store the flag.
  bool isNDArrayTracking() { return _ndArrayTracking.load(); }
  void setNDArrayTracking(bool v) { _ndArrayTracking.store(v); }
  bool isDataBufferTracking() { return _dataBufferTracking.load(); }
  void setDataBufferTracking(bool v) { _dataBufferTracking.store(v); }
  bool isTADCacheTracking() { return _tadCacheTracking.load(); }
  void setTADCacheTracking(bool v) { _tadCacheTracking.store(v); }
  bool isShapeCacheTracking() { return _shapeCacheTracking.load(); }
  void setShapeCacheTracking(bool v) { _shapeCacheTracking.store(v); }
  bool isOpContextTracking() { return _opContextTracking.load(); }
  void setOpContextTracking(bool v) { _opContextTracking.store(v); }

  /**
   * Initialize from environment variables.
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_LIFECYCLE_CONFIG_H
