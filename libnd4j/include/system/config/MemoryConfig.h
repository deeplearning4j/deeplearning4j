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

#ifndef LIBND4J_MEMORY_CONFIG_H
#define LIBND4J_MEMORY_CONFIG_H

#include <system/common.h>

#include <atomic>

namespace sd {
namespace config {

/**
 * Memory pool and allocator configuration: release thresholds, failover
 * headroom gates, and other memory-management tuning knobs.
 *
 * Accessible from Java via: Nd4j.getEnvironment().memory().setXxx(v)
 * Accessible from C++ via:  sd::Environment::getInstance().memory().xxx()
 */
class SD_LIB_EXPORT MemoryConfig {
 private:
  // Pool release threshold: percentage of total device memory the CUDA memory
  // pool is allowed to keep reserved. Values 1-100. At pool init, this is
  // converted to an absolute byte threshold via cudaMemPoolAttrReleaseThreshold.
  std::atomic<int> _poolReleaseThresholdPercent{75};

  // Non-peer failover headroom: minimum percentage of total GPU memory that
  // must remain free AFTER a managed-memory allocation for it to use the fast
  // cudaMallocManaged path on non-peer devices. Below this threshold, the
  // allocator falls through to pinned host memory (slower but safe from
  // page-eviction error 700). Values 1-100.
  std::atomic<int> _nonPeerHeadroomPercent{50};

 public:
  MemoryConfig();

  // --- Pool release threshold ---
  int poolReleaseThresholdPercent();
  void setPoolReleaseThresholdPercent(int percent);

  // --- Non-peer failover headroom ---
  int nonPeerHeadroomPercent();
  void setNonPeerHeadroomPercent(int percent);

  /**
   * Initialize from environment variables.
   */
  void initFromEnvironment();
};

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_MEMORY_CONFIG_H
