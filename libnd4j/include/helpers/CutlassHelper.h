/* ******************************************************************************
 *
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

#ifndef LIBND4J_CUTLASS_HELPER_H
#define LIBND4J_CUTLASS_HELPER_H

#include <system/common.h>

namespace sd {

/**
 * Utility for querying CUDA SM (Streaming Multiprocessor) version
 * and CUTLASS feature availability.
 */
class SD_LIB_EXPORT CutlassHelper {
 public:
  /**
   * Returns the SM version as an integer for the given device.
   * E.g., 89 for RTX 4090 (Ada Lovelace), 90 for H100 (Hopper).
   * Returns 0 on CPU builds or if device is invalid.
   */
  static int getSmVersion(int deviceId = 0);

  /**
   * Returns true if the given device supports native FP8 (E4M3/E5M2)
   * tensor core operations. Requires SM >= 89 (Ada Lovelace).
   */
  static bool hasFp8NativeSupport(int deviceId = 0);

  /**
   * Returns true if the given device has Hopper-class features
   * (TMA, warp-group MMA, etc.). Requires SM >= 90.
   */
  static bool hasHopperFeatures(int deviceId = 0);

  /**
   * Returns true if CUTLASS is available in this build.
   */
  static bool isCutlassAvailable();
};

}  // namespace sd

#endif  // LIBND4J_CUTLASS_HELPER_H
