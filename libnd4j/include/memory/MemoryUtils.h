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

//
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_MEMORYUTILS_H
#define LIBND4J_MEMORYUTILS_H

#include "MemoryReport.h"

namespace sd {
namespace memory {
class SD_LIB_EXPORT MemoryUtils {
 public:
  static bool retrieveMemoryStatistics(MemoryReport& report);

  /**
   * Returns available (free) system RAM in bytes.
   * Linux:   reads MemAvailable from /proc/meminfo
   * macOS:   vm_statistics64 free+inactive pages
   * Windows: GlobalMemoryStatusEx ullAvailPhys
   * Returns 0 if the query fails or the platform is unsupported.
   */
  static size_t getSystemFreeMemoryBytes();
};
}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_MEMORYUTILS_H
