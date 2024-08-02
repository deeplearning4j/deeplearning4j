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
// Created by raver119 on 12.09.17.
//

#ifndef LIBND4J_MEMORYREGISTRATOR_H
#define LIBND4J_MEMORYREGISTRATOR_H

#include <system/common.h>
#include <system/op_boilerplate.h>

#include <map>
#include <mutex>
#include <unordered_map>

#include "Workspace.h"

namespace sd {
namespace memory {
class SD_LIB_EXPORT MemoryRegistrator {
 protected:
  Workspace* _workspace;
  SD_MAP_IMPL<LongType, LongType> _footprint;
  std::mutex _lock;

  MemoryRegistrator();
  ~MemoryRegistrator() = default;

 public:
  static MemoryRegistrator& getInstance();
  bool hasWorkspaceAttached();
  Workspace* getWorkspace();
  void attachWorkspace(Workspace* workspace);
  void forgetWorkspace();

  /**
   * This method allows you to set memory requirements for given graph
   */
  void setGraphMemoryFootprint(LongType hash, LongType bytes);

  /**
   * This method allows you to set memory requirements for given graph, ONLY if
   * new amount of bytes is greater then current one
   */
  void setGraphMemoryFootprintIfGreater(LongType hash, LongType bytes);

  /**
   * This method returns memory requirements for given graph
   */
  LongType getGraphMemoryFootprint(LongType hash);
};
}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_MEMORYREGISTRATOR_H
