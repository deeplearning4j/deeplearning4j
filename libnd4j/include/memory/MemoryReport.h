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

#ifndef LIBND4J_MEMORYREPORT_H
#define LIBND4J_MEMORYREPORT_H

#include <system/common.h>

namespace sd {
namespace memory {
class SD_LIB_EXPORT MemoryReport {
 private:
  LongType _vm = 0;
  LongType _rss = 0;

 public:
  MemoryReport() = default;
  ~MemoryReport() = default;

  bool operator<(const MemoryReport& other) const;
  bool operator<=(const MemoryReport& other) const;
  bool operator>(const MemoryReport& other) const;
  bool operator>=(const MemoryReport& other) const;
  bool operator==(const MemoryReport& other) const;
  bool operator!=(const MemoryReport& other) const;

  LongType getVM() const;
  void setVM(LongType vm);

  LongType getRSS() const;
  void setRSS(LongType rss);
};
}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_MEMORYREPORT_H
