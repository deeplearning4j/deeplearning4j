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
// Stronger 64-bit hash function helper, as described here:
// http://www.javamex.com/tutorials/collections/strong_hash_code_implementation.shtml
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_HASH_H
#define LIBND4J_HELPER_HASH_H

#include <system/common.h>

#include <mutex>
#include <string>

namespace sd {
namespace ops {
class SD_LIB_EXPORT HashHelper {
 private:
  sd::LongType _byteTable[256];
  const sd::LongType HSTART = 0xBB40E64DA205B064L;
  const sd::LongType HMULT = 7664345821815920749L;

  bool _isInit = false;
  std::mutex _locker;

 public:
  static HashHelper& getInstance();
  sd::LongType getLongHash(std::string& str);
};
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPER_HASH_H
