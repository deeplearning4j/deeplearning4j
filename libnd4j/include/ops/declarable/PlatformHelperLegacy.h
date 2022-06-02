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

#ifndef SD_PLATFORMHELPERLEGACY_H
#define SD_PLATFORMHELPERLEGACY_H
#include <array/InteropDataBuffer.h>
#include <execution/Engine.h>
#include <graph/Context.h>
#include <helpers/ShapeUtils.h>
#include <system/RequirementsHelper.h>
#include <system/op_enums.h>

#include <string>

namespace sd {
namespace ops {
namespace platforms {

struct PlatformHelperLegacyEntry {
  // prefix for the legacy. must be constant with global scope
  const char *prefix;

  int opNum;
  // target engine for this impl
  samediff::Engine engine;

  bool operator==(const PlatformHelperLegacyEntry &other) const {
    return (prefix == other.prefix && opNum == other.opNum && engine == other.engine);
  }
};

struct PlatformHelperLegacyEntryHasher {
  std::size_t operator()(PlatformHelperLegacyEntry const &p) const noexcept {
    auto res = std::hash<sd::LongType>()(reinterpret_cast<sd::LongType>(p.prefix));
    res ^= std::hash<int>()(p.opNum) + 0x9e3779b9 + (res << 6) + (res >> 2);
    res ^= std::hash<int>()(p.engine) + 0x9e3779b9 + (res << 6) + (res >> 2);
    return res;
  }
};
/**
 * This abstract class defines methods used by platform-specific helpers implementations for legacy ops
 */
class SD_LIB_EXPORT PlatformHelperLegacy {
 protected:
  PlatformHelperLegacyEntry entry;

 public:
  PlatformHelperLegacy(const char *prefix, int opNum, samediff::Engine engine) : entry{prefix, opNum, engine} {}

  ~PlatformHelperLegacy() = default;

  PlatformHelperLegacyEntry getEntry() const { return entry; }

  /**
   * This method checks, if given helper can be used with given input/output shapes
   *
   * @param context
   * @return
   */
  virtual bool isUsable(void *extraParams, const sd::LongType *outShapeInfo, const sd::LongType *inArg0ShapeInfo,
                        const sd::LongType *inArg1ShapeInfo) = 0;

  /**
   * This method invokes helper
   *
   * @param context
   * @return
   */
  virtual sd::Status invokeHelper(void *extraParams, const sd::LongType *outShapeInfo,
                                  sd::InteropDataBuffer *outputBuffer, const sd::LongType *inArg0ShapeInfo,
                                  const sd::InteropDataBuffer *inArg0Buffer, const sd::LongType *inArg1ShapeInfo,
                                  const sd::InteropDataBuffer *inArg1Buffer) = 0;
};
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_PLATFORMHELPERLEGACY_H
