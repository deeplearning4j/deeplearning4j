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
// @author raver119@gmail.com
//

#ifndef SD_PLATFORM_BOILERPLATE_H
#define SD_PLATFORM_BOILERPLATE_H
#include <ConstMessages.h>
#include <execution/Engine.h>

#define CONCATP(A, B) A##_##B

#define DECLARE_PLATFORM_F(NAME, ENGINE, CNAME)                             \
  class SD_LIB_EXPORT PLATFORM_##CNAME : public PlatformHelper {            \
   public:                                                                  \
    PLATFORM_##CNAME() : PlatformHelper(#NAME, samediff::Engine::ENGINE) {} \
    bool isUsable(graph::Context &context) override;                        \
    sd::Status invokeHelper(graph::Context &context) override;              \
  };

#define DECLARE_PLATFORM(NAME, ENGINE) DECLARE_PLATFORM_F(NAME, ENGINE, NAME##_##ENGINE)

#define PLATFORM_IMPL_F(NAME, ENGINE, CNAME)                         \
  struct SD_LIB_EXPORT __registratorPlatformHelper_##CNAME {         \
    __registratorPlatformHelper_##CNAME() {                          \
      auto helper = new PLATFORM_##CNAME();                          \
      OpRegistrator::getInstance().registerHelper(helper);           \
    }                                                                \
  };                                                                 \
  static __registratorPlatformHelper_##CNAME platformHelper_##CNAME; \
  sd::Status PLATFORM_##CNAME::invokeHelper(sd::graph::Context &block)

#define PLATFORM_IMPL(NAME, ENGINE) PLATFORM_IMPL_F(NAME, ENGINE, NAME##_##ENGINE)

#define PLATFORM_CHECK_F(NAME, ENGINE, CNAME) bool PLATFORM_##CNAME::isUsable(graph::Context &block)
#define PLATFORM_CHECK(NAME, ENGINE) PLATFORM_CHECK_F(NAME, ENGINE, NAME##_##ENGINE)

#define DECLARE_PLATFORM_LEGACY_F(PREFIX, OPNUM, ENGINE, CNAME)                                                       \
  class SD_LIB_EXPORT PLATFORM_LEGACY_##CNAME : public PlatformHelperLegacy {                                         \
   public:                                                                                                            \
    PLATFORM_LEGACY_##CNAME() : PlatformHelperLegacy(PREFIX, OPNUM, samediff::Engine::ENGINE) {}                      \
    bool isUsable(void *extraParams, const sd::LongType *outShapeInfo, const sd::LongType *inArg0ShapeInfo,           \
                  const sd::LongType *inArg1ShapeInfo) override;                                                      \
    sd::Status invokeHelper(void *extraParams, const sd::LongType *outShapeInfo, sd::InteropDataBuffer *outputBuffer, \
                            const sd::LongType *inArg0ShapeInfo, const sd::InteropDataBuffer *inArg0Buffer,           \
                            const sd::LongType *inArg1ShapeInfo, const sd::InteropDataBuffer *inArg1Buffer) override; \
  };

#define PLATFORM_LEGACY_IMPL_F(CNAME)                                                           \
  struct SD_LIB_EXPORT __registratorPlatformHelper_##CNAME {                                    \
    __registratorPlatformHelper_##CNAME() {                                                     \
      auto helper = new PLATFORM_LEGACY_##CNAME();                                              \
      OpRegistrator::getInstance().registerHelperLegacy(helper);                                \
    }                                                                                           \
  };                                                                                            \
  static __registratorPlatformHelper_##CNAME platformHelper_##CNAME;                            \
  sd::Status PLATFORM_LEGACY_##CNAME::invokeHelper(                                             \
      void *extraParams, const sd::LongType *outShapeInfo, sd::InteropDataBuffer *outputBuffer, \
      const sd::LongType *inArg0ShapeInfo, const sd::InteropDataBuffer *inArg0Buffer,           \
      const sd::LongType *inArg1ShapeInfo, const sd::InteropDataBuffer *inArg1Buffer)

#define PLATFORM_LEGACY_CHECK_F(CNAME)                                                        \
  bool PLATFORM_LEGACY_##CNAME::isUsable(void *extraParams, const sd::LongType *outShapeInfo, \
                                         const sd::LongType *inArg0ShapeInfo, const sd::LongType *inArg1ShapeInfo)

#define DECLARE_PLATFORM_TRANSFORM_STRICT(OP_ENUM_ENTRY, ENGINE)                                         \
  DECLARE_PLATFORM_LEGACY_F(UNIQUE_TRANSFORM_STRICT_PREFIX, transform::StrictOps::OP_ENUM_ENTRY, ENGINE, \
                            TRANSFORM_STRICT##_##OP_ENUM_ENTRY##_##ENGINE)
#define PLATFORM_TRANSFORM_STRICT_IMPL(OP_ENUM_ENTRY, ENGINE) \
  PLATFORM_LEGACY_IMPL_F(TRANSFORM_STRICT##_##OP_ENUM_ENTRY##_##ENGINE)
#define PLATFORM_TRANSFORM_STRICT_CHECK(OP_ENUM_ENTRY, ENGINE) \
  PLATFORM_LEGACY_CHECK_F(TRANSFORM_STRICT##_##OP_ENUM_ENTRY##_##ENGINE)

  #define DECLARE_PLATFORM_SCALAR_OP(OP_ENUM_ENTRY, ENGINE)                                         \
  DECLARE_PLATFORM_LEGACY_F(UNIQUE_SCALAROP_PREFIX, scalar::Ops::OP_ENUM_ENTRY, ENGINE, \
                            SCALAR_OP##_##OP_ENUM_ENTRY##_##ENGINE)
#define PLATFORM_SCALAR_OP_IMPL(OP_ENUM_ENTRY, ENGINE) \
  PLATFORM_LEGACY_IMPL_F(SCALAR_OP##_##OP_ENUM_ENTRY##_##ENGINE)
#define PLATFORM_SCALAR_OP_CHECK(OP_ENUM_ENTRY, ENGINE) \
  PLATFORM_LEGACY_CHECK_F(SCALAR_OP##_##OP_ENUM_ENTRY##_##ENGINE)

#endif  // SD_PLATFORM_BOILERPLATE_H
