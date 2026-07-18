/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_TAD_CALCULATOR_PLATFORM_H
#define LIBND4J_TAD_CALCULATOR_PLATFORM_H

#include <array/ConstantOffsetsBuffer.h>
#include <array/ConstantShapeBuffer.h>
#include <array/TadPack.h>
#include <system/BackendNamespace.h>
#include <system/common.h>

#include <memory>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace tad_platform {

SD_LIB_EXPORT TadShapeOwnership shapeOwnership();
SD_LIB_EXPORT ConstantShapeBuffer* createShape(const LongType* shapeInfo,
                                               LongType rank);
SD_LIB_EXPORT ConstantOffsetsBuffer* createOffsets(
    std::unique_ptr<LongType[]> offsets, LongType count);

}  // namespace tad_platform
SD_BACKEND_ABI_NAMESPACE_END

#endif  // LIBND4J_TAD_CALCULATOR_PLATFORM_H
