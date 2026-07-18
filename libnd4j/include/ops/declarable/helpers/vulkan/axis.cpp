/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <ops/declarable/helpers/axis.h>

namespace sd {
namespace ops {
namespace helpers {

void adjustAxis(LongType rank, NDArray* axisVector,
                std::vector<LongType>& output) {
  if (axisVector == nullptr) return;

  if (axisVector->isScalar()) {
    output.resize(1);
    auto axis = axisVector->e<LongType>(0);
    if (axis < 0) axis += rank;
    output[0] = axis;
    return;
  }

  output.resize(axisVector->lengthOf());
  axisVector->tickReadDevice();
  axisVector->syncToHost();
  for (LongType index = 0; index < axisVector->lengthOf(); ++index) {
    auto axis = axisVector->e<LongType>(index);
    if (axis < 0) axis += rank;
    output[index] = axis;
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
