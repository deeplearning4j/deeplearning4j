/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace sd {
namespace ops {
namespace helpers {

LongType barnes_row_count(NDArray* rowP, NDArray* colP, LongType N,
                          NDArray& rowCounts) {
  THROW_EXCEPTION(
      "Vulkan barnes_row_count requires a declared device descriptor or an "
      "eager scalar-count API; neither is currently available");
  return 0;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
