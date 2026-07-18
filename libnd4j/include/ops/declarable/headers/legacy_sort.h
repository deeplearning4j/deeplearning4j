/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_LEGACY_SORT_OPS_H
#define LIBND4J_LEGACY_SORT_OPS_H

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

DECLARE_CUSTOM_OP(legacy_sort, 1, 1, true, 0, 0);
DECLARE_CUSTOM_OP(legacy_sort_tad, 1, 1, true, 0, -1);
DECLARE_CUSTOM_OP(legacy_sort_by_key, 2, 2, true, 0, 0);
DECLARE_CUSTOM_OP(legacy_sort_by_value, 2, 2, true, 0, 0);
DECLARE_CUSTOM_OP(legacy_sort_tad_by_key, 2, 2, true, 0, -1);
DECLARE_CUSTOM_OP(legacy_sort_tad_by_value, 2, 2, true, 0, -1);

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LEGACY_SORT_OPS_H
