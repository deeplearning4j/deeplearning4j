/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_LEGACY_INDEXING_OPS_H
#define LIBND4J_LEGACY_INDEXING_OPS_H

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

/**
 * Descriptor identities for the NativeOps indexed-TAD movement ABI.
 *
 * These are ordinary declarable ops so every backend uses the canonical
 * descriptor hash and op-local traits. Backend emitters select their lowering
 * by descriptor identity; no operation-name routing is required.
 */
DECLARE_CUSTOM_OP(legacy_pull_rows, 2, 1, false, 0, 2);
DECLARE_CUSTOM_OP(legacy_shuffle, -1, -1, true, 0, -1);

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LEGACY_INDEXING_OPS_H
