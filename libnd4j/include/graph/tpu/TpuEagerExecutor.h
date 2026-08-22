/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef LIBND4J_TPU_EAGEREXECUTOR_H
#define LIBND4J_TPU_EAGEREXECUTOR_H

#include <system/common.h>

#ifdef SD_TPU

#include <string>

namespace sd {
namespace graph {

class Context;

/**
 * One-op execution through the same trait/KernelSpec/StableHLO/PJRT path used by
 * DSP segments. Unsupported descriptors fail; eager TPU never invokes the
 * source-authored CPU numerical implementation.
 */
class SD_LIB_EXPORT TpuEagerExecutor {
 public:
  static Status execute(LongType descriptorHash, Context& context,
                        std::string* errorMessage = nullptr);
};

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
#endif  // LIBND4J_TPU_EAGEREXECUTOR_H
