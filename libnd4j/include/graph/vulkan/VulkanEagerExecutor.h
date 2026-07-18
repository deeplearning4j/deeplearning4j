/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_EAGER_EXECUTOR_H
#define LIBND4J_VULKAN_EAGER_EXECUTOR_H

#include <system/common.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanLegacyOpCatalog.h>

#include <string>

namespace sd {
namespace graph {

class Context;
class RandomGenerator;
class VulkanExecutionStream;

/**
 * Descriptor-hash entry point for one-op Vulkan execution.
 *
 * The executor deliberately shares VulkanSegmentRecorder with DSP replay so
 * eager and captured execution have one capability gate, one MLIR emitter path,
 * and one descriptor binding implementation. Unsupported descriptors fail
 * explicitly; there is no host or slot-by-slot execution path here.
 */
class SD_LIB_EXPORT VulkanEagerExecutor {
 public:
  static Status execute(LongType descriptorHash, Context& context,
                        VulkanExecutionStream& stream,
                        std::string* errorMessage = nullptr);

  static Status execute(VulkanLegacyOpFamily family, int opNum,
                        Context& context, VulkanExecutionStream& stream,
                        RandomGenerator* randomState,
                        std::string* errorMessage = nullptr);

  /**
   * Execute on the stream owned by Context::launchContext().  If the context
   * has no explicit stream, resolve the default stream for that exact device.
   */
  static Status execute(LongType descriptorHash, Context& context,
                        std::string* errorMessage = nullptr);
};

}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
#endif  // LIBND4J_VULKAN_EAGER_EXECUTOR_H
