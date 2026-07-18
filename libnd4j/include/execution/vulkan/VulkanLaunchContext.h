/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_LAUNCH_CONTEXT_H
#define LIBND4J_VULKAN_LAUNCH_CONTEXT_H

#if !defined(SD_VULKAN)
#error "VulkanLaunchContext is only available in SD_VULKAN builds"
#endif

#include <system/common.h>
#include <system/op_boilerplate.h>

namespace sd {

class LaunchContext;

namespace graph {

/** Bind an externally owned Vulkan stream to one LaunchContext instance. */
SD_LIB_EXPORT void bindVulkanExecutionStream(LaunchContext* context,
                                             Pointer stream);

/** Return the externally owned stream bound to this context, if any. */
SD_LIB_EXPORT Pointer vulkanExecutionStream(const LaunchContext* context);

/** Remove a context binding without taking ownership of the stream. */
SD_LIB_EXPORT void releaseVulkanExecutionStream(const LaunchContext* context);

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_VULKAN_LAUNCH_CONTEXT_H
