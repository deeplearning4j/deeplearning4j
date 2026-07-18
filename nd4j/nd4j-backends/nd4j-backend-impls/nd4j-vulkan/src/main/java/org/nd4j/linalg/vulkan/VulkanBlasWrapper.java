/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.linalg.factory.BaseBlasWrapper;

/**
 * Vulkan BLAS facade.
 *
 * <p>Like CUDA's wrapper, this uses the shared facade and delegates to the
 * backend's level implementations. Vulkan's array factory explicitly rejects
 * levels that do not yet have device implementations, so no host BLAS library
 * is substituted.</p>
 */
public final class VulkanBlasWrapper extends BaseBlasWrapper {
}
