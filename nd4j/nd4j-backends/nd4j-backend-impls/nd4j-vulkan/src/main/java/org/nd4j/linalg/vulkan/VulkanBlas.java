/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.nativeblas.Nd4jBlas;

/**
 * Vulkan implementation of the legacy ND4J BLAS metadata contract.
 *
 * <p>Actual level operations are emitted as Vulkan custom ops by
 * {@link VulkanBlasWrapper}; this object exists because the legacy factory API
 * requires a {@code Blas} instance during backend initialization.</p>
 */
public final class VulkanBlas extends Nd4jBlas {
    @Override
    public void setMaxThreads(int num) {
        // Vulkan schedules work on device queues rather than an OpenMP BLAS pool.
    }

    @Override
    public int getMaxThreads() {
        return 0;
    }

    @Override
    public int getBlasVendorId() {
        return 0;
    }
}
