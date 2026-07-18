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

package org.nd4j.nativeblas;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;

/**
 * Exact backend authority for an {@link OpaqueDataBuffer}.
 *
 * <p>The owner is attached when a native buffer is created and remains attached
 * for its complete lifetime. Buffer synchronization, migration, and
 * deallocation therefore never have to infer an owner from ND4J's process-wide
 * primary backend.</p>
 */
public interface NativeBufferOwner {

    NativeOps nativeOps();

    DeallocatorService deallocatorService();

    int currentDevice();

    int deviceCount();

    void setDevice(int deviceId);

    void commit();

    DeviceDescriptor deviceDescriptor(int deviceId);

    /** Records bytes owned by this backend on the exact allocation device. */
    void recordAllocation(DeviceDescriptor device, long bytes);

    /** Releases bytes previously recorded by this backend on the exact allocation device. */
    void recordDeallocation(DeviceDescriptor device, long bytes);
}
