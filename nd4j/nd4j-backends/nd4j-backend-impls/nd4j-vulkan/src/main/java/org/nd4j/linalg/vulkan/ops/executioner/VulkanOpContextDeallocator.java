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
package org.nd4j.linalg.vulkan.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.nativeblas.NativeBufferOwner;
import org.nd4j.nativeblas.OpaqueContext;

/**
 * Releases a Vulkan native op context on the device that owns it.
 */
@Slf4j
public final class VulkanOpContextDeallocator implements Deallocator {
    private final transient OpaqueContext context;
    private final transient NativeBufferOwner owner;
    private final int deviceId;
    private volatile boolean deallocated;

    public VulkanOpContextDeallocator(VulkanOpContext opContext) {
        context = opContext.contextPointer();
        owner = context.backendOwner();
        deviceId = opContext.targetDevice();
    }

    @Override
    public void deallocate() {
        if (deallocated) {
            return;
        }

        synchronized (this) {
            if (deallocated) {
                return;
            }

            if (context == null || context.isNull()) {
                deallocated = true;
                return;
            }

            int currentDevice = owner.currentDevice();
            boolean switched = currentDevice != deviceId;
            try {
                if (switched) {
                    owner.setDevice(deviceId);
                }

                owner.commit();
                if (!context.isNull()) {
                    owner.nativeOps().ctxPurge(context);
                    context.close();
                }
            } catch (Exception e) {
                log.error("Error during Vulkan op-context deallocation", e);
            } finally {
                if (switched) {
                    try {
                        owner.setDevice(currentDevice);
                    } catch (Exception e) {
                        log.error("Could not restore device after Vulkan op-context deallocation", e);
                    }
                }
                deallocated = true;
            }
        }
    }

    @Override
    public boolean isConstant() {
        return false;
    }
}
