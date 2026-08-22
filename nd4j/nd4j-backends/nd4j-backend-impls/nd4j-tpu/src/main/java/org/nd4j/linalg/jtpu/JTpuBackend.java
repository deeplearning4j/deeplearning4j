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

package org.nd4j.linalg.jtpu;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.memory.MemoryManager;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * ND4J backend for Google TPU through the runtime-loaded PJRT C API.
 *
 * <p>Java arrays remain host-native NDArrays. DSP lowers admitted SameDiff
 * segments to StableHLO and the native TPU replay handle owns all transient PJRT
 * buffers and executables. This keeps one authoritative ND4J buffer lifecycle
 * while still providing compiled TPU execution.</p>
 */
public final class JTpuBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JTpuBackend.class);
    private static final String LINALG_PROPS = "/nd4j-jtpu.properties";
    private static final String BINDING_CLASS = "org.nd4j.linalg.jtpu.bindings.Nd4jTpu";

    static NativeOps createNativeOps() throws ReflectiveOperationException {
        Class<?> binding = Class.forName(BINDING_CLASS, true, JTpuBackend.class.getClassLoader());
        Object instance = binding.getDeclaredConstructor().newInstance();
        if (!(instance instanceof NativeOps)) {
            throw new IllegalStateException(BINDING_CLASS + " does not implement NativeOps");
        }
        return (NativeOps) instance;
    }

    @Override
    public boolean isAvailable() {
        return canRun();
    }

    @Override
    public boolean canRun() {
        try {
            NativeOps nativeOps = createNativeOps();
            nativeOps.initializeDevicesAndFunctions();
            return nativeOps.getAvailableDevices() > 0;
        } catch (Throwable failure) {
            log.debug("TPU/PJRT backend is unavailable", failure);
            return false;
        }
    }

    @Override
    public int getPriority() {
        // Keep TPU strictly above CPU and, when priorities leave room, below GPU.
        // User-configured CPU/GPU priorities remain authoritative.
        if (BACKEND_PRIORITY_GPU > BACKEND_PRIORITY_CPU + 1) {
            return BACKEND_PRIORITY_CPU
                    + Math.max(1, (BACKEND_PRIORITY_GPU - BACKEND_PRIORITY_CPU) / 2);
        }
        return BACKEND_PRIORITY_CPU + 1;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JTpuBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        return NDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        return TpuEnvironment.getInstance();
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public String buildInfo() {
        try {
            return createNativeOps().buildInfo();
        } catch (Throwable failure) {
            return "ND4J TPU Backend (PJRT unavailable: " + failure.getMessage() + ")";
        }
    }

    @Override
    public void logBackendInit() {
        if (!Boolean.parseBoolean(System.getProperty(
                ND4JSystemProperties.LOG_INITIALIZATION, "true"))) {
            return;
        }
        log.info("ND4J TPU backend initialized with {} addressable device(s)",
                discoverDevices().size());
        log.info("Backend build information:\n{}", buildInfo());
    }

    @Override
    public List<DeviceDescriptor> discoverDevices() {
        try {
            NativeOps nativeOps = createNativeOps();
            nativeOps.initializeDevicesAndFunctions();
            int count = nativeOps.getAvailableDevices();
            if (count <= 0) {
                return Collections.emptyList();
            }
            List<DeviceDescriptor> devices = new ArrayList<>(count);
            for (int i = 0; i < count; ++i) {
                devices.add(DeviceDescriptor.accelerator(getBackendId(), DeviceType.TPU, i));
            }
            return devices;
        } catch (Throwable failure) {
            log.debug("Unable to enumerate TPU devices", failure);
            return Collections.emptyList();
        }
    }

    @Override
    public OpExecutioner createExecutioner() {
        return Nd4j.getExecutioner();
    }

    @Override
    public MemoryManager createMemoryManager() {
        return Nd4j.getMemoryManager();
    }

    @Override
    public String getBackendId() {
        return "tpu";
    }
}
