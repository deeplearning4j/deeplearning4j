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
package org.nd4j.linalg.cpu.nativecpu;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.NativeSymbolResolution;

import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OpenblasLapackDelegatorTest {
    private static final AtomicInteger OPENBLAS_THREADS = new AtomicInteger(1);
    private static NativeOps previousNativeOps;
    private static String previousInitProperty;
    private static String previousResolutionProperty;

    @BeforeAll
    static void configureProcessSymbolBackend() {
        previousInitProperty = System.getProperty(ND4JSystemProperties.INIT_NATIVEOPS_HOLDER);
        previousResolutionProperty = System.getProperty(NativeSymbolResolution.PROPERTY);
        System.setProperty(ND4JSystemProperties.INIT_NATIVEOPS_HOLDER, "false");
        System.setProperty(NativeSymbolResolution.PROPERTY, NativeSymbolResolution.PROCESS);

        NativeOpsHolder holder = NativeOpsHolder.getInstance();
        previousNativeOps = holder.getDeviceNativeOps();
        NativeOps nativeOps = (NativeOps) Proxy.newProxyInstance(
                NativeOps.class.getClassLoader(),
                new Class<?>[]{NativeOps.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("setOpenBlasThreads")) {
                        OPENBLAS_THREADS.set((Integer) args[0]);
                        return null;
                    }
                    if (method.getName().equals("getOpenBlasThreads")) {
                        return OPENBLAS_THREADS.get();
                    }
                    return defaultValue(method.getReturnType());
                });
        holder.setDeviceNativeOps(nativeOps);
    }

    @AfterAll
    static void restoreNativeOpsHolder() {
        NativeOpsHolder.getInstance().setDeviceNativeOps(previousNativeOps);
        restoreProperty(ND4JSystemProperties.INIT_NATIVEOPS_HOLDER, previousInitProperty);
        restoreProperty(NativeSymbolResolution.PROPERTY, previousResolutionProperty);
    }

    @Test
    void processSymbolThreadControlNeverCallsJavaCppVendorProbes() {
        OpenblasLapackDelegator delegator = new OpenblasLapackDelegator();

        delegator.blas_set_num_threads(7);

        assertEquals(7, OPENBLAS_THREADS.get());
        assertEquals(7, delegator.blas_get_num_threads());
        assertEquals(2, delegator.blas_get_vendor());
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }

    private static Object defaultValue(Class<?> type) {
        if (!type.isPrimitive() || type == void.class) {
            return null;
        }
        if (type == boolean.class) {
            return false;
        }
        if (type == byte.class) {
            return (byte) 0;
        }
        if (type == short.class) {
            return (short) 0;
        }
        if (type == int.class) {
            return 0;
        }
        if (type == long.class) {
            return 0L;
        }
        if (type == float.class) {
            return 0.0f;
        }
        if (type == double.class) {
            return 0.0d;
        }
        if (type == char.class) {
            return '\0';
        }
        throw new AssertionError("Unhandled primitive return type: " + type);
    }
}
