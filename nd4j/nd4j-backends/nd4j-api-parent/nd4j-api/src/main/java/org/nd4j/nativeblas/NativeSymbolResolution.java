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

import java.util.Locale;

/**
 * Selects who resolves optional native function entry points.
 *
 * <p>Ordinary JVM deployments use JavaCPP because it owns the loaded native-library
 * handles and can safely return function pointers. Embedded runtimes can instead
 * select {@code process} when their native backend resolves the same entry points
 * from the process image. Backends branch directly on the immutable policy before
 * constructing JavaCPP arguments, allowing Native Image to remove the host-JNI path
 * during reachability analysis.</p>
 */
public final class NativeSymbolResolution {

    public static final String PROPERTY = "org.nd4j.native.symbolResolution";
    public static final String JAVACPP = "javacpp";
    public static final String PROCESS = "process";

    /**
     * Immutable policy used directly at backend call sites before any JavaCPP lookup
     * object is constructed. Native Image producers initialize this class at build
     * time, which lets reachability analysis remove the host-JNI branch completely.
     */
    public static final boolean PROCESS_SYMBOLS = resolveProcessSymbols(
            System.getProperty(PROPERTY, JAVACPP));

    private NativeSymbolResolution() {
    }

    static boolean resolveProcessSymbols(String configuredValue) {
        String normalized = configuredValue == null
                ? JAVACPP
                : configuredValue.trim().toLowerCase(Locale.ROOT);
        if (JAVACPP.equals(normalized)) {
            return false;
        }
        if (PROCESS.equals(normalized)) {
            return true;
        }
        throw new IllegalArgumentException(
                "Unsupported " + PROPERTY + " value '" + configuredValue
                        + "'; expected '" + JAVACPP + "' or '" + PROCESS + "'");
    }
}
