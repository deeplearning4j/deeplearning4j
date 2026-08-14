/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.factory;

/**
 * Process-wide ND4J bootstrap state that remains inspectable even if
 * {@link Nd4j}'s class initializer fails.
 */
public final class Nd4jInitialization {

    private static final InitializationController CONTROLLER =
            new InitializationController("ND4J");

    private Nd4jInitialization() {
    }

    static boolean begin() {
        return CONTROLLER.begin();
    }

    static void complete() {
        CONTROLLER.complete();
    }

    static void fail(Throwable failure) {
        CONTROLLER.fail(failure);
    }

    public static InitializationController.Phase getPhase() {
        return CONTROLLER.getPhase();
    }

    public static boolean isReady() {
        return CONTROLLER.isReady();
    }

    public static boolean isInitializingByCurrentThread() {
        return CONTROLLER.isInitializingByCurrentThread();
    }

    public static Throwable getFailure() {
        return CONTROLLER.getFailure();
    }

    public static void throwIfFailed() {
        CONTROLLER.throwIfFailed();
    }

    static RuntimeException propagate(Throwable failure) {
        return InitializationController.propagate(failure);
    }
}
