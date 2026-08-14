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
 * Coordinates one process-local initialization lifecycle.
 *
 * <p>The controller deliberately retains the first failure. Callers may retry only
 * by creating a new controller; an object graph whose initialization failed must
 * never be advertised as ready or rebuilt over partially published state.</p>
 */
public final class InitializationController {

    public enum Phase {
        NEW,
        INITIALIZING,
        READY,
        FAILED
    }

    private final String componentName;
    private Phase phase = Phase.NEW;
    private Thread initializingThread;
    private Throwable failure;

    public InitializationController(String componentName) {
        if (componentName == null || componentName.trim().isEmpty()) {
            throw new IllegalArgumentException("componentName must not be blank");
        }
        this.componentName = componentName;
    }

    /**
     * Acquire initialization ownership.
     *
     * @return {@code true} when the caller owns a new initialization attempt,
     *         or {@code false} when initialization already completed.
     */
    public synchronized boolean begin() {
        Thread currentThread = Thread.currentThread();
        while (phase == Phase.INITIALIZING && initializingThread != currentThread) {
            try {
                wait();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(
                        "Interrupted while waiting for " + componentName + " initialization", e);
            }
        }

        if (phase == Phase.READY) {
            return false;
        }
        if (phase == Phase.FAILED) {
            throw propagate(failure);
        }
        if (phase == Phase.INITIALIZING) {
            throw new IllegalStateException(
                    "Reentrant " + componentName + " initialization on thread "
                            + currentThread.getName());
        }

        phase = Phase.INITIALIZING;
        initializingThread = currentThread;
        return true;
    }

    public synchronized void complete() {
        requireOwner("complete");
        phase = Phase.READY;
        initializingThread = null;
        notifyAll();
    }

    public synchronized void fail(Throwable cause) {
        if (cause == null) {
            throw new IllegalArgumentException("Initialization failure must not be null");
        }

        if (phase == Phase.FAILED) {
            return;
        }
        requireOwner("fail");
        failure = cause;
        phase = Phase.FAILED;
        initializingThread = null;
        notifyAll();
    }

    public synchronized Phase getPhase() {
        return phase;
    }

    public synchronized boolean isReady() {
        return phase == Phase.READY;
    }

    public synchronized boolean isInitializingByCurrentThread() {
        return phase == Phase.INITIALIZING
                && initializingThread == Thread.currentThread();
    }

    public synchronized Throwable getFailure() {
        return failure;
    }

    /**
     * Rethrow the retained first failure, if initialization failed.
     */
    public synchronized void throwIfFailed() {
        if (phase == Phase.FAILED) {
            throw propagate(failure);
        }
    }

    /**
     * Convert checked failures to an initialization exception while preserving
     * {@link RuntimeException} and {@link Error} instances exactly.
     */
    public static RuntimeException propagate(Throwable cause) {
        if (cause instanceof Error) {
            throw (Error) cause;
        }
        if (cause instanceof RuntimeException) {
            return (RuntimeException) cause;
        }
        return new IllegalStateException("Initialization failed", cause);
    }

    private void requireOwner(String transition) {
        if (phase != Phase.INITIALIZING
                || initializingThread != Thread.currentThread()) {
            throw new IllegalStateException(
                    "Cannot " + transition + " " + componentName
                            + " initialization from thread "
                            + Thread.currentThread().getName()
                            + " while phase is " + phase);
        }
    }
}
