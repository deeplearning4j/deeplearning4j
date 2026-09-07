/*
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.autodiff.samediff.execution;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Typed, test-only access to executor lifecycle boundaries. Collection getters
 * intentionally expose live state for assertions inside failure-injection callbacks;
 * use them only while execution is quiescent or on the thread holding nativeExecLock.
 * Lifecycle mutations use the same executor lock as normal execution.
 */
public final class DynamicShapePlanExecutorTestAccess {
    private DynamicShapePlanExecutorTestAccess() { }

    public static Map<Long, INDArray[]> retainedExternalInputsByPlanHandle(DynamicShapePlanExecutor executor) {
        return executor.retainedExternalInputsByPlanHandle;
    }

    public static Map<Long, Map<String, INDArray>> mutableReplicaCaches(DynamicShapePlanExecutor executor) {
        return executor.nativeMutableReplicaCaches;
    }

    public static Map<?, Long> pinnedPlanHandlesByIdentity(DynamicShapePlanExecutor executor) {
        return executor.pinnedPlanHandlesByIdentity;
    }

    public static Map<?, Long> pinnedLeaseEstimatedBytes(DynamicShapePlanExecutor executor) {
        return executor.pinnedLeaseEstimatedBytes;
    }

    public static List<INDArray> retiredMigrationArrays(DynamicShapePlanExecutor executor) {
        return executor.retiredMigrationArrays;
    }

    public static boolean migrationInputsBound(DynamicShapePlanExecutor executor) {
        return executor.migrationInputsBound;
    }

    public static List<String> cachedRequestedOutputNames(DynamicShapePlanExecutor executor) {
        return executor.cachedRequestedOutputNames;
    }

    public static void setCachedRequestedOutputNames(DynamicShapePlanExecutor executor, List<String> names) {
        withExecutionLock(executor, () -> executor.cachedRequestedOutputNames = names);
    }

    public static Thread outputReadbackThread(DynamicShapePlanExecutor executor) {
        return executor.outputReadbackThread;
    }

    public static Set<Integer> outputReadbackDevices(DynamicShapePlanExecutor executor) {
        return executor.outputReadbackDevices;
    }

    public static Set<Integer> migrationCopyDevices(DynamicShapePlanExecutor executor) {
        return executor.migrationCopyDevices;
    }

    public static void redispatchForCurrentShapes(DynamicShapePlanExecutor executor,
                                                  Map<String, INDArray> inputs, boolean shapeChangeExpected) {
        withExecutionLock(executor, () -> executor.redispatchForCurrentShapes(inputs, shapeChangeExpected));
    }

    public static void evictPinnedLeasesForCapacity(DynamicShapePlanExecutor executor,
                                                   long shapeHash, int graphMode) {
        withExecutionLock(executor, () -> executor.evictPinnedLeasesForCapacity(
                new DynamicShapePlanExecutor.PlanLeaseKey(shapeHash, graphMode)));
    }

    public static void closeZeroCopyOutputCache(DynamicShapePlanExecutor executor) {
        withExecutionLock(executor, executor::closeZeroCopyOutputCache);
    }

    public static Map<String, INDArray> executeNative(DynamicShapePlanExecutor executor,
                                                      DynamicShapePlan plan, Map<String, INDArray> inputs) {
        // executeNative acquires nativeExecLock itself, including exceptional cleanup.
        return executor.executeNative(plan, inputs);
    }

    private static void withExecutionLock(DynamicShapePlanExecutor executor, Runnable action) {
        executor.nativeExecLock.lock();
        try {
            action.run();
        } finally {
            executor.nativeExecLock.unlock();
        }
    }
}
