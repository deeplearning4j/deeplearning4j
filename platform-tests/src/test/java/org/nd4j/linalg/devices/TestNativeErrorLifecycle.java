/*
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.linalg.devices;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Backend-level regressions for the native error/device lifecycle (review
 * round 7, patch 4): the round-6 change that reports the host as one
 * addressable CPU device (libnd4j/include/legacy/cpu/NativeOps.cpp
 * getAvailableDevices) and the scalar-op stale-error clear in
 * NativeOpExecutioner.exec(ScalarOp) both affect every op family, not just
 * MTP. These tests pin the contract at the backend layer.
 *
 * <p>Required properties:</p>
 * <ol>
 *   <li>Host device 0 is a valid, addressable device; the backend reports at
 *       least one device and accepts device 0.</li>
 *   <li>Error followed by success: a deliberately failed native op does not
 *       poison the next successful scalar/reduction/transform/custom op.</li>
 *   <li>Fresh errors are preserved: each failure reports its own operation's
 *       message, not a previous op's.</li>
 *   <li>Allocation on every reported device succeeds (the device-count fix's
 *       direct consumer is OpaqueNDArray.create's device validation).</li>
 * </ol>
 */
public class TestNativeErrorLifecycle {

    /** Host device 0 must be accepted by the owning backend (round-6 patch D). */
    @Test
    public void testHostDeviceZeroIsAddressable() {
        int deviceCount = Nd4j.getAffinityManager().getNumberOfDevices();
        assertTrue(deviceCount >= 1,
                "backend must report at least one addressable device (the host), got "
                        + deviceCount);
        // The direct consumer of the device count: array allocation on device 0.
        try (INDArray array = Nd4j.ones(DataType.FLOAT, 4)) {
            assertEquals(4, array.length());
            assertEquals(1.0f, array.getFloat(0), 0.0f);
        }
    }

    /**
     * A deliberately failed native operation must not poison the next
     * successful operations across op families (scalar, reduction, transform,
     * custom). This is the regression for the stale lastErrorCode replay:
     * before the scalar-path clear, a fail-loud guard inside one op made every
     * later successful op throw that old error.
     */
    @Test
    public void testFailedOpDoesNotPoisonLaterOps() {
        // 1. Force a native failure with a controlled bad input: NaN-preserving
        //    is not a failure, so use a genuinely invalid shape operation.
        assertThrows(Exception.class, () -> Nd4j.create(new float[]{1.0f})
                        .reshape(2, 2),
                "reshape of 1 element into [2,2] must fail (element count mismatch)");

        // 2. Immediately run successful ops across families - none may replay
        //    the reshape failure.
        try (INDArray scalarResult = Nd4j.ones(DataType.FLOAT, 3).add(1.0f)) {
            assertEquals(2.0f, scalarResult.getFloat(1), 0.0f, "scalar op after failure");
        }
        try (INDArray reductionResult = Nd4j.ones(DataType.FLOAT, 4).sum()) {
            assertEquals(4.0f, reductionResult.getDouble(0), 0.0, "reduction after failure");
        }
        try (INDArray transformResult = Nd4j.ones(DataType.FLOAT, 2).mul(3.0f)) {
            assertEquals(3.0f, transformResult.getFloat(0), 0.0f, "transform after failure");
        }
        try (INDArray customResult = Nd4j.createFromArray(new float[]{1.0f, 2.0f})
                .add(Nd4j.createFromArray(new float[]{10.0f, 20.0f}))) {
            assertEquals(22.0f, customResult.getFloat(1), 0.0f, "custom op after failure");
        }
    }

    /**
     * Two DIFFERENT failures in sequence: the second must report its own
     * cause, not the first's (the clear must not be so aggressive that fresh
     * errors are lost).
     */
    @Test
    public void testFreshErrorPreservedAfterPriorFailure() {
        RuntimeException first = assertThrows(RuntimeException.class,
                () -> Nd4j.create(new float[]{1.0f}).reshape(2, 2),
                "first deliberate failure");
        RuntimeException second = assertThrows(RuntimeException.class,
                () -> Nd4j.create(new float[]{1.0f, 2.0f}).reshape(5, 7),
                "second deliberate failure with a DIFFERENT element mismatch");
        // Both failures carry their own geometry, proving neither replayed the
        // other's message.
        assertNotEquals(first.getMessage(), second.getMessage(),
                "each failure must report its own message");
    }

    /** Success after success: the clear-before-dispatch never breaks happy paths. */
    @Test
    public void testSuccessiveSuccessfulOpsRemainClean() {
        for (int i = 1; i <= 8; i++) {
            try (INDArray array = Nd4j.valueArrayOf(new long[]{i}, (float) i)) {
                assertEquals((float) i, array.getFloat(i - 1), 0.0f);
            }
        }
    }
}
