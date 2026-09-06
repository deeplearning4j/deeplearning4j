/* ******************************************************************************
 * Copyright (c) Contributors to the Eclipse Foundation
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import java.lang.reflect.InvocationTargetException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.parallel.Isolated;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Exercises the real counter without GPU allocations, using isolated synthetic charges. */
@Tag("cuda-only")
@Isolated("Temporarily injects native accounting states and restores every counter/limit")
class CudaMemoryCounterTransferTest {
    private static final int DEVICE = 10;

    @Test
    void corruptChargesThrowBeforeLimitsOrMutation() throws Exception {
        try (Counters c = new Counters()) {
            for (boolean commit : new boolean[]{false, true}) {
                c.state(0, 0, 8);
                c.reject(0, 8, 1, 8, commit, "source debit exceeds charge");
                c.state(4, 0, 8);
                c.reject(0, 8, 0, 8, commit, "source debit exceeds charge");
                c.state(8, 0, 4);
                c.reject(0, 8, 1, 8, commit, "DEVICE group debit exceeds charge");
                c.reject(0, 8, 0, 8, commit, "DEVICE group debit exceeds charge");
                c.state(-1, 0, 8);
                c.reject(0, 1, 1, 1, commit, "negative source");
                c.state(8, -1, 8);
                c.reject(0, 8, 1, 8, commit, "negative target");
                c.state(8, 0, -1);
                c.reject(0, 8, 1, 8, commit, "negative DEVICE group");
                c.state(8, Long.MAX_VALUE, 8);
                c.reject(0, 8, 1, 8, commit, "target overflow");
                c.state(0, 0, Long.MAX_VALUE);
                c.reject(-1, 0, 1, 1, commit, "DEVICE group overflow");
                c.state(0, 0, 0);
                c.reject(0, -1, 1, 0, commit, "invalid allocation charge or device");
                c.reject(-1, 1, 1, 1, commit, "invalid allocation charge or device");
                c.reject(0, 0, -1, 1, commit, "invalid allocation charge or device");
            }
        }
    }

    @Test
    void capacityRefusalIsUnchangedAndExactNetCapacitySucceeds() throws Exception {
        try (Counters c = new Counters()) {
            c.state(8, 0, 8);
            c.env.setDeviceLimit(0, 8);
            c.env.setDeviceLimit(1, 7);
            c.env.setGroupLimit(DEVICE, 8);
            for (boolean commit : new boolean[]{false, true}) {
                assertFalse(c.transfer(0, 8, 1, 8, commit));
                c.assertState(8, 0, 8);
            }
            c.env.setDeviceLimit(1, 8);
            assertTrue(c.transfer(0, 8, 1, 8, false));
            c.assertState(8, 0, 8);
            assertTrue(c.transfer(0, 8, 1, 8, true));
            c.assertState(0, 8, 8);
            // Lowering a cap below an existing charge must not prevent releasing it.
            c.env.setDeviceLimit(1, 1);
            c.env.setGroupLimit(DEVICE, 1);
            assertTrue(c.transfer(1, 8, 1, 8, true));
            assertTrue(c.transfer(1, 8, 1, 4, true));
            c.assertState(0, 4, 4);
            assertTrue(c.transfer(1, 4, -99, 0, true));
            c.assertState(0, 0, 0);
            assertTrue(c.transfer(-99, 0, -98, 0, true));
            // Group cap rejects positive net growth independently of device caps.
            c.env.setDeviceLimit(0, 8);
            assertFalse(c.transfer(-1, 0, 0, 2, true));
            c.assertState(0, 0, 0);
            assertTrue(c.transfer(-1, 0, 0, 1, true));
            c.assertState(1, 0, 1);
        }
    }

    private static final class Counters implements AutoCloseable {
        final Environment env;
        final Class<?> type;
        final Object counter;
        final long[] original;
        final long[] limits;

        Counters() throws Exception {
            assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner()
                    .getEnvironmentInformation().getProperty("backend")), "CUDA bindings required");
            assumeTrue(Nd4j.getAffinityManager().getNumberOfDevices() >= 2, "Two device counters required");
            env = Nd4j.getEnvironment();
            type = Class.forName("org.nd4j.linalg.jcublas.bindings.Nd4jCuda$MemoryCounter");
            counter = type.getMethod("getInstance").invoke(null);
            original = snapshot();
            for (long value : original) assertTrue(value >= 0, "Pre-existing accounting corruption");
            limits = new long[]{env.getDeviceLimit(0), env.getDeviceLimit(1), env.getGroupLimit(DEVICE)};
        }

        long[] snapshot() throws Exception {
            return new long[]{env.getDeviceCounter(0), env.getDeviceCounter(1),
                    ((Number) type.getMethod("allocatedGroup", int.class).invoke(counter, DEVICE)).longValue()};
        }

        void state(long source, long target, long group) throws Exception {
            long[] before = snapshot();
            // These legacy counter mutators deliberately inject corrupt states;
            // no real allocation is made or freed while the synthetic state is active.
            type.getMethod("countIn", int.class, long.class).invoke(counter, 0, source - before[0]);
            type.getMethod("countIn", int.class, long.class).invoke(counter, 1, target - before[1]);
            type.getMethod("countInGroup", int.class, long.class).invoke(counter, DEVICE, group - before[2]);
        }

        boolean transfer(int source, long from, int target, long to, boolean commit) throws Exception {
            try {
                return (Boolean) type.getMethod("transferDeviceAllocation", int.class, long.class,
                        int.class, long.class, boolean.class).invoke(counter, source, from, target, to, commit);
            } catch (InvocationTargetException e) {
                if (e.getCause() instanceof RuntimeException) throw (RuntimeException) e.getCause();
                throw e;
            }
        }

        void reject(int source, long from, int target, long to, boolean commit, String reason) throws Exception {
            long[] before = snapshot();
            // Invariant errors must take precedence over ordinary cap rejection.
            env.setDeviceLimit(0, 1);
            env.setDeviceLimit(1, 1);
            env.setGroupLimit(DEVICE, 1);
            RuntimeException error = assertThrows(RuntimeException.class,
                    () -> transfer(source, from, target, to, commit));
            assertTrue(error.getMessage().contains(reason), error.getMessage());
            assertArrayEquals(before, snapshot(), "Rejected transfer must not change counters");
        }

        void assertState(long source, long target, long group) throws Exception {
            assertArrayEquals(new long[]{source, target, group}, snapshot());
        }

        @Override
        public void close() throws Exception {
            try {
                state(original[0], original[1], original[2]);
                assertArrayEquals(original, snapshot(), "Restore all original accounting");
            } finally {
                env.setDeviceLimit(0, limits[0]);
                env.setDeviceLimit(1, limits[1]);
                env.setGroupLimit(DEVICE, limits[2]);
            }
        }
    }
}
