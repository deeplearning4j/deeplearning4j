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
    private static final int HOST = 0;
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

    @Test
    void corruptHostChargesThrowBeforeLimitsOrMutation() throws Exception {
        try (HostCounters c = new HostCounters()) {
            for (boolean commit : new boolean[]{false, true}) {
                c.state(0);
                c.rejectReserve(8, 2, commit, "HOST accounting invariant violated");
                c.state(4);
                c.rejectReserve(8, 1, commit, "HOST accounting invariant violated");
                c.rejectReserve(8, 0, commit, "HOST accounting invariant violated");
                c.state(-1);
                c.rejectReserve(0, 3, commit, "HOST accounting invariant violated");
                c.rejectReserve(0, 0, commit, "HOST accounting invariant violated");
                c.state(Long.MAX_VALUE);
                c.rejectReserve(0, 1, commit, "HOST accounting invariant violated");
            }
        }
    }

    @Test
    void invalidHostArgumentsAndDebitsNeverMutateAccounting() throws Exception {
        try (HostCounters c = new HostCounters()) {
            c.state(8);
            for (boolean commit : new boolean[]{false, true}) {
                c.rejectReserve(-1, 1, commit, "invalid HOST charge or growth");
                c.rejectReserve(0, -1, commit, "invalid HOST charge or growth");
            }
            c.rejectRelease(-1, "invalid HOST debit");
            c.rejectRelease(9, "HOST accounting invariant violated");
            // A cap lowered below the charge cannot block releasing that charge.
            c.release(8);
            c.assertState(0);
            c.release(0);
            c.assertState(0);
            c.rejectRelease(1, "HOST accounting invariant violated");
            c.state(-1);
            c.rejectRelease(0, "HOST accounting invariant violated");
        }
    }

    @Test
    void hostExactCapacitySucceedsAndOneByteOverIsUnchanged() throws Exception {
        try (HostCounters c = new HostCounters()) {
            c.state(8);
            c.env.setGroupLimit(HOST, 12);
            for (boolean commit : new boolean[]{false, true}) {
                assertFalse(c.reserve(8, 5, commit));
                c.assertState(8);
            }
            assertTrue(c.reserve(8, 4, false));
            c.assertState(8);
            assertTrue(c.reserve(8, 4, true));
            c.assertState(12);
            c.release(4);
            c.assertState(8);
            c.env.setGroupLimit(HOST, 1);
            for (boolean commit : new boolean[]{false, true}) {
                assertTrue(c.reserve(8, 0, commit));
                c.assertState(8);
            }
        }
    }

    @Test
    void hostReservationsRecheckCapacityAndRollbackExactlyTheirGrowth() throws Exception {
        try (HostCounters c = new HostCounters()) {
            c.state(8); // Two synthetic owners already charged four bytes each.
            c.env.setGroupLimit(HOST, 14);
            assertTrue(c.reserve(4, 4, false));
            c.assertState(8);
            assertTrue(c.reserve(4, 4, false));
            c.assertState(8);
            assertTrue(c.reserve(4, 4, true));
            c.assertState(12);
            // The second owner's earlier dry-run is not a reservation.
            assertFalse(c.reserve(4, 4, true));
            c.assertState(12);
            c.release(4);
            c.assertState(8);
            // Two successful reservations retain independent rollback deltas.
            assertTrue(c.reserve(4, 2, true));
            c.assertState(10);
            assertTrue(c.reserve(4, 4, true));
            c.assertState(14);
            c.release(4);
            c.assertState(10);
            c.release(2);
            c.assertState(8);
        }
    }

    /** HOST coverage needs CUDA bindings, but does not require two physical GPUs. */
    private static final class HostCounters implements AutoCloseable {
        final Environment env;
        final Class<?> type;
        final Object counter;
        final long original;
        final long limit;
        final int softLimit;

        HostCounters() throws Exception {
            assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner()
                    .getEnvironmentInformation().getProperty("backend")), "CUDA bindings required");
            env = Nd4j.getEnvironment();
            type = Class.forName("org.nd4j.linalg.jcublas.bindings.Nd4jCuda$MemoryCounter");
            counter = type.getMethod("getInstance").invoke(null);
            original = snapshot();
            assertTrue(original >= 0, "Pre-existing HOST accounting corruption");
            limit = env.getGroupLimit(HOST);
            softLimit = ((Number) type.getMethod("getSoftLimitPercent").invoke(counter)).intValue();
            // Make synthetic cap cases independent of the machine's free RAM.
            type.getMethod("setSoftLimitPercent", int.class).invoke(counter, 0);
        }

        long snapshot() throws Exception {
            return ((Number) type.getMethod("allocatedGroup", int.class).invoke(counter, HOST)).longValue();
        }

        void state(long allocated) throws Exception {
            // Reset through zero rather than subtracting two extreme signed values.
            // No native allocation/free is performed while synthetic charges are active.
            type.getMethod("countOutGroup", int.class, long.class).invoke(counter, HOST, snapshot());
            type.getMethod("countInGroup", int.class, long.class).invoke(counter, HOST, allocated);
        }

        boolean reserve(long owned, long growth, boolean commit) throws Exception {
            try {
                return (Boolean) type.getMethod("reserveHostGrowth", long.class, long.class, boolean.class)
                        .invoke(counter, owned, growth, commit);
            } catch (InvocationTargetException e) {
                if (e.getCause() instanceof RuntimeException) throw (RuntimeException) e.getCause();
                throw e;
            }
        }

        void release(long bytes) throws Exception {
            try {
                type.getMethod("releaseHostGrowth", long.class).invoke(counter, bytes);
            } catch (InvocationTargetException e) {
                if (e.getCause() instanceof RuntimeException) throw (RuntimeException) e.getCause();
                throw e;
            }
        }

        void rejectReserve(long owned, long growth, boolean commit, String reason) throws Exception {
            long before = snapshot();
            env.setGroupLimit(HOST, 1);
            RuntimeException error = assertThrows(RuntimeException.class, () -> reserve(owned, growth, commit));
            assertTrue(error.getMessage().contains(reason), error.getMessage());
            assertEquals(before, snapshot(), "Rejected HOST reservation must not change accounting");
        }

        void rejectRelease(long bytes, String reason) throws Exception {
            long before = snapshot();
            RuntimeException error = assertThrows(RuntimeException.class, () -> release(bytes));
            assertTrue(error.getMessage().contains(reason), error.getMessage());
            assertEquals(before, snapshot(), "Rejected HOST debit must not change accounting");
        }

        void assertState(long allocated) throws Exception {
            assertEquals(allocated, snapshot(), "HOST accounting");
        }

        @Override
        public void close() throws Exception {
            try {
                state(original);
                assertEquals(original, snapshot(), "Restore original HOST accounting");
            } finally {
                try {
                    env.setGroupLimit(HOST, limit);
                } finally {
                    type.getMethod("setSoftLimitPercent", int.class).invoke(counter, softLimit);
                }
            }
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
