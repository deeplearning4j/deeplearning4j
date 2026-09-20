/*
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Run with CUDA compute-sanitizer memcheck as well as numerical assertions.
 *  No allocation-stream barrier is permitted before the first copy.
 */
@NativeTag
@Tag("cuda-only")
public class CudaAllocationReadinessTest {
    private static final int ELEMENTS = 6720;
    private static final long BYTES = ELEMENTS * 4L; // allocator adds eight bytes
    private static final int REPETITIONS = 64;

    private static NativeOps cudaOps() {
        assumeTrue("CUDA".equalsIgnoreCase(Nd4j.getExecutioner()
                .getEnvironmentInformation().getProperty("backend")), "CUDA allocation ordering");
        return NativeOpsHolder.getInstance().getDeviceNativeOps();
    }

    private static float[] pattern(int iteration) {
        float[] result = new float[ELEMENTS];
        for (int i = 0; i < result.length; i++) result[i] = iteration * 8192 + i - 0.25f;
        return result;
    }

    @ParameterizedTest
    @CsvSource({"0,false", "17,false", "0,true", "17,true"})
    void rawCopyOrdersAllocationAndInteriorPointers(int offset, boolean sameStream) {
        NativeOps ops = cudaOps();
        OpaqueLaunchContext context = ops.defaultLaunchContext();
        Pointer execution = ops.lcExecutionStream(context);
        Pointer copy = ops.lcCopyStream(context);
        // Compare CUDA handles, not the addresses of their native storage cells.
        assertNotEquals(new PointerPointer<Pointer>(execution).get(0).address(),
                new PointerPointer<Pointer>(copy).get(0).address());
        Pointer consumer = sameStream ? execution : copy;
        int device = ops.getDevice();
        for (int iteration = 0; iteration < REPETITIONS; iteration++) {
            float[] expected = pattern(iteration);
            try (FloatPointer input = new FloatPointer(expected);
                 FloatPointer output = new FloatPointer(ELEMENTS)) {
                Pointer allocation = ops.mallocDevice(BYTES, device, 0);
                assertNotNull(allocation);
                assertFalse(allocation.isNull(), ops.lastErrorMessage());
                try {
                    // First access is interior in offset cases; no preceding base copy
                    // may accidentally supply the missing allocation dependency.
                    BytePointer destination = new BytePointer(allocation).position(offset * 4L);
                    BytePointer source = new BytePointer(input).position(offset * 4L);
                    assertEquals(1, ops.memcpyAsync(destination, source, BYTES - offset * 4L, 1, consumer),
                            ops.lastErrorMessage());
                    if (offset != 0)
                        assertEquals(1, ops.memcpyAsync(allocation, input, offset * 4L, 1, consumer));
                    BytePointer hostDestination = new BytePointer(output).position(offset * 4L);
                    assertEquals(1, ops.memcpyAsync(hostDestination, destination,
                            BYTES - offset * 4L, 2, consumer));
                    if (offset != 0)
                        assertEquals(1, ops.memcpyAsync(output, allocation, offset * 4L, 2, consumer));
                    // Completion AFTER the copies is necessary to inspect host results,
                    // and cannot repair a missing allocation -> first-copy edge.
                    assertEquals(1, ops.streamSynchronize(consumer));
                    float[] actual = new float[ELEMENTS];
                    output.get(actual);
                    assertArrayEquals(expected, actual, 0.0f, "generation " + iteration);
                } finally {
                    // The next allocation may reuse this address. Its readiness must
                    // describe the new allocation, never the retired event generation.
                    assertEquals(1, ops.freeDevice(allocation, device));
                }
            }
        }
    }

    @Test
    void pointerConstructorAndDataBufferRoundTrip() {
        NativeOps ops = cudaOps();
        for (int iteration = 0; iteration < 16; iteration++) {
            float[] expected = pattern(iteration);
            try (FloatPointer input = new FloatPointer(expected);
                 FloatPointer output = new FloatPointer(ELEMENTS)) {
                FloatIndexer indexer = FloatIndexer.create(input);
                try (DataBuffer buffer = Nd4j.createBuffer(input, DataType.FLOAT, ELEMENTS, indexer)) {
                    // Match FloatDataBufferTest.testPointerCreation, but verify device
                    // contents rather than merely reading the original primary pointer.
                    assertEquals(1, ops.memcpySync(output, ops.dbSpecialBuffer(buffer.opaqueBuffer()),
                            BYTES, 2, null));
                    float[] actual = new float[ELEMENTS];
                    output.get(actual);
                    assertArrayEquals(expected, actual, 0.0f);
                    ops.dbTickDeviceWrite(buffer.opaqueBuffer());
                    ops.dbForceSyncToPrimary(buffer.opaqueBuffer());
                    assertArrayEquals(expected, buffer.asFloat(), 0.0f);
                } finally {
                    indexer.release();
                }
            }
        }
    }

    @Test
    void freshAllocationD2HHasReadinessWithoutPriorH2D() {
        NativeOps ops = cudaOps();
        Pointer copy = ops.lcCopyStream(ops.defaultLaunchContext());
        for (int iteration = 0; iteration < REPETITIONS; iteration++) {
            try (FloatPointer output = new FloatPointer(ELEMENTS)) {
                OpaqueDataBuffer buffer = ops.allocateDataBuffer(ELEMENTS, DataType.FLOAT.toInt(), true);
                assertNotNull(buffer);
                assertFalse(buffer.isNull(), ops.lastErrorMessage());
                try {
                    // memcheck (not initcheck) oracle: the contents are unspecified,
                    // but reading an allocated range must not be use-before-alloc.
                    // No earlier H2D or stream synchronization can mask source lookup.
                    assertEquals(1, ops.memcpyAsync(output, ops.dbSpecialBuffer(buffer), BYTES, 2, copy));
                    assertEquals(1, ops.streamSynchronize(copy));
                } finally {
                    ops.deleteDataBuffer(buffer);
                }
                buffer = ops.allocateDataBuffer(ELEMENTS, DataType.FLOAT.toInt(), true);
                assertNotNull(buffer);
                assertFalse(buffer.isNull(), ops.lastErrorMessage());
                try {
                    // Independently exercise DataBuffer's D2H path as the first use.
                    ops.dbForceSyncToPrimary(buffer);
                    assertEquals(0, ops.lastErrorCode(), ops.lastErrorMessage());
                } finally {
                    ops.deleteDataBuffer(buffer);
                }
            }
        }
    }
}
