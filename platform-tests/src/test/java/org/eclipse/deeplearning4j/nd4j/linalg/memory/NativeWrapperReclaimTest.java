/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.memory;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.ShapeInfoInterner;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeBufferOwner;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.nativeblas.OpaqueNDArrayArr;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Native wrappers must be freed once unreachable, and shape information, TAD packs and constant
 * buffers must be shared per content. Each case here leaked native memory on every op call:
 * wrappers of views and workspace arrays were marked constant and never freed, reduction axes were
 * constant arrays, every shape inference built a new constant shape buffer, and the constant caches
 * keyed entries by the caller's array, which the caller could change after the lookup.
 * <p>
 * The helpers that create registered objects return only unique IDs, so that nothing on the test's
 * stack keeps the phantom referents reachable while the test waits for their cleanup.
 */
@NativeTag
public class NativeWrapperReclaimTest extends BaseNd4jTestWithBackends {

    private static final WorkspaceConfiguration WORKSPACE = WorkspaceConfiguration.builder()
            .initialSize(1024 * 1024)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.NONE)
            .build();

    private static boolean isCudaBackend() {
        String backendName = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        return backendName.contains("cuda") || backendName.contains("jcublas");
    }

    private static boolean pointerGcEnabled() {
        return !Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.NO_ARRAY_GC, "false"))
                && !Boolean.parseBoolean(System.getProperty("org.bytedeco.javacpp.nopointergc", "false"));
    }

    private static void awaitRegistrationRetired(DeallocatorService service, long uniqueId, String message)
            throws InterruptedException {
        for (int attempt = 0; attempt < 40 && service.getReferenceMap().containsKey(uniqueId); attempt++) {
            System.gc();
            Thread.sleep(25L);
            service.flushCollectedReferences();
        }
        assertFalse(service.getReferenceMap().containsKey(uniqueId), message);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuffersFromEveryAllocationPathShareOneOwner(Nd4jBackend backend) {
        INDArray created = Nd4j.create(DataType.FLOAT, 3, 4).assign(1);
        INDArray fromArray = Nd4j.createFromArray(1L, 2L, 3L);
        INDArray result = created.add(2.0);
        INDArray copy = result.dup();

        NativeBufferOwner owner = created.data().opaqueBuffer().backendOwner();
        assertNotNull(owner, "Buffer has no native owner");
        assertSame(owner, fromArray.data().opaqueBuffer().backendOwner(), "Array created from values has another owner");
        assertSame(owner, result.data().opaqueBuffer().backendOwner(), "Op result has another owner");
        assertSame(owner, copy.data().opaqueBuffer().backendOwner(), "Duplicated array has another owner");

        try (MemoryWorkspace ignored = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WORKSPACE, "NativeWrapperReclaimTest-owner")) {
            INDArray attached = Nd4j.create(DataType.FLOAT, 2, 2).assign(3);
            assertTrue(attached.isAttached());
            assertSame(owner, attached.data().opaqueBuffer().backendOwner(), "Workspace array has another owner");

            // Wrapping arrays together requires them to share one owner instance
            try (OpaqueNDArrayArr wrapped = OpaqueNDArrayArr.createFrom(created, fromArray, result, copy, attached)) {
                assertFalse(wrapped.isNull());
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testViewWrappersAreFreedWhenUnreachable(Nd4jBackend backend) throws Exception {
        assumeTrue(pointerGcEnabled(), "Wrapper cleanup requires pointer GC");
        DeallocatorService service = Nd4j.getDeallocatorService();
        INDArray parent = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 12).reshape(3, 4);

        long cachedId = viewWrapperId(service, parent, true);
        long uncachedId = viewWrapperId(service, parent, false);
        awaitRegistrationRetired(service, cachedId, "Cached wrapper of an unreachable view was never freed");
        awaitRegistrationRetired(service, uncachedId, "Uncached wrapper of an unreachable view was never freed");

        // Freeing the wrappers must leave the viewed memory alone
        assertEquals(78.0, parent.sumNumber().doubleValue(), 1e-5);
    }

    private static long viewWrapperId(DeallocatorService service, INDArray parent, boolean cached) {
        INDArray view = parent.getRow(1);
        assertTrue(view.isView());
        assertFalse(view.closeable());

        OpaqueNDArray wrapper = cached ? OpaqueNDArray.fromINDArray(view) : OpaqueNDArray.fromINDArrayUncached(view);
        assertEquals(4, wrapper.length());
        assertFalse(wrapper.isConstant(), "View wrapper is constant and would never be freed");
        long id = wrapper.getDeallocator().getUniqueId();
        assertTrue(service.getReferenceMap().containsKey(id), "View wrapper is not registered for cleanup");
        return id;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWorkspaceWrappersAreFreedWhenUnreachable(Nd4jBackend backend) throws Exception {
        assumeTrue(pointerGcEnabled(), "Wrapper cleanup requires pointer GC");
        DeallocatorService service = Nd4j.getDeallocatorService();

        long[] ids = workspaceWrapperIds(service);
        awaitRegistrationRetired(service, ids[0], "Wrapper of an unreachable workspace array was never freed");
        awaitRegistrationRetired(service, ids[1], "Native buffer of an unreachable workspace array was never freed");
    }

    private static long[] workspaceWrapperIds(DeallocatorService service) {
        try (MemoryWorkspace ignored = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WORKSPACE, "NativeWrapperReclaimTest-wrapper")) {
            INDArray attached = Nd4j.create(DataType.FLOAT, 3, 4).assign(2);
            assertTrue(attached.isAttached());

            OpaqueNDArray wrapper = OpaqueNDArray.fromINDArray(attached);
            assertFalse(wrapper.isConstant(), "Workspace array wrapper is constant and would never be freed");
            long wrapperId = wrapper.getDeallocator().getUniqueId();
            long bufferId = attached.data().opaqueBuffer().getDeallocator().getUniqueId();
            assertTrue(service.getReferenceMap().containsKey(wrapperId), "Workspace array wrapper is not registered for cleanup");
            assertTrue(service.getReferenceMap().containsKey(bufferId), "Workspace native buffer is not registered for cleanup");
            return new long[]{wrapperId, bufferId};
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testReductionAxesAreFreedWithTheirOp(Nd4jBackend backend) throws Exception {
        assumeTrue(pointerGcEnabled(), "Buffer cleanup requires pointer GC");
        DeallocatorService service = Nd4j.getDeallocatorService();
        INDArray x = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 12).reshape(3, 4);

        long axesId = reductionAxesId(service, x);
        awaitRegistrationRetired(service, axesId, "Axes of an unreachable reduction were never freed");

        assertNull(Shape.ndArrayDimFromLong(), "No dimensions means a full reduction, which has no axes array");
        long dimensionsId = dimensionArrayId(service);
        awaitRegistrationRetired(service, dimensionsId, "Unreachable dimension array was never freed");
    }

    private static long reductionAxesId(DeallocatorService service, INDArray x) {
        Sum sum = new Sum(x, 1L);
        INDArray result = Nd4j.getExecutioner().exec(sum);
        assertArrayEquals(new float[]{10, 26, 42}, result.toFloatVector(), 1e-5f);

        INDArray axes = sum.dimensions();
        assertNotNull(axes, "Reduction has no axes array");
        assertArrayEquals(new long[]{1}, axes.toLongVector());
        assertFalse(axes.data().isConstant(), "Axes array is constant and would never be freed");
        long id = axes.data().opaqueBuffer().getDeallocator().getUniqueId();
        assertTrue(service.getReferenceMap().containsKey(id), "Axes array is not registered for cleanup");
        return id;
    }

    private static long dimensionArrayId(DeallocatorService service) {
        INDArray dimensions = Shape.ndArrayDimFromLong(1L);
        assertArrayEquals(new long[]{1}, dimensions.toLongVector());
        assertFalse(dimensions.isAttached());
        assertFalse(dimensions.data().isConstant(), "Dimension array is constant and would never be freed");
        long id = dimensions.data().opaqueBuffer().getDeallocator().getUniqueId();
        assertTrue(service.getReferenceMap().containsKey(id), "Dimension array is not registered for cleanup");
        return id;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShapeInfoIsSharedPerContent(Nd4jBackend backend) {
        long[] info = LongShapeDescriptor.fromShape(new long[]{7, 5}, DataType.FLOAT).toShapeInfo();
        DataBuffer first = ShapeInfoInterner.intern(info);
        assertSame(first, ShapeInfoInterner.intern(info.clone()), "Equal shape info got separate buffers");
        assertArrayEquals(info, first.asLong());
        assertTrue(first.isConstant());

        long[] transposedInfo = LongShapeDescriptor.fromShape(new long[]{5, 7}, DataType.FLOAT).toShapeInfo();
        DataBuffer transposed = ShapeInfoInterner.intern(transposedInfo);
        assertNotSame(first, transposed, "Different shape info shares a buffer");
        assertArrayEquals(transposedInfo, transposed.asLong());
        assertArrayEquals(info, first.asLong());

        // The cache keys by its own copy of the content, not by the caller's array
        long[] callerInfo = info.clone();
        assertSame(first, ShapeInfoInterner.intern(callerInfo));
        callerInfo[1] = 11;
        assertArrayEquals(info, first.asLong());
        assertSame(first, ShapeInfoInterner.intern(info));

        DataBuffer described = Shape.createShapeInformation(LongShapeDescriptor.fromShape(new long[]{5, 7}, DataType.FLOAT));
        assertSame(transposed, described, "Shape info built from a descriptor is not shared");

        INDArray a = Nd4j.create(DataType.FLOAT, 3, 4);
        List<DataBuffer> firstShapes = Nd4j.getExecutioner().calculateOutputShape(DynamicCustomOp.builder("add").addInputs(a, a).build());
        List<DataBuffer> secondShapes = Nd4j.getExecutioner().calculateOutputShape(DynamicCustomOp.builder("add").addInputs(a, a).build());
        assertEquals(1, firstShapes.size());
        assertEquals(1, secondShapes.size());
        assertSame(firstShapes.get(0), secondShapes.get(0), "Each shape inference built a new shape info buffer");
        long[] outputInfo = firstShapes.get(0).asLong();
        assertEquals(2, Shape.rank(outputInfo));
        assertArrayEquals(new long[]{3, 4}, Shape.shape(outputInfo));
        assertEquals(DataType.FLOAT, Shape.dataType(outputInfo));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTadPacksAreSharedAndOutliveCacheClear(Nd4jBackend backend) {
        INDArray x = Nd4j.create(DataType.FLOAT, 3, 4);
        TadPack rows = Nd4j.getExecutioner().tadShapeInfoAndOffsets(x, new long[]{1});
        assertSame(rows, Nd4j.getExecutioner().tadShapeInfoAndOffsets(x, new long[]{1}), "Each request wrapped the native TAD pack again");
        assertTads(rows, new long[]{0, 4, 8}, 4, 1);
        long[] rowInfo = rows.getTadShapeInfo().asLong();

        Nd4j.clearTADCache();

        INDArray y = Nd4j.create(DataType.FLOAT, 4, 6);
        TadPack columns = Nd4j.getExecutioner().tadShapeInfoAndOffsets(y, new long[]{0});
        assertTads(columns, new long[]{0, 1, 2, 3, 4, 5}, 4, 6);

        // The pack handed out before the clear is still registered, so the clear keeps it and the next
        // request returns it instead of building and registering a duplicate
        assertSame(rows, Nd4j.getExecutioner().tadShapeInfoAndOffsets(x, new long[]{1}), "Clearing the TAD cache made the next request build a duplicate pack");
        assertTads(rows, new long[]{0, 4, 8}, 4, 1);
        assertArrayEquals(rowInfo, rows.getTadShapeInfo().asLong());
    }

    private static void assertTads(TadPack pack, long[] expectedOffsets, long expectedLength, long expectedStride) {
        assertArrayEquals(expectedOffsets, pack.getTadOffsets().asLong());
        long[] info = pack.getTadShapeInfo().asLong();
        long[] shape = Shape.shape(info);
        long[] stride = Shape.stride(info);
        long length = 1;
        for (int i = 0; i < shape.length; i++) {
            length *= shape[i];
            if (shape[i] > 1) {
                assertEquals(expectedStride, stride[i], "TAD stride along dimension " + i + " of " + Arrays.toString(info));
            }
        }
        assertEquals(expectedLength, length, "TAD length of " + Arrays.toString(info));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testConstantBuffersAreKeyedByContent(Nd4jBackend backend) {
        ConstantHandler handler = Nd4j.getConstantHandler();
        // The CPU cache stops adding entries once full, so start from an empty cache
        handler.purgeConstants();
        long a = 1_234_567L;
        long b = 7_654_321L;
        // The changed content below hashes like the original, so it lands in the same bucket
        assertEquals(Arrays.hashCode(new long[]{a + 1, b}), Arrays.hashCode(new long[]{a, b + 31}));

        long[] values = {a + 1, b};
        DataBuffer first = handler.getConstantBuffer(values, DataType.LONG);
        assertArrayEquals(new long[]{a + 1, b}, first.asLong());
        assertSame(first, handler.getConstantBuffer(new long[]{a + 1, b}, DataType.LONG), "Equal content got separate buffers");

        // A cache keyed by the caller's array would now hold {a, b + 31} under the first entry
        values[0] = a;
        values[1] = b + 31;
        DataBuffer changed = handler.getConstantBuffer(values, DataType.LONG);
        assertNotSame(first, changed, "Changed content got the buffer of the original content");
        assertArrayEquals(new long[]{a, b + 31}, changed.asLong());
        assertSame(first, handler.getConstantBuffer(new long[]{a + 1, b}, DataType.LONG), "Original content lost its buffer");
        assertArrayEquals(new long[]{a + 1, b}, first.asLong());

        int[] ints = {3, -5, 7};
        DataBuffer intBuffer = handler.getConstantBuffer(ints, DataType.INT);
        assertSame(intBuffer, handler.getConstantBuffer(ints.clone(), DataType.INT));
        ints[1] = 9;
        DataBuffer changedInts = handler.getConstantBuffer(ints, DataType.INT);
        assertNotSame(intBuffer, changedInts);
        assertArrayEquals(new int[]{3, 9, 7}, changedInts.asInt());
        assertArrayEquals(new int[]{3, -5, 7}, intBuffer.asInt());

        float[] floats = {1.5f, -2.25f, 3.0f};
        DataBuffer floatBuffer = handler.getConstantBuffer(floats, DataType.FLOAT);
        assertSame(floatBuffer, handler.getConstantBuffer(floats.clone(), DataType.FLOAT));
        floats[0] = 4.5f;
        DataBuffer changedFloats = handler.getConstantBuffer(floats, DataType.FLOAT);
        assertNotSame(floatBuffer, changedFloats);
        assertArrayEquals(new float[]{4.5f, -2.25f, 3.0f}, changedFloats.asFloat(), 0f);
        assertArrayEquals(new float[]{1.5f, -2.25f, 3.0f}, floatBuffer.asFloat(), 0f);

        double[] doubles = {0.125, -8.5};
        DataBuffer doubleBuffer = handler.getConstantBuffer(doubles, DataType.DOUBLE);
        assertSame(doubleBuffer, handler.getConstantBuffer(doubles.clone(), DataType.DOUBLE));
        doubles[1] = 6.75;
        DataBuffer changedDoubles = handler.getConstantBuffer(doubles, DataType.DOUBLE);
        assertNotSame(doubleBuffer, changedDoubles);
        assertArrayEquals(new double[]{0.125, 6.75}, changedDoubles.asDouble(), 0.0);
        assertArrayEquals(new double[]{0.125, -8.5}, doubleBuffer.asDouble(), 0.0);

        DataBuffer trueFalse = handler.getConstantBuffer(new boolean[]{true, false}, DataType.BOOL);
        DataBuffer falseTrue = handler.getConstantBuffer(new boolean[]{false, true}, DataType.BOOL);
        assertNotSame(trueFalse, falseTrue, "Boolean constants with different content share a buffer");
        assertEquals(1, trueFalse.getInt(0));
        assertEquals(0, trueFalse.getInt(1));
        assertEquals(0, falseTrue.getInt(0));
        assertEquals(1, falseTrue.getInt(1));

        handler.purgeConstants();
        DataBuffer afterPurge = handler.getConstantBuffer(new long[]{a + 1, b}, DataType.LONG);
        assertArrayEquals(new long[]{a + 1, b}, afterPurge.asLong());
        assertSame(afterPurge, handler.getConstantBuffer(new long[]{a + 1, b}, DataType.LONG), "Cache stopped caching after a purge");
        if (isCudaBackend()) {
            // CUDA constant buffers wrap entries of the native constant cache, which a purge leaves in
            // place, so the Java cache keeps its wrappers rather than create them again
            assertSame(first, afterPurge, "Purge dropped a constant buffer the native cache still holds");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayDescriptorComparesContentAndType(Nd4jBackend backend) {
        assertEqualDescriptors(new ArrayDescriptor(new boolean[]{true, false}, DataType.BOOL), new ArrayDescriptor(new boolean[]{true, false}, DataType.BOOL));
        assertEqualDescriptors(new ArrayDescriptor(new int[]{1, 2}, DataType.INT), new ArrayDescriptor(new int[]{1, 2}, DataType.INT));
        assertEqualDescriptors(new ArrayDescriptor(new float[]{1f, 2f}, DataType.FLOAT), new ArrayDescriptor(new float[]{1f, 2f}, DataType.FLOAT));
        assertEqualDescriptors(new ArrayDescriptor(new double[]{1.0, 2.0}, DataType.DOUBLE), new ArrayDescriptor(new double[]{1.0, 2.0}, DataType.DOUBLE));
        assertEqualDescriptors(new ArrayDescriptor(new long[]{1, 2}, DataType.LONG), new ArrayDescriptor(new long[]{1, 2}, DataType.LONG));

        assertNotEquals(new ArrayDescriptor(new boolean[]{true, false}, DataType.BOOL), new ArrayDescriptor(new boolean[]{false, true}, DataType.BOOL));
        assertNotEquals(new ArrayDescriptor(new int[]{1, 2}, DataType.INT), new ArrayDescriptor(new int[]{2, 1}, DataType.INT));
        assertNotEquals(new ArrayDescriptor(new float[]{1f, 2f}, DataType.FLOAT), new ArrayDescriptor(new float[]{1f, 3f}, DataType.FLOAT));
        assertNotEquals(new ArrayDescriptor(new double[]{1.0, 2.0}, DataType.DOUBLE), new ArrayDescriptor(new double[]{1.0}, DataType.DOUBLE));
        assertNotEquals(new ArrayDescriptor(new long[]{1, 2}, DataType.LONG), new ArrayDescriptor(new long[]{1, 3}, DataType.LONG));

        assertNotEquals(new ArrayDescriptor(new long[]{1, 2}, DataType.LONG), new ArrayDescriptor(new long[]{1, 2}, DataType.INT), "Same content with another type");
        assertNotEquals(new ArrayDescriptor(new int[]{1, 2}, DataType.INT), new ArrayDescriptor(new long[]{1, 2}, DataType.INT), "Same values in another array type");

        boolean[] bools = {true, false};
        ArrayDescriptor boolDescriptor = new ArrayDescriptor(bools, DataType.BOOL);
        ArrayDescriptor boolCopy = boolDescriptor.copy();
        bools[0] = false;
        assertEqualDescriptors(new ArrayDescriptor(new boolean[]{true, false}, DataType.BOOL), boolCopy);
        assertNotEquals(boolDescriptor, boolCopy);

        int[] ints = {1, 2};
        ArrayDescriptor intCopy = new ArrayDescriptor(ints, DataType.INT).copy();
        ints[0] = 5;
        assertEqualDescriptors(new ArrayDescriptor(new int[]{1, 2}, DataType.INT), intCopy);

        float[] floats = {1f, 2f};
        ArrayDescriptor floatCopy = new ArrayDescriptor(floats, DataType.FLOAT).copy();
        floats[0] = 5f;
        assertEqualDescriptors(new ArrayDescriptor(new float[]{1f, 2f}, DataType.FLOAT), floatCopy);

        double[] doubles = {1.0, 2.0};
        ArrayDescriptor doubleCopy = new ArrayDescriptor(doubles, DataType.DOUBLE).copy();
        doubles[0] = 5.0;
        assertEqualDescriptors(new ArrayDescriptor(new double[]{1.0, 2.0}, DataType.DOUBLE), doubleCopy);

        long[] longs = {1, 2};
        ArrayDescriptor longCopy = new ArrayDescriptor(longs, DataType.LONG).copy();
        longs[0] = 5;
        assertEqualDescriptors(new ArrayDescriptor(new long[]{1, 2}, DataType.LONG), longCopy);
    }

    private static void assertEqualDescriptors(ArrayDescriptor expected, ArrayDescriptor actual) {
        assertEquals(expected, actual);
        assertEquals(expected.hashCode(), actual.hashCode());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testClosedBufferWrapperIsFreedWhenUnreachable(Nd4jBackend backend) throws Exception {
        assumeTrue(pointerGcEnabled(), "Wrapper cleanup requires pointer GC");
        DeallocatorService service = Nd4j.getDeallocatorService();

        long id = closedBufferRegistrationId(service);
        // The close freed the data; the cleanup action must still free the native wrapper, and only that
        awaitRegistrationRetired(service, id, "Native wrapper of a closed, unreachable buffer was never freed");
    }

    private static long closedBufferRegistrationId(DeallocatorService service) {
        INDArray array = Nd4j.create(DataType.FLOAT, 64, 64).assign(1);
        assertTrue(array.closeable());
        OpaqueDataBufferDeallocator deallocator = array.data().opaqueBuffer().getDeallocator();
        assertNotNull(deallocator, "Buffer is not registered for cleanup");
        assertFalse(deallocator.isReleaseClaimed());
        long id = deallocator.getUniqueId();

        array.close();
        assertTrue(array.wasClosed());
        assertTrue(deallocator.isReleaseClaimed(), "Close freed the data without claiming the release");
        assertTrue(service.getReferenceMap().containsKey(id), "Close dropped the registration that frees the native wrapper");
        return id;
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
