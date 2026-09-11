package org.eclipse.deeplearning4j.nd4j.linalg.api.ndarray;

import java.lang.reflect.Method;
import java.util.stream.Stream;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.OpaqueNDArray;

import static org.junit.jupiter.api.Assertions.*;

/** Calls the C++ dup implementation, not the independent Java INDArray.dup path. */
public class NativeScalarDupTest extends BaseNd4jTestWithBackends {
    static Stream<Arguments> scalars() {
        return Stream.of(DataType.FLOAT, DataType.DOUBLE, DataType.HALF, DataType.BFLOAT16,
                        DataType.INT32, DataType.INT64)
                .flatMap(dtype -> Stream.of('a', 'c', 'f')
                        .flatMap(order -> Stream.of(false, true)
                                .map(view -> Arguments.of(dtype, order, view))));
    }

    @ParameterizedTest
    @MethodSource("scalars")
    void scalarDupPreservesRankValueAndIndependentStorage(DataType dtype, char order, boolean view)
            throws Exception {
        INDArray parent = view ? Nd4j.createFromArray(-7, 3, 9).castTo(dtype) : Nd4j.scalar(dtype, 3);
        INDArray source = view ? parent.get(NDArrayIndex.point(1)) : parent;
        assertEquals(0, source.rank());
        if (view) {
            assertTrue(source.isView());
            assertEquals(1, source.offset());
        }
        checkDup(source, order);
        assertEquals(3, source.getDouble(0), 0);
        if (view) {
            assertEquals(-7, parent.getDouble(0), 0);
            assertEquals(9, parent.getDouble(2), 0);
        }
    }

    @Test
    void nonScalarDupPreservesShapeAndRequestedOrder() throws Exception {
        for (char inputOrder : new char[]{'c', 'f'}) {
            INDArray source = Nd4j.create(DataType.FLOAT, new long[]{2, 3}, inputOrder);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) source.putScalar(new long[]{i, j}, 1 + 3 * i + j);
            }
            for (char order : new char[]{'a', 'c', 'f'}) checkDup(source, order);
        }
        checkDup(Nd4j.ones(DataType.FLOAT, 1, 1), 'c');
    }

    @Test
    void emptyDupRemainsEmpty() throws Exception {
        checkDup(Nd4j.empty(DataType.FLOAT), 'a');
        checkDup(Nd4j.create(DataType.DOUBLE, 2, 0, 3), 'f');
    }

    private static void checkDup(INDArray source, char order) throws Exception {
        // Reflection keeps this platform test compilable with either CPU or CUDA bindings.
        String backend = source.shapeInfoDataBuffer().opaqueBuffer().backendOwner().nativeOps().getClass().getName();
        String binding;
        if (backend.contains("Nd4jCuda")) {
            binding = "org.nd4j.linalg.jcublas.bindings.Nd4jCuda$NDArray";
        } else if (backend.contains("Nd4jCpu")) {
            binding = "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu$NDArray";
        } else {
            throw new IllegalStateException("Native dup test requires CPU or CUDA bindings: " + backend);
        }
        Class<?> type = Class.forName(binding);
        Method dup = type.getMethod("dup", byte.class, boolean.class);
        try (OpaqueNDArray input = OpaqueNDArray.fromINDArrayUncached(source)) {
            // OpaqueNDArray is a @ByVal typedef of NDArray*: JavaCPP stores the pointer in a holder.
            Object nativeInput = type.getConstructor(Pointer.class).newInstance(new PointerPointer<Pointer>(input).get(0));
            assertEquals(source.rank(), type.getMethod("rankOf").invoke(nativeInput));
            assertEquals(source.isEmpty(), type.getMethod("isEmpty").invoke(nativeInput));
            Pointer copy = (Pointer) dup.invoke(nativeInput, (byte) order, false);
            assertNotNull(copy);
            assertFalse(copy.isNull());
            try {
                assertEquals(type.getMethod("dataType").invoke(nativeInput),
                        type.getMethod("dataType").invoke(copy));
                assertEquals(source.isEmpty(), type.getMethod("isEmpty").invoke(copy));
                assertEquals(source.length(), type.getMethod("lengthOf").invoke(copy));
                if (source.isEmpty()) return; // Preserve the existing canonical-empty dup contract.
                assertEquals(source.rank(), type.getMethod("rankOf").invoke(copy));
                for (int dimension = 0; dimension < source.rank(); dimension++) {
                    assertEquals(source.size(dimension), type.getMethod("sizeAt", int.class).invoke(copy, dimension));
                }
                char expectedOrder = source.rank() == 0 ? 'c' : order == 'a' ? source.ordering() : order;
                assertEquals(expectedOrder, type.getMethod("ordering").invoke(copy));
                assertEquals(true, type.getMethod("equalsTo", type, double.class).invoke(copy, nativeInput, 0.0));
                Pointer inputBuffer = (Pointer) type.getMethod("buffer").invoke(nativeInput);
                Pointer copyBuffer = (Pointer) type.getMethod("buffer").invoke(copy);
                assertNotEquals(inputBuffer.address(), copyBuffer.address());
                // Assign a different value through native code and verify the original remains intact.
                try (INDArray changed = Nd4j.valueArrayOf(source.shape(), 17, source.dataType());
                     OpaqueNDArray replacement = OpaqueNDArray.fromINDArrayUncached(changed)) {
                    Object nativeReplacement = type.getConstructor(Pointer.class).newInstance(new PointerPointer<Pointer>(replacement).get(0));
                    type.getMethod("assign", type).invoke(copy, nativeReplacement);
                    assertEquals(true, type.getMethod("equalsTo", type, double.class)
                            .invoke(copy, nativeReplacement, 0.0));
                    assertEquals(false, type.getMethod("equalsTo", type, double.class)
                            .invoke(nativeInput, nativeReplacement, 0.0));
                }
            } finally {
                // dup returns an owning raw pointer, not a JavaCPP-allocated object.
                try (PointerPointer<Pointer> holder = new PointerPointer<>(new Pointer[]{copy})) {
                    input.backendOwner().nativeOps().deleteNDArray(new OpaqueNDArray(holder));
                }
            }
        }
    }
}
