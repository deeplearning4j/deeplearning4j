package org.nd4j.ggml.convert;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

class FileBackedTensorTest {
    @Test
    void mappedStorageSurvivesChannelCloseAndSupportsViewsAndCasts() throws Exception {
        assertTrue(Nd4j.getBackend().getClass().getName().contains(".cpu."));
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE, DataType.HALF, DataType.BFLOAT16}) {
            try (var array = FileBackedTensor.allocate(type, new long[]{3, 5})) {
                array.assign(2);
                array.putScalar(new long[]{2, 4}, 7);
                System.gc();
                assertEquals(type, array.dataType());
                assertArrayEquals(new long[]{3, 5}, array.shape());
                assertEquals(7, array.getDouble(2, 4), 0);
                assertEquals(2, array.getDouble(0, 0), 0);
                try (var copy = array.dup()) {
                    assertEquals(array, copy);
                }
                assertEquals(7, array.transpose().getDouble(4, 2), 0);
            }
        }
    }

    @Test
    void invalidAndOversizedMappingsFailBeforeAllocation() {
        assertThrows(IllegalArgumentException.class,
                () -> FileBackedTensor.allocate(DataType.FLOAT, new long[]{0}));
        assertThrows(IllegalArgumentException.class,
                () -> FileBackedTensor.allocate(DataType.FLOAT, new long[]{-1}));
        assertThrows(IllegalArgumentException.class,
                () -> FileBackedTensor.allocate(DataType.FLOAT, new long[]{Integer.MAX_VALUE}));
        assertThrows(ArithmeticException.class,
                () -> FileBackedTensor.allocate(DataType.DOUBLE, new long[]{Long.MAX_VALUE}));
    }
}
