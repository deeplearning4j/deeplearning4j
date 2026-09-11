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
    @org.junit.jupiter.api.condition.EnabledOnOs(org.junit.jupiter.api.condition.OS.LINUX)
    void nativePointerRemainsInFileMappingAfterGraphRegistration() throws Exception {
        try (var array = FileBackedTensor.allocate(DataType.FLOAT, new long[]{2048, 1024});
             var graph = org.nd4j.autodiff.samediff.SameDiff.create()) {
            array.assign(3);
            graph.var("weight", array);
            System.gc();
            var stored = graph.getVariable("weight").getArr();
            long address = stored.data().addressPointer().address();
            boolean mapped = false;
            for (String line : java.nio.file.Files.readAllLines(java.nio.file.Path.of("/proc/self/maps"))) {
                if (!line.contains("sdx-converted-tensor-")) continue;
                String[] bounds = line.substring(0, line.indexOf(' ')).split("-");
                long start = Long.parseUnsignedLong(bounds[0], 16);
                long end = Long.parseUnsignedLong(bounds[1], 16);
                if (Long.compareUnsigned(address, start) >= 0 && Long.compareUnsigned(address, end) < 0) {
                    mapped = true;
                    break;
                }
            }
            assertTrue(mapped, "Graph weight must reference mapped file pages, not a copied native allocation");
            assertEquals(3, stored.getDouble(2047, 1023), 0);
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
