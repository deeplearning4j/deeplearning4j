package org.nd4j.ggml.convert;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

@EnabledOnOs(OS.LINUX)
class FileBackedSdzTest {
    @Test
    void privateMappedWeightsSurviveExtractionCleanupWithoutChangingArchive(@TempDir Path root) throws Exception {
        Path archive = root.resolve("weights.sdz");
        try (SameDiff graph = SameDiff.create()) {
            graph.var("weight", Nd4j.valueArrayOf(new long[]{2048, 1024}, 3, DataType.FLOAT));
            SDZSerializer.save(graph, archive.toFile(), false, null);
        }
        byte[] before = Files.readAllBytes(archive);
        try (SameDiff graph = SDZSerializer.loadFileBackedCpu(archive.toFile())) {
            var array = graph.getVariable("weight").getArr();
            long address = array.data().addressPointer().address();
            boolean mapped = false;
            for (String line : Files.readAllLines(Path.of("/proc/self/maps"))) {
                if (!line.contains("sdz-serializer-load-")) continue;
                String[] bounds = line.substring(0, line.indexOf(' ')).split("-");
                if (Long.compareUnsigned(address, Long.parseUnsignedLong(bounds[0], 16)) >= 0
                        && Long.compareUnsigned(address, Long.parseUnsignedLong(bounds[1], 16)) < 0) {
                    mapped = true;
                    assertTrue(line.contains("rw-p"), "Mapping must be private and writable");
                }
            }
            assertTrue(mapped, "Weight must remain file-backed after extraction directory cleanup");
            System.gc();
            assertEquals(3, array.getDouble(2047, 1023), 0);
            array.putScalar(new long[]{0, 0}, 9);
            assertEquals(9, array.getDouble(0, 0), 0);
        }
        assertArrayEquals(before, Files.readAllBytes(archive));
        try (SameDiff original = SDZSerializer.load(archive.toFile(), false)) {
            assertEquals(3, original.getVariable("weight").getArr().getDouble(0, 0), 0);
        }
    }
}
