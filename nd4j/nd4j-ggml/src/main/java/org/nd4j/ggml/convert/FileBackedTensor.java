package org.nd4j.ggml.convert;

import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** CPU conversion storage. The pointer retains the mapped buffer, not a copied payload. */
final class FileBackedTensor {
    private FileBackedTensor() {}

    static INDArray allocate(DataType type, long[] shape) throws IOException {
        long elements = 1;
        for (long dimension : shape) {
            if (dimension < 0) throw new IllegalArgumentException("Negative tensor dimension");
            elements = Math.multiplyExact(elements, dimension);
        }
        long bytes = Math.multiplyExact(elements, type.width());
        if (bytes <= 0 || bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Mapped conversion tensor must contain 1..2147483647 bytes");
        }
        Path temporary = Files.createTempFile("sdx-converted-tensor-", ".bin");
        try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.READ,
                StandardOpenOption.WRITE, StandardOpenOption.DELETE_ON_CLOSE)) {
            var mapping = channel.map(FileChannel.MapMode.READ_WRITE, 0, bytes);
            mapping.order(ByteOrder.nativeOrder());
            // Do not use createBuffer(ByteBuffer): the CPU constructor copies it.
            // JavaCPP's direct-buffer pointer retains the mapping through typed pointer proxies.
            var pointer = new BytePointer(mapping);
            var buffer = Nd4j.createBuffer(pointer, elements, type);
            try {
                return Nd4j.create(buffer, shape, Nd4j.getStrides(shape, 'c'), 0L, 'c');
            } catch (RuntimeException | Error failure) {
                buffer.close();
                throw failure;
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }
}
