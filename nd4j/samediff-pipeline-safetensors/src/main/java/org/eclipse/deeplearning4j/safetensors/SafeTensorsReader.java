/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.safetensors;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * Reader for SafeTensors format files.
 * Supports reading individual tensors or all tensors from a file.
 */
@Slf4j
public class SafeTensorsReader implements Closeable {

    // Divisible by every supported element width; staging never scales with tensor size.
    private static final int READ_CHUNK_BYTES = 1024 * 1024;

    @Getter
    private final File file;
    @Getter
    private final SafeTensorsHeader header;
    private RandomAccessFile raf;
    private FileChannel channel;

    public SafeTensorsReader(File file) throws IOException {
        this.file = file;
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
        try {
            this.header = SafeTensorsHeader.fromRandomAccessFile(raf);
        } catch (IOException | RuntimeException | Error failure) {
            try {
                raf.close();
            } catch (IOException cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
        log.debug("Opened SafeTensors file: {} with {} tensors", file.getName(), header.getTensorCount());
    }

    public static SafeTensorsReader open(File file) throws IOException {
        return new SafeTensorsReader(file);
    }

    public static SafeTensorsReader open(String path) throws IOException {
        return new SafeTensorsReader(new File(path));
    }

    public Set<String> getTensorNames() {
        return header.getTensorNames();
    }

    public int getTensorCount() {
        return header.getTensorCount();
    }

    public SafeTensorsHeader.TensorInfo getTensorInfo(String name) {
        return header.getTensorInfo(name);
    }

    public INDArray readTensor(String name) throws IOException {
        SafeTensorsHeader.TensorInfo info = header.getTensorInfo(name);
        if (info == null) {
            throw new IllegalArgumentException("Tensor not found: " + name);
        }
        return readTensorData(info);
    }

    public Map<String, INDArray> readAllTensors() throws IOException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : header.getTensorNames()) {
            result.put(name, readTensor(name));
        }
        return result;
    }

    public Map<String, INDArray> readTensors(Collection<String> names) throws IOException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : names) {
            if (header.getTensorInfo(name) != null) {
                result.put(name, readTensor(name));
            } else {
                log.warn("Tensor not found in file: {}", name);
            }
        }
        return result;
    }

    private INDArray readTensorData(SafeTensorsHeader.TensorInfo info) throws IOException {
        DataType dtype = info.getSafeTensorsDtype().toNd4jType();
        long[] shape = info.getShape();
        long[] offsets = info.getDataOffsets();
        if (shape == null || offsets == null || offsets.length != 2
                || offsets[0] < 0 || offsets[1] < offsets[0]) {
            throw new IOException("Invalid shape or data offsets for tensor: " + info.getName());
        }
        boolean empty = false;
        for (long dim : shape) {
            if (dim < 0) {
                throw new IOException("Negative dimension for tensor: " + info.getName());
            }
            empty |= dim == 0;
        }
        long dataOffset;
        long dataEnd;
        long expectedBytes;
        int width = dtype.width();
        try {
            long elements = empty ? 0 : 1; // [] is a scalar, not [1].
            if (!empty) {
                for (long dim : shape) {
                    elements = Math.multiplyExact(elements, dim);
                }
            }
            expectedBytes = Math.multiplyExact(elements, width);
            dataOffset = Math.addExact(header.getDataOffset(), offsets[0]);
            dataEnd = Math.addExact(header.getDataOffset(), offsets[1]);
        } catch (ArithmeticException failure) {
            throw new IOException("Tensor size or offset overflow: " + info.getName(), failure);
        }
        long dataLength = offsets[1] - offsets[0];
        if (dataLength != expectedBytes) {
            throw new IOException("Tensor payload size mismatch: " + info.getName()
                    + " (expected " + expectedBytes + ", got " + dataLength + ")");
        }
        if (dataEnd > channel.size()) {
            throw new EOFException("Truncated tensor payload: " + info.getName());
        }

        // The returned array owns its storage, independent of the reader and caller workspace.
        try (MemoryWorkspace ignored = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            INDArray result = Nd4j.createUninitialized(dtype, shape, 'c');
            try {
                if (dataLength == 0) {
                    return result;
                }
                byte[] bytes = new byte[(int) Math.min(dataLength, READ_CHUNK_BYTES)];
                // Borrow the array's host pointer; do not close/deallocate this alias.
                BytePointer destination = new BytePointer(result.data().pointer()).capacity(dataLength);
                for (long offset = 0; offset < dataLength; ) {
                    int count = (int) Math.min(dataLength - offset, bytes.length);
                    ByteBuffer chunk = ByteBuffer.wrap(bytes, 0, count);
                    long position = dataOffset + offset;
                    while (chunk.hasRemaining()) {
                        int read = channel.read(chunk, position);
                        if (read < 0) {
                            throw new EOFException("Truncated tensor payload: " + info.getName());
                        }
                        if (read == 0) {
                            throw new IOException("No progress reading tensor: " + info.getName());
                        }
                        position += read;
                    }
                    // SafeTensors is little-endian; preserve storage bits, not numeric casts.
                    if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN && width > 1) {
                        for (int base = 0; base < count; base += width) {
                            for (int i = 0; i < width / 2; i++) {
                                byte value = bytes[base + i];
                                bytes[base + i] = bytes[base + width - 1 - i];
                                bytes[base + width - 1 - i] = value;
                            }
                        }
                    }
                    destination.position(offset).put(bytes, 0, count);
                    offset += count;
                }
                Nd4j.getAffinityManager().tagLocation(result, AffinityManager.Location.HOST);
                return result;
            } catch (IOException | RuntimeException | Error failure) {
                try {
                    result.close();
                } catch (RuntimeException | Error cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
                throw failure;
            }
        }
    }

    public static Map<String, INDArray> loadFile(File file) throws IOException {
        try (SafeTensorsReader reader = new SafeTensorsReader(file)) {
            return reader.readAllTensors();
        }
    }

    public static Map<String, INDArray> loadFiles(List<File> files) throws IOException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (File file : files) {
            try (SafeTensorsReader reader = new SafeTensorsReader(file)) {
                result.putAll(reader.readAllTensors());
            }
        }
        return result;
    }

    @Override
    public void close() throws IOException {
        if (channel != null) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }
}
