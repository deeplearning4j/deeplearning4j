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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
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
        this.header = SafeTensorsHeader.fromRandomAccessFile(raf);
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
        long dataOffset = header.getDataOffset() + info.getDataStart();
        long dataLength = info.getDataLength();
        DataType dtype = info.getSafeTensorsDtype().toNd4jType();
        long[] shape = info.getShape();

        if (shape == null || shape.length == 0) {
            shape = new long[]{1};
        }

        raf.seek(dataOffset);
        byte[] data = new byte[(int) dataLength];
        raf.readFully(data);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        DataBuffer dataBuffer = createDataBuffer(buffer, dtype, shape);

        return Nd4j.create(dataBuffer, shape, Nd4j.getStrides(shape, 'c'), 0, 'c', dtype);
    }

    private DataBuffer createDataBuffer(ByteBuffer buffer, DataType dtype, long[] shape) {
        long numElements = 1;
        for (long dim : shape) {
            numElements *= dim;
        }

        switch (dtype) {
            case FLOAT:
                float[] floats = new float[(int) numElements];
                buffer.asFloatBuffer().get(floats);
                return Nd4j.createBuffer(floats);

            case DOUBLE:
                double[] doubles = new double[(int) numElements];
                buffer.asDoubleBuffer().get(doubles);
                return Nd4j.createBuffer(doubles);

            case HALF:
            case BFLOAT16:
                short[] halfs = new short[(int) numElements];
                buffer.asShortBuffer().get(halfs);
                return Nd4j.createTypedBuffer(halfs, dtype);

            case LONG:
                long[] longs = new long[(int) numElements];
                buffer.asLongBuffer().get(longs);
                return Nd4j.createBuffer(longs);

            case INT:
                int[] ints = new int[(int) numElements];
                buffer.asIntBuffer().get(ints);
                return Nd4j.createBuffer(ints);

            case SHORT:
                short[] shorts = new short[(int) numElements];
                buffer.asShortBuffer().get(shorts);
                return Nd4j.createTypedBuffer(shorts, dtype);

            case BYTE:
            case UBYTE:
            case BOOL:
                byte[] bytes = new byte[(int) numElements];
                buffer.get(bytes);
                return Nd4j.createTypedBuffer(bytes, dtype);

            case UINT16:
                short[] ushorts = new short[(int) numElements];
                buffer.asShortBuffer().get(ushorts);
                return Nd4j.createTypedBuffer(ushorts, dtype);

            case UINT32:
                int[] uints = new int[(int) numElements];
                buffer.asIntBuffer().get(uints);
                return Nd4j.createTypedBuffer(uints, dtype);

            case UINT64:
                long[] ulongs = new long[(int) numElements];
                buffer.asLongBuffer().get(ulongs);
                return Nd4j.createTypedBuffer(ulongs, dtype);

            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dtype);
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
