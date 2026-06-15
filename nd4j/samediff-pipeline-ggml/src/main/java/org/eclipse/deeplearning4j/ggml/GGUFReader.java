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

package org.eclipse.deeplearning4j.ggml;

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
 * Reader for GGUF format files.
 * Supports reading tensors from GGML/GGUF files used by llama.cpp.
 */
@Slf4j
public class GGUFReader implements Closeable {

    @Getter
    private final File file;
    @Getter
    private final GGUFHeader header;
    private RandomAccessFile raf;
    private FileChannel channel;

    public GGUFReader(File file) throws IOException {
        this.file = file;
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
        this.header = GGUFHeader.fromRandomAccessFile(raf);
        log.debug("Opened GGUF file: {} version {} with {} tensors",
                file.getName(), header.getVersion(), header.getTensorCount());
    }

    public static GGUFReader open(File file) throws IOException {
        return new GGUFReader(file);
    }

    public static GGUFReader open(String path) throws IOException {
        return new GGUFReader(new File(path));
    }

    public Set<String> getTensorNames() {
        return header.getTensorNames();
    }

    public int getTensorCount() {
        return header.getTensorCount();
    }

    public GGUFHeader.TensorInfo getTensorInfo(String name) {
        return header.getTensorInfo(name);
    }

    public Map<String, Object> getMetadata() {
        return header.getMetadata();
    }

    public INDArray readTensor(String name) throws IOException {
        return readTensor(name, true);
    }

    public INDArray readTensor(String name, boolean dequantize) throws IOException {
        GGUFHeader.TensorInfo info = header.getTensorInfo(name);
        if (info == null) {
            throw new IllegalArgumentException("Tensor not found: " + name);
        }
        return readTensorData(info, dequantize);
    }

    public Map<String, INDArray> readAllTensors() throws IOException {
        return readAllTensors(true);
    }

    public Map<String, INDArray> readAllTensors(boolean dequantize) throws IOException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : header.getTensorNames()) {
            try {
                result.put(name, readTensor(name, dequantize));
            } catch (UnsupportedOperationException e) {
                log.warn("Skipping unsupported tensor {}: {}", name, e.getMessage());
            }
        }
        return result;
    }

    public Map<String, INDArray> readTensors(Collection<String> names, boolean dequantize) throws IOException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : names) {
            if (header.getTensorInfo(name) != null) {
                try {
                    result.put(name, readTensor(name, dequantize));
                } catch (UnsupportedOperationException e) {
                    log.warn("Skipping unsupported tensor {}: {}", name, e.getMessage());
                }
            } else {
                log.warn("Tensor not found in file: {}", name);
            }
        }
        return result;
    }

    private INDArray readTensorData(GGUFHeader.TensorInfo info, boolean dequantize) throws IOException {
        long dataOffset = header.getDataOffset() + info.getOffset();
        GGUFType type = info.getType();
        long[] shape = info.getShape();

        if (shape == null || shape.length == 0) {
            shape = new long[]{1};
        }

        long elementCount = info.getElementCount();

        if (type.isQuantized()) {
            if (dequantize) {
                return readAndDequantize(dataOffset, type, shape, elementCount);
            } else {
                return readQuantizedRaw(dataOffset, type, shape, elementCount);
            }
        } else {
            return readUnquantizedTensor(dataOffset, type, shape, elementCount);
        }
    }

    private INDArray readUnquantizedTensor(long dataOffset, GGUFType type, long[] shape, long elementCount) throws IOException {
        raf.seek(dataOffset);
        int bytesPerElement = type.getBytesPerElement();
        byte[] data = new byte[(int) (elementCount * bytesPerElement)];
        raf.readFully(data);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        DataType dtype = type.toNd4jType();

        switch (type) {
            case F32:
                float[] floats = new float[(int) elementCount];
                buffer.asFloatBuffer().get(floats);
                return Nd4j.create(floats, shape, 'c');

            case F16:
            case BF16:
                short[] halfs = new short[(int) elementCount];
                buffer.asShortBuffer().get(halfs);
                DataBuffer db = Nd4j.createTypedBuffer(halfs, dtype);
                return Nd4j.create(db, shape, Nd4j.getStrides(shape, 'c'), 0, 'c', dtype);

            case F64:
                double[] doubles = new double[(int) elementCount];
                buffer.asDoubleBuffer().get(doubles);
                return Nd4j.create(doubles, shape, 'c');

            case I32:
                int[] ints = new int[(int) elementCount];
                buffer.asIntBuffer().get(ints);
                return Nd4j.createFromArray(ints).reshape(shape);

            case I64:
                long[] longs = new long[(int) elementCount];
                buffer.asLongBuffer().get(longs);
                return Nd4j.createFromArray(longs).reshape(shape);

            case I8:
                byte[] bytes = new byte[(int) elementCount];
                buffer.get(bytes);
                return Nd4j.createFromArray(bytes).reshape(shape);

            case I16:
                short[] shorts = new short[(int) elementCount];
                buffer.asShortBuffer().get(shorts);
                return Nd4j.createFromArray(shorts).reshape(shape);

            default:
                throw new UnsupportedOperationException("Unsupported GGUF type: " + type);
        }
    }

    private INDArray readQuantizedRaw(long dataOffset, GGUFType type, long[] shape, long elementCount) throws IOException {
        long blockSize = type.calculateBlockSize();
        int blockElements = type.getBlockElements();
        long numBlocks = (elementCount + blockElements - 1) / blockElements;
        long totalBytes = numBlocks * blockSize;

        raf.seek(dataOffset);
        byte[] data = new byte[(int) totalBytes];
        raf.readFully(data);

        return Nd4j.createFromArray(data).reshape(1, data.length);
    }

    private INDArray readAndDequantize(long dataOffset, GGUFType type, long[] shape, long elementCount) throws IOException {
        long blockSize = type.calculateBlockSize();
        int blockElements = type.getBlockElements();
        long numBlocks = (elementCount + blockElements - 1) / blockElements;
        long totalBytes = numBlocks * blockSize;

        raf.seek(dataOffset);
        byte[] data = new byte[(int) totalBytes];
        raf.readFully(data);

        float[] result = dequantize(data, type, (int) elementCount);
        return Nd4j.create(result, shape, 'c');
    }

    private float[] dequantize(byte[] data, GGUFType type, int elementCount) {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        float[] result = new float[elementCount];

        switch (type) {
            case Q4_0:
                dequantizeQ4_0(buffer, result);
                break;
            case Q4_1:
                dequantizeQ4_1(buffer, result);
                break;
            case Q8_0:
                dequantizeQ8_0(buffer, result);
                break;
            default:
                log.warn("Dequantization not implemented for {}, returning zeros", type);
        }

        return result;
    }

    private void dequantizeQ4_0(ByteBuffer buffer, float[] result) {
        int blockSize = 32;
        int numBlocks = result.length / blockSize;

        for (int block = 0; block < numBlocks; block++) {
            short scaleHalf = buffer.getShort();
            float scale = halfToFloat(scaleHalf);

            for (int i = 0; i < blockSize / 2; i++) {
                byte b = buffer.get();
                int low = (b & 0x0F) - 8;
                int high = ((b >> 4) & 0x0F) - 8;

                int idx = block * blockSize + i * 2;
                if (idx < result.length) result[idx] = scale * low;
                if (idx + 1 < result.length) result[idx + 1] = scale * high;
            }
        }
    }

    private void dequantizeQ4_1(ByteBuffer buffer, float[] result) {
        int blockSize = 32;
        int numBlocks = result.length / blockSize;

        for (int block = 0; block < numBlocks; block++) {
            short scaleHalf = buffer.getShort();
            short minHalf = buffer.getShort();
            float scale = halfToFloat(scaleHalf);
            float min = halfToFloat(minHalf);

            for (int i = 0; i < blockSize / 2; i++) {
                byte b = buffer.get();
                int low = b & 0x0F;
                int high = (b >> 4) & 0x0F;

                int idx = block * blockSize + i * 2;
                if (idx < result.length) result[idx] = scale * low + min;
                if (idx + 1 < result.length) result[idx + 1] = scale * high + min;
            }
        }
    }

    private void dequantizeQ8_0(ByteBuffer buffer, float[] result) {
        int blockSize = 32;
        int numBlocks = result.length / blockSize;

        for (int block = 0; block < numBlocks; block++) {
            short scaleHalf = buffer.getShort();
            float scale = halfToFloat(scaleHalf);

            for (int i = 0; i < blockSize; i++) {
                byte b = buffer.get();
                int idx = block * blockSize + i;
                if (idx < result.length) result[idx] = scale * b;
            }
        }
    }

    private float halfToFloat(short half) {
        int s = (half >> 15) & 0x1;
        int e = (half >> 10) & 0x1F;
        int m = half & 0x3FF;

        if (e == 0) {
            if (m == 0) {
                return s == 0 ? 0.0f : -0.0f;
            }
            float result = (float) (Math.pow(2, -14) * (m / 1024.0));
            return s == 0 ? result : -result;
        } else if (e == 31) {
            if (m == 0) {
                return s == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }

        float result = (float) (Math.pow(2, e - 15) * (1 + m / 1024.0));
        return s == 0 ? result : -result;
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
