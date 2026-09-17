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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.pipeline;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.pipeline.ModelFormat;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.eclipse.deeplearning4j.safetensors.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for samediff-pipeline-safetensors module.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SafeTensorsTest extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;

    @Override
    public char ordering() {
        return 'c';
    }

    // ==================== SafeTensorsDtype Tests ====================

    @Test
    public void testSafeTensorsDtypeFromString() {
        assertEquals(SafeTensorsDtype.F32, SafeTensorsDtype.fromString("F32"));
        assertEquals(SafeTensorsDtype.F16, SafeTensorsDtype.fromString("F16"));
        assertEquals(SafeTensorsDtype.BF16, SafeTensorsDtype.fromString("BF16"));
        assertEquals(SafeTensorsDtype.F64, SafeTensorsDtype.fromString("F64"));
        assertEquals(SafeTensorsDtype.I64, SafeTensorsDtype.fromString("I64"));
        assertEquals(SafeTensorsDtype.I32, SafeTensorsDtype.fromString("I32"));
        assertEquals(SafeTensorsDtype.I16, SafeTensorsDtype.fromString("I16"));
        assertEquals(SafeTensorsDtype.I8, SafeTensorsDtype.fromString("I8"));
        assertEquals(SafeTensorsDtype.U8, SafeTensorsDtype.fromString("U8"));
        assertEquals(SafeTensorsDtype.BOOL, SafeTensorsDtype.fromString("BOOL"));
    }

    @Test
    public void testSafeTensorsDtypeToNd4jType() {
        assertEquals(DataType.FLOAT, SafeTensorsDtype.F32.toNd4jType());
        assertEquals(DataType.HALF, SafeTensorsDtype.F16.toNd4jType());
        assertEquals(DataType.BFLOAT16, SafeTensorsDtype.BF16.toNd4jType());
        assertEquals(DataType.DOUBLE, SafeTensorsDtype.F64.toNd4jType());
        assertEquals(DataType.LONG, SafeTensorsDtype.I64.toNd4jType());
        assertEquals(DataType.INT, SafeTensorsDtype.I32.toNd4jType());
        assertEquals(DataType.SHORT, SafeTensorsDtype.I16.toNd4jType());
        assertEquals(DataType.BYTE, SafeTensorsDtype.I8.toNd4jType());
        assertEquals(DataType.UBYTE, SafeTensorsDtype.U8.toNd4jType());
        assertEquals(DataType.BOOL, SafeTensorsDtype.BOOL.toNd4jType());
    }

    @Test
    public void testSafeTensorsDtypeFromNd4jType() {
        assertEquals(SafeTensorsDtype.F32, SafeTensorsDtype.fromNd4jType(DataType.FLOAT));
        assertEquals(SafeTensorsDtype.F16, SafeTensorsDtype.fromNd4jType(DataType.HALF));
        assertEquals(SafeTensorsDtype.BF16, SafeTensorsDtype.fromNd4jType(DataType.BFLOAT16));
        assertEquals(SafeTensorsDtype.F64, SafeTensorsDtype.fromNd4jType(DataType.DOUBLE));
        assertEquals(SafeTensorsDtype.I64, SafeTensorsDtype.fromNd4jType(DataType.LONG));
        assertEquals(SafeTensorsDtype.I32, SafeTensorsDtype.fromNd4jType(DataType.INT));
    }

    @Test
    public void testSafeTensorsDtypeElementSize() {
        assertEquals(4, SafeTensorsDtype.F32.getElementSize());
        assertEquals(2, SafeTensorsDtype.F16.getElementSize());
        assertEquals(2, SafeTensorsDtype.BF16.getElementSize());
        assertEquals(8, SafeTensorsDtype.F64.getElementSize());
        assertEquals(8, SafeTensorsDtype.I64.getElementSize());
        assertEquals(4, SafeTensorsDtype.I32.getElementSize());
        assertEquals(2, SafeTensorsDtype.I16.getElementSize());
        assertEquals(1, SafeTensorsDtype.I8.getElementSize());
        assertEquals(1, SafeTensorsDtype.U8.getElementSize());
        assertEquals(1, SafeTensorsDtype.BOOL.getElementSize());
    }

    // ==================== SafeTensors File Read/Write Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsRoundTrip(Nd4jBackend backend) throws IOException {
        // Create test tensors
        INDArray tensor1 = Nd4j.rand(DataType.FLOAT, 4, 8);
        INDArray tensor2 = Nd4j.rand(DataType.FLOAT, 16);
        INDArray tensor3 = Nd4j.rand(DataType.FLOAT, 2, 3, 4);

        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("layer1.weight", tensor1);
        tensors.put("layer1.bias", tensor2);
        tensors.put("layer2.weight", tensor3);

        // Write to file
        File stFile = tempDir.resolve("test_model.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        assertTrue(stFile.exists());
        assertTrue(stFile.length() > 0);

        // Read back
        try (SafeTensorsReader reader = SafeTensorsReader.open(stFile)) {
            assertEquals(3, reader.getTensorCount());
            assertTrue(reader.getTensorNames().contains("layer1.weight"));
            assertTrue(reader.getTensorNames().contains("layer1.bias"));
            assertTrue(reader.getTensorNames().contains("layer2.weight"));

            INDArray readTensor1 = reader.readTensor("layer1.weight");
            assertArrayEquals(tensor1.shape(), readTensor1.shape());
            assertEquals(DataType.FLOAT, readTensor1.dataType());

            // Check values match
            for (int i = 0; i < tensor1.length(); i++) {
                assertEquals(tensor1.getFloat(i), readTensor1.getFloat(i), 1e-5f);
            }

            INDArray readTensor2 = reader.readTensor("layer1.bias");
            assertArrayEquals(tensor2.shape(), readTensor2.shape());

            INDArray readTensor3 = reader.readTensor("layer2.weight");
            assertArrayEquals(tensor3.shape(), readTensor3.shape());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsReadAllTensors(Nd4jBackend backend) throws IOException {
        INDArray tensor1 = Nd4j.ones(DataType.FLOAT, 10);
        INDArray tensor2 = Nd4j.zeros(DataType.FLOAT, 5, 5);

        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("ones", tensor1);
        tensors.put("zeros", tensor2);

        File stFile = tempDir.resolve("multi_tensor.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        try (SafeTensorsReader reader = SafeTensorsReader.open(stFile)) {
            Map<String, INDArray> readTensors = reader.readAllTensors();
            assertEquals(2, readTensors.size());
            assertTrue(readTensors.containsKey("ones"));
            assertTrue(readTensors.containsKey("zeros"));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsReadSubset(Nd4jBackend backend) throws IOException {
        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("a", Nd4j.ones(5));
        tensors.put("b", Nd4j.ones(5));
        tensors.put("c", Nd4j.ones(5));
        tensors.put("d", Nd4j.ones(5));

        File stFile = tempDir.resolve("subset.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        try (SafeTensorsReader reader = SafeTensorsReader.open(stFile)) {
            Map<String, INDArray> subset = reader.readTensors(Arrays.asList("a", "c"));
            assertEquals(2, subset.size());
            assertTrue(subset.containsKey("a"));
            assertTrue(subset.containsKey("c"));
            assertFalse(subset.containsKey("b"));
            assertFalse(subset.containsKey("d"));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsHeader(Nd4jBackend backend) throws IOException {
        INDArray tensor = Nd4j.rand(DataType.FLOAT, 10, 20);

        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("test_tensor", tensor);

        File stFile = tempDir.resolve("header_test.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        try (SafeTensorsReader reader = SafeTensorsReader.open(stFile)) {
            SafeTensorsHeader header = reader.getHeader();
            assertNotNull(header);
            assertEquals(1, header.getTensorCount());

            SafeTensorsHeader.TensorInfo info = header.getTensorInfo("test_tensor");
            assertNotNull(info);
            assertEquals(SafeTensorsDtype.F32, info.getSafeTensorsDtype());
            assertArrayEquals(new long[]{10, 20}, info.getShape());
            assertEquals(10 * 20 * 4, info.getDataLength()); // float = 4 bytes
        }
    }

    @Test
    public void testSafeTensorsTensorNotFound() throws IOException {
        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("exists", Nd4j.ones(5));

        File stFile = tempDir.resolve("notfound.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        try (SafeTensorsReader reader = SafeTensorsReader.open(stFile)) {
            assertThrows(IllegalArgumentException.class, () -> reader.readTensor("nonexistent"));
        }
    }

    @Test
    public void testSafeTensorsFileNotFound() {
        File nonExistent = new File("/non/existent/file.safetensors");
        assertThrows(IOException.class, () -> SafeTensorsReader.open(nonExistent));
    }

    // ==================== SafeTensorsPipelineLoader Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsPipelineLoaderSupports(Nd4jBackend backend) {
        SafeTensorsPipelineLoader loader = new SafeTensorsPipelineLoader();

        assertTrue(loader.supports(ModelFormat.SAFETENSORS));
        assertFalse(loader.supports(ModelFormat.GGUF));
        assertFalse(loader.supports(ModelFormat.ONNX));
        assertFalse(loader.supports(ModelFormat.PYTORCH));
        assertEquals(ModelFormat.SAFETENSORS, loader.getFormat());
        assertEquals("SafeTensors", loader.getName());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsPipelineLoaderLoadFile(Nd4jBackend backend) throws IOException {
        // Create a simple model
        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("model.weight", Nd4j.rand(DataType.FLOAT, 100, 50));
        tensors.put("model.bias", Nd4j.rand(DataType.FLOAT, 50));

        File stFile = tempDir.resolve("pipeline_model.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        SafeTensorsPipelineLoader loader = new SafeTensorsPipelineLoader();
        var sd = loader.loadModel(stFile, PipelineLoader.LoadConfig.defaults());

        assertNotNull(sd);
        assertEquals(2, sd.variables().size());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeTensorsStaticLoadMethods(Nd4jBackend backend) throws IOException {
        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("static.weight", Nd4j.rand(DataType.FLOAT, 32, 64));

        File stFile = tempDir.resolve("static_load.safetensors").toFile();
        writeSafeTensorsFile(stFile, tensors);

        // Test static loadFile method
        Map<String, INDArray> loaded = SafeTensorsReader.loadFile(stFile);
        assertEquals(1, loaded.size());
        assertTrue(loaded.containsKey("static.weight"));
        assertArrayEquals(new long[]{32, 64}, loaded.get("static.weight").shape());
    }

    // ==================== Helper Methods ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLowPrecisionStorageBits(Nd4jBackend backend) throws IOException {
        // Include every encoding: subnormals, signed zero, Inf and NaN payloads.
        for (SafeTensorsDtype type : new SafeTensorsDtype[]{SafeTensorsDtype.F16,
                SafeTensorsDtype.BF16, SafeTensorsDtype.F8_E4M3, SafeTensorsDtype.F8_E5M2}) {
            int width = type.getElementSize();
            int elements = width == 2 ? 65536 : 256;
            ByteBuffer payload = ByteBuffer.allocate(elements * width).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < elements; i++) {
                if (width == 2) {
                    payload.putShort((short) i);
                } else {
                    payload.put((byte) i);
                }
            }
            File file = writeRawTensor(type.getName(), "[" + elements + "]", 0,
                    payload.capacity(), payload.array());
            INDArray tensor;
            try (SafeTensorsReader reader = SafeTensorsReader.open(file)) {
                tensor = reader.readTensor("tensor");
            }
            // Reader closure must not invalidate the returned allocation.
            try (INDArray owned = tensor) {
                assertEquals(type.toNd4jType(), owned.dataType());
                assertArrayEquals(new long[]{elements}, owned.shape());
                assertRawPayload(payload.array(), owned, width);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLowPrecisionValuesAndDeviceUse(Nd4jBackend backend) throws IOException {
        SafeTensorsDtype[] types = {SafeTensorsDtype.F16, SafeTensorsDtype.BF16,
                SafeTensorsDtype.F8_E4M3, SafeTensorsDtype.F8_E5M2};
        int[][] encodings = {{0x3c00, 0xc000, 0x3800}, {0x3f80, 0xc000, 0x3f00},
                {0x38, 0xc0, 0x30}, {0x3c, 0xc0, 0x38}};
        for (int t = 0; t < types.length; t++) {
            SafeTensorsDtype type = types[t];
            assertEquals(type, SafeTensorsDtype.fromString(type.getName()));
            assertEquals(type, SafeTensorsDtype.fromNd4jType(type.toNd4jType()));
            int width = type.getElementSize();
            ByteBuffer bytes = ByteBuffer.allocate(3 * width).order(ByteOrder.LITTLE_ENDIAN);
            for (int bits : encodings[t]) {
                if (width == 2) bytes.putShort((short) bits);
                else bytes.put((byte) bits);
            }
            File file = writeRawTensor(type.getName(), "[3]", 0, bytes.capacity(), bytes.array());
            try (SafeTensorsReader reader = SafeTensorsReader.open(file);
                 INDArray tensor = reader.readTensor("tensor");
                 INDArray converted = tensor.castTo(DataType.FLOAT)) {
                // Exercises normal H2D coherency on CUDA, not only host-pointer inspection.
                assertArrayEquals(new float[]{1.0f, -2.0f, 0.5f}, converted.toFloatVector(), 0.0f);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBoundedReadChunkBoundary(Nd4jBackend backend) throws IOException {
        int elements = 1024 * 1024 / Long.BYTES + 3;
        ByteBuffer bytes = ByteBuffer.allocate(elements * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < elements; i++) {
            bytes.putLong(0x1234567800000000L + i);
        }
        File file = writeRawTensor("I64", "[" + elements + "]", 0, bytes.capacity(), bytes.array());
        try (SafeTensorsReader reader = SafeTensorsReader.open(file);
             INDArray tensor = reader.readTensor("tensor")) {
            assertEquals(DataType.LONG, tensor.dataType());
            assertRawPayload(bytes.array(), tensor, Long.BYTES);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarAndEmptyShapes(Nd4jBackend backend) throws IOException {
        byte[] scalar = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(1.25f).array();
        File scalarFile = writeRawTensor("F32", "[]", 0, scalar.length, scalar);
        try (SafeTensorsReader reader = SafeTensorsReader.open(scalarFile);
             INDArray tensor = reader.readTensor("tensor")) {
            assertArrayEquals(new long[0], tensor.shape());
            assertEquals(0, tensor.rank());
            assertEquals(1.25, tensor.getDouble(0), 0.0);
        }
        File emptyFile = writeRawTensor("BF16", "[2,0,3]", 0, 0, new byte[0]);
        try (SafeTensorsReader reader = SafeTensorsReader.open(emptyFile);
             INDArray tensor = reader.readTensor("tensor")) {
            assertArrayEquals(new long[]{2, 0, 3}, tensor.shape());
            assertEquals(DataType.BFLOAT16, tensor.dataType());
            assertEquals(0, tensor.length());
        }
    }

    @Test
    public void testInvalidPayloadLengthsBeforeAllocation() throws IOException {
        assertInvalidRawTensor("F32", "[2]", 0, 4, new byte[4], "size mismatch");
        assertInvalidRawTensor("F32", "[1]", 0, 8, new byte[8], "size mismatch");
        assertInvalidRawTensor("F16", "[1]", 0, 1, new byte[1], "size mismatch");
        assertInvalidRawTensor("F32", "[1]", 0, 4, new byte[3], "Truncated");
        assertInvalidRawTensor("F32", "[-1]", 0, 0, new byte[0], "Negative");
        assertInvalidRawTensor("F32", "[1]", 4, 0, new byte[0], "Invalid");
        assertInvalidRawTensor("U8", "[1]", -1, 0, new byte[0], "Invalid");
        assertInvalidRawTensor("F64", "[9223372036854775807,2]", 0, 0, new byte[0], "overflow");
        assertInvalidRawTensor("F64", "[9223372036854775807]", 0, 0, new byte[0], "overflow");
        assertInvalidRawTensor("U8", "[1]", Long.MAX_VALUE - 1, Long.MAX_VALUE, new byte[0], "overflow");
        // A >2 GiB declaration must fail file-range validation, not narrow to int or allocate.
        assertInvalidRawTensor("U8", "[2147483648]", 0, 2147483648L, new byte[0], "Truncated");
    }

    private void assertInvalidRawTensor(String dtype, String shape, long start, long end,
                                        byte[] payload, String message) throws IOException {
        File file = writeRawTensor(dtype, shape, start, end, payload);
        try (SafeTensorsReader reader = SafeTensorsReader.open(file)) {
            IOException failure = assertThrows(IOException.class, () -> reader.readTensor("tensor"));
            assertTrue(failure.getMessage().contains(message), failure.getMessage());
        }
    }

    private File writeRawTensor(String dtype, String shape, long start, long end, byte[] payload)
            throws IOException {
        File file = tempDir.resolve(UUID.randomUUID() + ".safetensors").toFile();
        String json = "{\"tensor\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape
                + ",\"data_offsets\":[" + start + "," + end + "]}}";
        byte[] header = json.getBytes(StandardCharsets.UTF_8);
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(file))) {
            out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).array());
            out.write(header);
            out.write(payload);
        }
        return file;
    }

    private void assertRawPayload(byte[] littleEndian, INDArray tensor, int width) {
        ByteBuffer actual = tensor.data().asNio().duplicate().order(ByteOrder.nativeOrder());
        ByteBuffer expected = ByteBuffer.wrap(littleEndian).order(ByteOrder.LITTLE_ENDIAN);
        for (int offset = 0; offset < littleEndian.length; offset += width) {
            switch (width) {
                case 1: assertEquals(expected.get(offset), actual.get(offset), "byte " + offset); break;
                case 2: assertEquals(expected.getShort(offset), actual.getShort(offset), "byte " + offset); break;
                case 4: assertEquals(expected.getInt(offset), actual.getInt(offset), "byte " + offset); break;
                case 8: assertEquals(expected.getLong(offset), actual.getLong(offset), "byte " + offset); break;
                default: fail("Unexpected element width: " + width);
            }
        }
    }

    /**
     * Write a SafeTensors file with the given tensors.
     * This is a minimal implementation for testing purposes.
     */
    private void writeSafeTensorsFile(File file, Map<String, INDArray> tensors) throws IOException {
        // Build header JSON
        StringBuilder headerJson = new StringBuilder();
        headerJson.append("{");

        long dataOffset = 0;
        List<Map.Entry<String, INDArray>> entries = new ArrayList<>(tensors.entrySet());
        for (int i = 0; i < entries.size(); i++) {
            Map.Entry<String, INDArray> entry = entries.get(i);
            String name = entry.getKey();
            INDArray arr = entry.getValue();

            long dataLength = arr.length() * arr.dataType().width();
            String dtype = SafeTensorsDtype.fromNd4jType(arr.dataType()).name();

            headerJson.append("\"").append(name).append("\":{");
            headerJson.append("\"dtype\":\"").append(dtype).append("\",");
            headerJson.append("\"shape\":[");
            long[] shape = arr.shape();
            for (int j = 0; j < shape.length; j++) {
                headerJson.append(shape[j]);
                if (j < shape.length - 1) headerJson.append(",");
            }
            headerJson.append("],");
            headerJson.append("\"data_offsets\":[").append(dataOffset).append(",").append(dataOffset + dataLength).append("]");
            headerJson.append("}");

            if (i < entries.size() - 1) {
                headerJson.append(",");
            }

            dataOffset += dataLength;
        }
        headerJson.append("}");

        byte[] headerBytes = headerJson.toString().getBytes(StandardCharsets.UTF_8);

        try (RandomAccessFile raf = new RandomAccessFile(file, "rw");
             FileChannel channel = raf.getChannel()) {

            // Write header size (8 bytes, little endian)
            ByteBuffer sizeBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            sizeBuffer.putLong(headerBytes.length);
            sizeBuffer.flip();
            channel.write(sizeBuffer);

            // Write header
            channel.write(ByteBuffer.wrap(headerBytes));

            // Write tensor data
            for (INDArray arr : tensors.values()) {
                ByteBuffer dataBuffer = arr.data().asNio();
                dataBuffer.order(ByteOrder.LITTLE_ENDIAN);
                channel.write(dataBuffer);
            }
        }
    }
}
