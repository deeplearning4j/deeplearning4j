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

package org.nd4j.torchscript.format;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.torchscript.TorchScriptImportException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("SafetensorsReader Test")
class SafetensorsReaderTest {

    @TempDir
    File tempDir;

    @Test
    @DisplayName("Test read header")
    void testReadHeader() throws IOException, TorchScriptImportException {
        File file = createSafetensorsFile("test_tensor", TorchScriptDataType.FLOAT32, new long[]{2, 3});

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            SafetensorsHeader header = reader.readHeader();

            assertNotNull(header);
            assertEquals(1, header.getTensorCount());
        }
    }

    @Test
    @DisplayName("Test read tensor infos")
    void testReadTensorInfos() throws IOException, TorchScriptImportException {
        File file = createSafetensorsFile("my_weight", TorchScriptDataType.FLOAT32, new long[]{4, 5});

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            List<TorchScriptTensorInfo> tensors = reader.readTensorInfos();

            assertEquals(1, tensors.size());

            TorchScriptTensorInfo info = tensors.get(0);
            assertEquals("my_weight", info.getName());
            assertEquals(TorchScriptDataType.FLOAT32, info.getDataType());
            assertArrayEquals(new long[]{4, 5}, info.getShape());
        }
    }

    @Test
    @DisplayName("Test read tensor data")
    void testReadTensorData() throws IOException, TorchScriptImportException {
        File file = createSafetensorsFileWithData();

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            List<TorchScriptTensorInfo> tensors = reader.readTensorInfos();

            assertEquals(1, tensors.size());

            byte[] data = reader.readTensorData(tensors.get(0));

            // Should be 6 floats = 24 bytes
            assertEquals(24, data.length);

            // Verify the data values
            ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(1.0f, buffer.getFloat(), 0.001f);
            assertEquals(2.0f, buffer.getFloat(), 0.001f);
            assertEquals(3.0f, buffer.getFloat(), 0.001f);
            assertEquals(4.0f, buffer.getFloat(), 0.001f);
            assertEquals(5.0f, buffer.getFloat(), 0.001f);
            assertEquals(6.0f, buffer.getFloat(), 0.001f);
        }
    }

    @Test
    @DisplayName("Test map tensor data")
    void testMapTensorData() throws IOException, TorchScriptImportException {
        File file = createSafetensorsFileWithData();

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            List<TorchScriptTensorInfo> tensors = reader.readTensorInfos();

            MappedByteBuffer mapped = reader.mapTensorData(tensors.get(0));

            assertNotNull(mapped);
            assertEquals(24, mapped.capacity());

            mapped.order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(1.0f, mapped.getFloat(), 0.001f);
        }
    }

    @Test
    @DisplayName("Test get metadata")
    void testGetMetadata() throws IOException, TorchScriptImportException {
        File file = createSafetensorsFile("layer1.weight", TorchScriptDataType.FLOAT32, new long[]{64, 32});

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            TorchScriptMetadata metadata = reader.getMetadata();

            assertNotNull(metadata);
            assertEquals(TorchScriptFormat.SAFETENSORS, metadata.getFormat());
            assertEquals(1, metadata.getTensors().size());
            assertTrue(metadata.getTotalParameters() > 0);
        }
    }

    @Test
    @DisplayName("Test get data offset")
    void testGetDataOffset() throws IOException, TorchScriptImportException {
        File file = createSafetensorsFile("test", TorchScriptDataType.FLOAT32, new long[]{2, 2});

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            reader.readHeader();

            long offset = reader.getDataOffset();
            assertTrue(offset > 8); // At least header size bytes + header
        }
    }

    @Test
    @DisplayName("Test get file")
    void testGetFile() throws IOException {
        File file = createSafetensorsFile("test", TorchScriptDataType.FLOAT32, new long[]{2, 2});

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            assertEquals(file, reader.getFile());
        }
    }

    @Test
    @DisplayName("Test invalid header size throws exception")
    void testInvalidHeaderSize() throws IOException {
        File file = new File(tempDir, "invalid.safetensors");
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(16);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            // Write an invalid header size (larger than file)
            buffer.putLong(1000000000L);
            buffer.put((byte) '{');
            fos.write(buffer.array());
        }

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            assertThrows(TorchScriptImportException.class, reader::readHeader);
        }
    }

    @Test
    @DisplayName("Test multiple tensors")
    void testMultipleTensors() throws IOException, TorchScriptImportException {
        File file = createMultiTensorSafetensorsFile();

        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            List<TorchScriptTensorInfo> tensors = reader.readTensorInfos();

            assertEquals(3, tensors.size());

            // Verify tensor names exist
            assertTrue(tensors.stream().anyMatch(t -> t.getName().equals("weight1")));
            assertTrue(tensors.stream().anyMatch(t -> t.getName().equals("weight2")));
            assertTrue(tensors.stream().anyMatch(t -> t.getName().equals("bias")));
        }
    }

    @Test
    @DisplayName("Test different data types")
    void testDifferentDataTypes() throws IOException, TorchScriptImportException {
        // Test with float16
        File float16File = createSafetensorsFile("fp16_tensor", TorchScriptDataType.FLOAT16, new long[]{4, 4});
        try (SafetensorsReader reader = new SafetensorsReader(float16File)) {
            List<TorchScriptTensorInfo> tensors = reader.readTensorInfos();
            assertEquals(TorchScriptDataType.FLOAT16, tensors.get(0).getDataType());
        }

        // Test with int64
        File int64File = createSafetensorsFile("int64_tensor", TorchScriptDataType.INT64, new long[]{10});
        try (SafetensorsReader reader = new SafetensorsReader(int64File)) {
            List<TorchScriptTensorInfo> tensors = reader.readTensorInfos();
            assertEquals(TorchScriptDataType.INT64, tensors.get(0).getDataType());
        }
    }

    // Helper methods

    private File createSafetensorsFile(String tensorName, TorchScriptDataType dtype, long[] shape)
            throws IOException {
        File file = new File(tempDir, tensorName.replace(".", "_") + ".safetensors");

        long numElements = 1;
        for (long dim : shape) {
            numElements *= dim;
        }
        long dataSize = numElements * dtype.getBytesPerElement();

        String header = String.format(
                "{\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[0,%d]}}",
                tensorName,
                dtype.getSafetensorsName(),
                arrayToJson(shape),
                dataSize
        );
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate((int) (8 + headerBytes.length + dataSize));
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            buffer.putLong(headerBytes.length);
            buffer.put(headerBytes);

            // Write dummy data
            for (int i = 0; i < dataSize; i++) {
                buffer.put((byte) 0);
            }

            fos.write(buffer.array());
        }

        return file;
    }

    private File createSafetensorsFileWithData() throws IOException {
        File file = new File(tempDir, "with_data.safetensors");

        String header = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(8 + headerBytes.length + 24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            buffer.putLong(headerBytes.length);
            buffer.put(headerBytes);

            // Write 6 floats with specific values
            buffer.putFloat(1.0f);
            buffer.putFloat(2.0f);
            buffer.putFloat(3.0f);
            buffer.putFloat(4.0f);
            buffer.putFloat(5.0f);
            buffer.putFloat(6.0f);

            fos.write(buffer.array());
        }

        return file;
    }

    private File createMultiTensorSafetensorsFile() throws IOException {
        File file = new File(tempDir, "multi_tensor.safetensors");

        // weight1: 4x4 float32 = 64 bytes at offset 0
        // weight2: 2x2 float32 = 16 bytes at offset 64
        // bias: 4 float32 = 16 bytes at offset 80
        String header = "{" +
                "\"weight1\":{\"dtype\":\"F32\",\"shape\":[4,4],\"data_offsets\":[0,64]}," +
                "\"weight2\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[64,80]}," +
                "\"bias\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[80,96]}" +
                "}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(8 + headerBytes.length + 96);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            buffer.putLong(headerBytes.length);
            buffer.put(headerBytes);

            // Write dummy data (96 bytes)
            for (int i = 0; i < 96; i++) {
                buffer.put((byte) 0);
            }

            fos.write(buffer.array());
        }

        return file;
    }

    private String arrayToJson(long[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}
