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
