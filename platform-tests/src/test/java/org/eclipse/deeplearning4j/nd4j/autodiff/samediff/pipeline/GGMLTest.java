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
import org.eclipse.deeplearning4j.ggml.*;
import org.eclipse.deeplearning4j.pipeline.ModelFormat;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
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
 * Tests for samediff-pipeline-ggml module.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class GGMLTest extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;

    @Override
    public char ordering() {
        return 'c';
    }

    // ==================== GGUFType Tests ====================

    @Test
    public void testGGUFTypeFromId() {
        assertEquals(GGUFType.F32, GGUFType.fromId(0));
        assertEquals(GGUFType.F16, GGUFType.fromId(1));
        assertEquals(GGUFType.Q4_0, GGUFType.fromId(2));
        assertEquals(GGUFType.Q4_1, GGUFType.fromId(3));
        assertEquals(GGUFType.Q8_0, GGUFType.fromId(8));
        assertEquals(GGUFType.BF16, GGUFType.fromId(30));
    }

    @Test
    public void testGGUFTypeToNd4jType() {
        assertEquals(DataType.FLOAT, GGUFType.F32.toNd4jType());
        assertEquals(DataType.HALF, GGUFType.F16.toNd4jType());
        assertEquals(DataType.BFLOAT16, GGUFType.BF16.toNd4jType());
        assertEquals(DataType.INT, GGUFType.I32.toNd4jType());
    }

    @Test
    public void testGGUFTypeIsQuantized() {
        assertFalse(GGUFType.F32.isQuantized());
        assertFalse(GGUFType.F16.isQuantized());
        assertFalse(GGUFType.BF16.isQuantized());
        assertTrue(GGUFType.Q4_0.isQuantized());
        assertTrue(GGUFType.Q4_1.isQuantized());
        assertTrue(GGUFType.Q5_0.isQuantized());
        assertTrue(GGUFType.Q5_1.isQuantized());
        assertTrue(GGUFType.Q8_0.isQuantized());
        assertTrue(GGUFType.Q4_K.isQuantized());
        assertTrue(GGUFType.Q5_K.isQuantized());
        assertTrue(GGUFType.Q6_K.isQuantized());
    }

    @Test
    public void testGGUFTypeBlockElements() {
        assertEquals(1, GGUFType.F32.getBlockElements());
        assertEquals(1, GGUFType.F16.getBlockElements());
        assertEquals(32, GGUFType.Q4_0.getBlockElements());
        assertEquals(32, GGUFType.Q4_1.getBlockElements());
        assertEquals(32, GGUFType.Q8_0.getBlockElements());
        assertEquals(256, GGUFType.Q4_K.getBlockElements());
    }

    @Test
    public void testGGUFTypeCalculateBlockSize() {
        assertEquals(4, GGUFType.F32.calculateBlockSize()); // 4 bytes per float
        assertEquals(2, GGUFType.F16.calculateBlockSize()); // 2 bytes per half
        assertEquals(18, GGUFType.Q4_0.calculateBlockSize()); // 32 elements * 4 bits / 8 + 2 bytes scale = 16 + 2 = 18
        assertEquals(20, GGUFType.Q4_1.calculateBlockSize()); // includes min value
        assertEquals(34, GGUFType.Q8_0.calculateBlockSize()); // 32 bytes + 2 bytes scale
    }

    // ==================== GGUFMetadataType Tests ====================

    @Test
    public void testGGUFMetadataTypeFromId() {
        assertEquals(GGUFMetadataType.UINT8, GGUFMetadataType.fromId(0));
        assertEquals(GGUFMetadataType.INT8, GGUFMetadataType.fromId(1));
        assertEquals(GGUFMetadataType.UINT16, GGUFMetadataType.fromId(2));
        assertEquals(GGUFMetadataType.INT16, GGUFMetadataType.fromId(3));
        assertEquals(GGUFMetadataType.UINT32, GGUFMetadataType.fromId(4));
        assertEquals(GGUFMetadataType.INT32, GGUFMetadataType.fromId(5));
        assertEquals(GGUFMetadataType.FLOAT32, GGUFMetadataType.fromId(6));
        assertEquals(GGUFMetadataType.BOOL, GGUFMetadataType.fromId(7));
        assertEquals(GGUFMetadataType.STRING, GGUFMetadataType.fromId(8));
        assertEquals(GGUFMetadataType.ARRAY, GGUFMetadataType.fromId(9));
        assertEquals(GGUFMetadataType.UINT64, GGUFMetadataType.fromId(10));
        assertEquals(GGUFMetadataType.INT64, GGUFMetadataType.fromId(11));
        assertEquals(GGUFMetadataType.FLOAT64, GGUFMetadataType.fromId(12));
    }

    @Test
    public void testGGUFMetadataTypeUnknown() {
        // Test that unknown IDs return UNKNOWN
        GGUFMetadataType unknown = GGUFMetadataType.fromId(999);
        assertEquals(GGUFMetadataType.UNKNOWN, unknown);
    }

    // ==================== GGUFHeader Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGGUFHeaderFromFile(Nd4jBackend backend) throws IOException {
        // Create a minimal GGUF file
        File ggufFile = tempDir.resolve("test.gguf").toFile();
        writeMinimalGGUFFile(ggufFile, "llama", 4096, 32, 32);

        GGUFHeader header = GGUFHeader.fromFile(ggufFile);

        assertNotNull(header);
        assertEquals(3, header.getVersion()); // GGUF v3
        assertEquals("llama", header.getArchitecture());
        assertEquals(4096, header.getContextLength());
        assertEquals(32, header.getEmbeddingLength());
        assertEquals(32, header.getBlockCount());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGGUFHeaderWithMetadata(Nd4jBackend backend) throws IOException {
        File ggufFile = tempDir.resolve("meta.gguf").toFile();
        writeMinimalGGUFFile(ggufFile, "mistral", 8192, 4096, 32);

        GGUFHeader header = GGUFHeader.fromFile(ggufFile);

        assertEquals("mistral", header.getArchitecture());
        assertEquals(8192, header.getContextLength());
        assertEquals(4096, header.getEmbeddingLength());
        assertEquals(32, header.getBlockCount());
    }

    @Test
    public void testGGUFHeaderInvalidMagic() throws IOException {
        File invalidFile = tempDir.resolve("invalid.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
            // Write invalid magic
            fos.write(new byte[]{0x00, 0x00, 0x00, 0x00});
        }

        assertThrows(IOException.class, () -> GGUFHeader.fromFile(invalidFile));
    }

    @Test
    public void testGGUFHeaderFileNotFound() {
        File nonExistent = new File("/non/existent/file.gguf");
        assertThrows(IOException.class, () -> GGUFHeader.fromFile(nonExistent));
    }

    // ==================== GGMLPipelineLoader Tests ====================

    @Test
    public void testGGMLPipelineLoaderSupports() {
        GGMLPipelineLoader loader = new GGMLPipelineLoader();

        assertTrue(loader.supports(ModelFormat.GGUF));
        assertFalse(loader.supports(ModelFormat.SAFETENSORS));
        assertFalse(loader.supports(ModelFormat.ONNX));
        assertFalse(loader.supports(ModelFormat.PYTORCH));
        assertEquals(ModelFormat.GGUF, loader.getFormat());
        assertEquals("GGML", loader.getName());
    }

    @Test
    public void testGGMLPipelineLoaderConvertsToSdz() {
        GGMLPipelineLoader loader = new GGMLPipelineLoader();
        assertTrue(loader.convertsToSdz());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGGMLInspectFile(Nd4jBackend backend) throws IOException {
        File ggufFile = tempDir.resolve("inspect.gguf").toFile();
        writeMinimalGGUFFile(ggufFile, "phi", 2048, 2560, 24);

        GGUFHeader header = GGMLPipelineLoader.inspectFile(ggufFile);

        assertNotNull(header);
        assertEquals("phi", header.getArchitecture());
        assertEquals(2048, header.getContextLength());
    }

    // ==================== GGUFReader Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGGUFReaderOpen(Nd4jBackend backend) throws IOException {
        File ggufFile = tempDir.resolve("reader.gguf").toFile();
        writeMinimalGGUFFile(ggufFile, "qwen", 4096, 1536, 28);

        try (GGUFReader reader = GGUFReader.open(ggufFile)) {
            assertNotNull(reader);
            assertNotNull(reader.getHeader());
            assertEquals("qwen", reader.getHeader().getArchitecture());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGGUFReaderGetMetadata(Nd4jBackend backend) throws IOException {
        File ggufFile = tempDir.resolve("metadata.gguf").toFile();
        writeMinimalGGUFFile(ggufFile, "llama", 4096, 4096, 32);

        try (GGUFReader reader = GGUFReader.open(ggufFile)) {
            Map<String, Object> metadata = reader.getMetadata();
            assertNotNull(metadata);
            // Metadata should contain architecture info
            assertTrue(metadata.containsKey("general.architecture") ||
                    reader.getHeader().getArchitecture() != null);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGGUFReaderWithTensors(Nd4jBackend backend) throws IOException {
        File ggufFile = tempDir.resolve("tensors.gguf").toFile();

        // Create a GGUF file with a simple F32 tensor
        Map<String, INDArray> tensors = new LinkedHashMap<>();
        tensors.put("test.weight", Nd4j.rand(DataType.FLOAT, 16, 32));
        writeGGUFFileWithTensors(ggufFile, "test", tensors);

        try (GGUFReader reader = GGUFReader.open(ggufFile)) {
            Set<String> tensorNames = reader.getTensorNames();
            assertNotNull(tensorNames);
            assertTrue(tensorNames.contains("test.weight"));
            assertEquals(1, reader.getTensorCount());
        }
    }

    // ==================== Helper Methods ====================

    /**
     * Write a minimal GGUF file for testing.
     */
    private void writeMinimalGGUFFile(File file, String architecture,
                                       int contextLength, int embeddingLength, int blockCount) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "rw");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer buffer = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);

            // GGUF Magic: "GGUF" in little endian
            buffer.put((byte) 'G');
            buffer.put((byte) 'G');
            buffer.put((byte) 'U');
            buffer.put((byte) 'F');

            // Version (3 for GGUF v3)
            buffer.putInt(3);

            // Tensor count (0 for this minimal file)
            buffer.putLong(0);

            // Metadata key-value count (4 entries)
            buffer.putLong(4);

            // Metadata 1: general.architecture (STRING)
            writeString(buffer, "general.architecture");
            buffer.putInt(8); // STRING type
            writeString(buffer, architecture);

            // Metadata 2: architecture.context_length (UINT32)
            writeString(buffer, architecture + ".context_length");
            buffer.putInt(4); // UINT32 type
            buffer.putInt(contextLength);

            // Metadata 3: architecture.embedding_length (UINT32)
            writeString(buffer, architecture + ".embedding_length");
            buffer.putInt(4); // UINT32 type
            buffer.putInt(embeddingLength);

            // Metadata 4: architecture.block_count (UINT32)
            writeString(buffer, architecture + ".block_count");
            buffer.putInt(4); // UINT32 type
            buffer.putInt(blockCount);

            buffer.flip();
            channel.write(buffer);
        }
    }

    /**
     * Write a GGUF file with tensors for testing.
     */
    private void writeGGUFFileWithTensors(File file, String architecture,
                                          Map<String, INDArray> tensors) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "rw");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer buffer = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN);

            // GGUF Magic
            buffer.put((byte) 'G');
            buffer.put((byte) 'G');
            buffer.put((byte) 'U');
            buffer.put((byte) 'F');

            // Version
            buffer.putInt(3);

            // Tensor count
            buffer.putLong(tensors.size());

            // Metadata key-value count
            buffer.putLong(1);

            // Metadata: general.architecture
            writeString(buffer, "general.architecture");
            buffer.putInt(8); // STRING type
            writeString(buffer, architecture);

            // Tensor infos
            long dataOffset = 0;
            for (Map.Entry<String, INDArray> entry : tensors.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();

                // Tensor name
                writeString(buffer, name);

                // Number of dimensions
                buffer.putInt(arr.rank());

                // Dimensions
                for (int i = 0; i < arr.rank(); i++) {
                    buffer.putLong(arr.size(i));
                }

                // Type (F32 = 0)
                buffer.putInt(0);

                // Offset
                buffer.putLong(dataOffset);

                dataOffset += arr.length() * 4; // F32 = 4 bytes
            }

            // Alignment padding (align to 32 bytes)
            int currentPos = buffer.position();
            int alignment = 32;
            int padding = (alignment - (currentPos % alignment)) % alignment;
            for (int i = 0; i < padding; i++) {
                buffer.put((byte) 0);
            }

            buffer.flip();
            channel.write(buffer);

            // Write tensor data
            for (INDArray arr : tensors.values()) {
                ByteBuffer dataBuffer = arr.data().asNio();
                dataBuffer.order(ByteOrder.LITTLE_ENDIAN);
                channel.write(dataBuffer);
            }
        }
    }

    private void writeString(ByteBuffer buffer, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        buffer.putLong(bytes.length);
        buffer.put(bytes);
    }
}
