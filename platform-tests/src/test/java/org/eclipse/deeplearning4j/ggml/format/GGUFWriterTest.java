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

package org.eclipse.deeplearning4j.ggml.format;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.format.GGUFWriter;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GGUFWriter Test")
class GGUFWriterTest {

    @TempDir
    Path tempDir;

    // ========== Constructor Tests ==========

    @Test
    @DisplayName("Test create GGUFWriter with version 3")
    void testCreateWriterVersion3() throws IOException {
        File outputFile = tempDir.resolve("test_v3.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            assertNotNull(writer);
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test create GGUFWriter with version 2")
    void testCreateWriterVersion2() throws IOException {
        File outputFile = tempDir.resolve("test_v2.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 2)) {
            assertNotNull(writer);
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test create GGUFWriter with version 1")
    void testCreateWriterVersion1() throws IOException {
        File outputFile = tempDir.resolve("test_v1.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 1)) {
            assertNotNull(writer);
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test create GGUFWriter with invalid version throws")
    void testCreateWriterInvalidVersion() {
        File outputFile = tempDir.resolve("test_invalid.gguf").toFile();
        assertThrows(IllegalArgumentException.class, () -> {
            new GGUFWriter(outputFile, 0);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new GGUFWriter(outputFile, 4);
        });
    }

    @Test
    @DisplayName("Test zero alignment metadata uses the GGUF default")
    void testZeroAlignmentMetadataUsesDefault() throws Exception {
        File outputFile = tempDir.resolve("zero_alignment.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            assertThrows(IllegalArgumentException.class, () -> writer.setAlignment(0));
            writer.addMetadataInt("general.alignment", 0);
            writer.registerTensor("tensor", new long[]{4}, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();
            writer.writeTensorData("tensor", new byte[4 * Float.BYTES]);
            writer.finalizeFile();
        }

        try (GGUFReader reader = new GGUFReader(outputFile)) {
            assertEquals(32, reader.getHeader().getAlignment());
            assertEquals(1, reader.getTensorInfos().size());
            assertEquals(4 * Float.BYTES,
                    reader.readTensorData(reader.getTensorInfos().get(0)).length);
        }
    }

    // ========== Metadata Tests ==========

    @Test
    @DisplayName("Test add string metadata")
    void testAddStringMetadata() throws IOException {
        File outputFile = tempDir.resolve("string_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.addMetadataString("general.name", "test-model");
            writer.writeHeader();
            writer.finalizeFile();
        }

        // Verify we can read it back
        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test add int metadata")
    void testAddIntMetadata() throws IOException {
        File outputFile = tempDir.resolve("int_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadataInt("llama.block_count", 32);
            writer.addMetadataInt("llama.embedding_length", 4096);
            writer.writeHeader();
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test add long metadata")
    void testAddLongMetadata() throws IOException {
        File outputFile = tempDir.resolve("long_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadataLong("general.parameter_count", 7000000000L);
            writer.writeHeader();
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test add float metadata")
    void testAddFloatMetadata() throws IOException {
        File outputFile = tempDir.resolve("float_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadataFloat("llama.attention.layer_norm_rms_epsilon", 1e-5f);
            writer.addMetadataFloat("llama.rope.freq_base", 10000.0f);
            writer.writeHeader();
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test add boolean metadata")
    void testAddBoolMetadata() throws IOException {
        File outputFile = tempDir.resolve("bool_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadataBool("general.use_cache", true);
            writer.addMetadataBool("general.quantized", false);
            writer.writeHeader();
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test add array metadata")
    void testAddArrayMetadata() throws IOException {
        File outputFile = tempDir.resolve("array_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("tensor_sizes", new int[]{100, 200, 300});
            writer.addMetadata("hidden_dims", new long[]{1024L, 2048L, 4096L});
            writer.addMetadata("scales", new float[]{1.0f, 0.5f, 0.25f});
            writer.writeHeader();
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test add list metadata")
    void testAddListMetadata() throws IOException {
        File outputFile = tempDir.resolve("list_meta.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("string_list", Arrays.asList("token1", "token2", "token3"));
            writer.addMetadata("int_list", Arrays.asList(1, 2, 3));
            writer.writeHeader();
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test metadata cannot be added after header written")
    void testMetadataAfterHeader() throws IOException {
        File outputFile = tempDir.resolve("meta_after_header.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("key1", "value1");
            writer.writeHeader();

            assertThrows(IllegalStateException.class, () -> {
                writer.addMetadata("key2", "value2");
            });
        }
    }

    // ========== Tensor Registration Tests ==========

    @Test
    @DisplayName("Test register tensor")
    void testRegisterTensor() throws IOException {
        File outputFile = tempDir.resolve("register_tensor.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("token_embd.weight", new long[]{4096, 32000}, GGMLDataType.GGML_TYPE_F16);
            writer.registerTensor("output.weight", new long[]{4096, 32000}, GGMLDataType.GGML_TYPE_F16);
            writer.writeHeader();

            // Write tensor data
            byte[] tensorData = new byte[4096 * 32000 * 2]; // F16 = 2 bytes per element
            writer.writeTensorData("token_embd.weight", tensorData);
            writer.writeTensorData("output.weight", tensorData);

            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    @Test
    @DisplayName("Test register tensor cannot happen after header written")
    void testRegisterTensorAfterHeader() throws IOException {
        File outputFile = tempDir.resolve("register_after_header.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.writeHeader();

            assertThrows(IllegalStateException.class, () -> {
                writer.registerTensor("tensor", new long[]{100}, GGMLDataType.GGML_TYPE_F32);
            });
        }
    }

    @Test
    @DisplayName("Test write tensor data requires header first")
    void testWriteTensorDataRequiresHeader() throws IOException {
        File outputFile = tempDir.resolve("data_before_header.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("tensor", new long[]{100}, GGMLDataType.GGML_TYPE_F32);

            assertThrows(IllegalStateException.class, () -> {
                writer.writeTensorData("tensor", new byte[400]);
            });
        }
    }

    @Test
    @DisplayName("Test write unregistered tensor throws")
    void testWriteUnregisteredTensor() throws IOException {
        File outputFile = tempDir.resolve("unregistered.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("tensor1", new long[]{100}, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();

            assertThrows(IllegalArgumentException.class, () -> {
                writer.writeTensorData("nonexistent", new byte[400]);
            });
        }
    }

    @Test
    @DisplayName("Test finalize requires all tensors written")
    void testFinalizeRequiresAllTensors() throws IOException {
        File outputFile = tempDir.resolve("incomplete.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("tensor1", new long[]{100}, GGMLDataType.GGML_TYPE_F32);
            writer.registerTensor("tensor2", new long[]{100}, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();

            writer.writeTensorData("tensor1", new byte[400]);
            // tensor2 not written

            assertThrows(IllegalStateException.class, writer::finalizeFile);
        }
    }

    // ========== Alignment Tests ==========

    @Test
    @DisplayName("Test set alignment")
    void testSetAlignment() throws IOException {
        File outputFile = tempDir.resolve("alignment.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.setAlignment(64);
            writer.registerTensor("tensor", new long[]{100}, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();
            writer.writeTensorData("tensor", new byte[400]);
            writer.finalizeFile();
        }

        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test alignment cannot be set after header written")
    void testAlignmentAfterHeader() throws IOException {
        File outputFile = tempDir.resolve("alignment_after.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.writeHeader();

            assertThrows(IllegalStateException.class, () -> {
                writer.setAlignment(64);
            });
        }
    }

    // ========== Write Header Tests ==========

    @Test
    @DisplayName("Test write header multiple times throws")
    void testWriteHeaderTwice() throws IOException {
        File outputFile = tempDir.resolve("header_twice.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.writeHeader();

            assertThrows(IllegalStateException.class, writer::writeHeader);
        }
    }

    @Test
    @DisplayName("Test finalize requires header")
    void testFinalizeRequiresHeader() throws IOException {
        File outputFile = tempDir.resolve("no_header.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            assertThrows(IllegalStateException.class, writer::finalizeFile);
        }
    }

    // ========== Round-trip Tests ==========

    @Test
    @DisplayName("Test write and read minimal GGUF")
    void testWriteReadMinimal() throws IOException, GGMLImportException {
        File outputFile = tempDir.resolve("roundtrip_minimal.gguf").toFile();

        // Write
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.writeHeader();
            writer.finalizeFile();
        }

        // Read
        try (GGUFReader reader = new GGUFReader(outputFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertEquals("llama", metadata.getArchitecture());
        }
    }

    @Test
    @DisplayName("Test write and read with tensors")
    void testWriteReadWithTensors() throws IOException, GGMLImportException {
        File outputFile = tempDir.resolve("roundtrip_tensors.gguf").toFile();

        long[] shape = new long[]{256, 128};
        int numElements = 256 * 128;
        int bytesPerElement = 4; // F32

        // Create test tensor data
        byte[] tensorData = new byte[numElements * bytesPerElement];
        ByteBuffer buffer = ByteBuffer.wrap(tensorData).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numElements; i++) {
            buffer.putFloat(i * 0.001f);
        }

        // Write
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.addMetadataInt("llama.block_count", 1);
            writer.registerTensor("test_tensor", shape, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();
            writer.writeTensorData("test_tensor", tensorData);
            writer.finalizeFile();
        }

        // Read
        try (GGUFReader reader = new GGUFReader(outputFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertEquals("llama", metadata.getArchitecture());
            assertEquals(1, metadata.getTensors().size());

            GGMLTensorInfo tensorInfo = metadata.getTensors().get(0);
            assertEquals("test_tensor", tensorInfo.getName());
            assertArrayEquals(shape, tensorInfo.getShape());
            assertEquals(GGMLDataType.GGML_TYPE_F32, tensorInfo.getDataType());
        }
    }

    @Test
    @DisplayName("Test write and read multiple tensors")
    void testWriteReadMultipleTensors() throws IOException, GGMLImportException {
        File outputFile = tempDir.resolve("roundtrip_multi.gguf").toFile();

        // Write
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("general.architecture", "llama");

            writer.registerTensor("tensor1", new long[]{100, 50}, GGMLDataType.GGML_TYPE_F32);
            writer.registerTensor("tensor2", new long[]{50, 100}, GGMLDataType.GGML_TYPE_F16);
            writer.registerTensor("tensor3", new long[]{25, 25}, GGMLDataType.GGML_TYPE_F32);

            writer.writeHeader();

            writer.writeTensorData("tensor1", new byte[100 * 50 * 4]);
            writer.writeTensorData("tensor2", new byte[50 * 100 * 2]);
            writer.writeTensorData("tensor3", new byte[25 * 25 * 4]);

            writer.finalizeFile();
        }

        // Read
        try (GGUFReader reader = new GGUFReader(outputFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertEquals(3, metadata.getTensors().size());

            // Check tensors
            GGMLTensorInfo t1 = metadata.getTensors().stream()
                    .filter(t -> t.getName().equals("tensor1"))
                    .findFirst().orElseThrow();
            assertEquals(GGMLDataType.GGML_TYPE_F32, t1.getDataType());
            assertArrayEquals(new long[]{100, 50}, t1.getShape());

            GGMLTensorInfo t2 = metadata.getTensors().stream()
                    .filter(t -> t.getName().equals("tensor2"))
                    .findFirst().orElseThrow();
            assertEquals(GGMLDataType.GGML_TYPE_F16, t2.getDataType());
            assertArrayEquals(new long[]{50, 100}, t2.getShape());
        }
    }

    @Test
    @DisplayName("Test write and read all metadata types")
    void testWriteReadAllMetadataTypes() throws IOException, GGMLImportException {
        File outputFile = tempDir.resolve("roundtrip_all_meta.gguf").toFile();

        // Write
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("general.architecture", "llama");
            writer.addMetadata("string_key", "string_value");
            writer.addMetadata("int_key", 42);
            writer.addMetadata("long_key", 123456789L);
            writer.addMetadata("float_key", 3.14159f);
            writer.addMetadata("bool_key", true);
            writer.writeHeader();
            writer.finalizeFile();
        }

        // Read and verify file is valid
        try (GGUFReader reader = new GGUFReader(outputFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
            assertEquals("llama", metadata.getArchitecture());
        }
    }

    // ========== Data Type Tests ==========

    @Test
    @DisplayName("Test write F32 tensor")
    void testWriteF32Tensor() throws IOException {
        File outputFile = tempDir.resolve("f32_tensor.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("f32_tensor", new long[]{100}, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();
            writer.writeTensorData("f32_tensor", new byte[400]); // 100 * 4 bytes
            writer.finalizeFile();
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test write F16 tensor")
    void testWriteF16Tensor() throws IOException {
        File outputFile = tempDir.resolve("f16_tensor.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("f16_tensor", new long[]{100}, GGMLDataType.GGML_TYPE_F16);
            writer.writeHeader();
            writer.writeTensorData("f16_tensor", new byte[200]); // 100 * 2 bytes
            writer.finalizeFile();
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test write Q4_0 tensor")
    void testWriteQ4_0Tensor() throws IOException {
        File outputFile = tempDir.resolve("q4_0_tensor.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            // Q4_0: 32 elements per block, 18 bytes per block
            // 128 elements = 4 blocks = 72 bytes
            writer.registerTensor("q4_0_tensor", new long[]{128}, GGMLDataType.GGML_TYPE_Q4_0);
            writer.writeHeader();
            writer.writeTensorData("q4_0_tensor", new byte[72]);
            writer.finalizeFile();
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test write Q8_0 tensor")
    void testWriteQ8_0Tensor() throws IOException {
        File outputFile = tempDir.resolve("q8_0_tensor.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            // Q8_0: 32 elements per block, 34 bytes per block
            // 64 elements = 2 blocks = 68 bytes
            writer.registerTensor("q8_0_tensor", new long[]{64}, GGMLDataType.GGML_TYPE_Q8_0);
            writer.writeHeader();
            writer.writeTensorData("q8_0_tensor", new byte[68]);
            writer.finalizeFile();
        }
        assertTrue(outputFile.exists());
    }

    @Test
    @DisplayName("Test streaming tensor chunks")
    void testStreamingTensorChunks() throws Exception {
        File outputFile = tempDir.resolve("streaming_tensor.gguf").toFile();
        byte[] expected = new byte[16];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = (byte) (i * 7);
        }

        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.registerTensor("streamed", new long[]{4}, GGMLDataType.GGML_TYPE_F32);
            writer.writeHeader();
            writer.beginTensorData("streamed");
            writer.writeTensorDataChunk(expected, 0, 3);
            writer.writeTensorDataChunk(expected, 3, expected.length - 3);
            writer.endTensorData();
            writer.finalizeFile();
        }

        try (GGUFReader reader = new GGUFReader(outputFile)) {
            GGMLTensorInfo tensor = reader.getTensorInfos().get(0);
            assertArrayEquals(expected, reader.readTensorData(tensor));
        }
    }

    @Test
    @DisplayName("Test metadata header grows beyond initial buffer")
    void testLargeMetadataHeader() throws Exception {
        File outputFile = tempDir.resolve("large_metadata.gguf").toFile();
        char[] characters = new char[2 * 1024 * 1024];
        Arrays.fill(characters, 'x');
        String largeValue = new String(characters);

        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.addMetadata("test.large_metadata", largeValue);
            writer.writeHeader();
            writer.finalizeFile();
        }

        try (GGUFReader reader = new GGUFReader(outputFile)) {
            assertEquals(largeValue, reader.getHeader().getMetadata().get("test.large_metadata"));
        }
    }

    // ========== File Structure Tests ==========

    @Test
    @DisplayName("Test file starts with GGUF magic")
    void testFileMagic() throws IOException {
        File outputFile = tempDir.resolve("magic_check.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
            writer.writeHeader();
            writer.finalizeFile();
        }

        // Read first 4 bytes
        byte[] magic = new byte[4];
        try (java.io.FileInputStream fis = new java.io.FileInputStream(outputFile)) {
            fis.read(magic);
        }

        ByteBuffer buffer = ByteBuffer.wrap(magic).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0x46554747, buffer.getInt()); // "GGUF"
    }

    @Test
    @DisplayName("Test file has correct version")
    void testFileVersion() throws IOException {
        File outputFile = tempDir.resolve("version_check.gguf").toFile();
        int expectedVersion = 3;

        try (GGUFWriter writer = new GGUFWriter(outputFile, expectedVersion)) {
            writer.writeHeader();
            writer.finalizeFile();
        }

        // Read magic (4 bytes) + version (4 bytes)
        byte[] header = new byte[8];
        try (java.io.FileInputStream fis = new java.io.FileInputStream(outputFile)) {
            fis.read(header);
        }

        ByteBuffer buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
        buffer.getInt(); // Skip magic
        assertEquals(expectedVersion, buffer.getInt());
    }
}
