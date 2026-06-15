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
import org.nd4j.ggml.format.GGUFHeader;
import org.nd4j.ggml.format.GGUFReader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GGUFReader Test")
class GGUFReaderTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("Test read minimal GGUF header")
    void testReadMinimalHeader() throws IOException, GGMLImportException {
        File ggufFile = createMinimalGGUFFile();

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();

            assertNotNull(metadata);
            assertEquals(0, metadata.getTensors().size());
        }
    }

    @Test
    @DisplayName("Test read GGUF with string metadata")
    void testReadStringMetadata() throws IOException, GGMLImportException {
        File ggufFile = createGGUFWithMetadata();

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            GGMLMetadata metadata = reader.getMetadata();

            assertNotNull(metadata);
            // The architecture should be set from the metadata
            assertNotNull(metadata.getArchitecture());
        }
    }

    @Test
    @DisplayName("Test GGUF header validation")
    void testHeaderValidation() throws IOException {
        File invalidFile = tempDir.resolve("invalid.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
            ByteBuffer buffer = ByteBuffer.allocate(24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            // Wrong magic
            buffer.putInt(0x12345678);
            buffer.putInt(3);
            buffer.putLong(0);
            buffer.putLong(0);
            fos.write(buffer.array());
        }

        // Header validation occurs when reading metadata, not on construction
        assertThrows(GGMLImportException.class, () -> {
            try (GGUFReader reader = new GGUFReader(invalidFile)) {
                reader.getMetadata();
            }
        });
    }

    @Test
    @DisplayName("Test GGUF version check")
    void testVersionCheck() throws IOException, GGMLImportException {
        File versionFile = createGGUFWithVersion(3);

        try (GGUFReader reader = new GGUFReader(versionFile)) {
            GGMLMetadata metadata = reader.getMetadata();
            assertNotNull(metadata);
        }
    }

    @Test
    @DisplayName("Test GGUFHeader properties")
    void testGGUFHeaderProperties() {
        GGUFHeader header = GGUFHeader.builder()
                .magic(0x46554747)
                .version(3)
                .tensorCount(10)
                .metadataKVCount(5)
                .build();

        assertEquals(0x46554747, header.getMagic());
        assertEquals(3, header.getVersion());
        assertEquals(10, header.getTensorCount());
        assertEquals(5, header.getMetadataKVCount());
    }

    @Test
    @DisplayName("Test GGMLMetadata builder")
    void testGGMLMetadataBuilder() {
        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("llama")
                .modelName("test-model")
                .vocabSize(32000)
                .hiddenSize(4096)
                .numLayers(32)
                .numAttentionHeads(32)
                .numKVHeads(8)
                .contextLength(4096)
                .build();

        assertEquals("llama", metadata.getArchitecture());
        assertEquals("test-model", metadata.getModelName());
        assertEquals(32000, metadata.getVocabSize());
        assertEquals(4096, metadata.getHiddenSize());
        assertEquals(32, metadata.getNumLayers());
        assertEquals(32, metadata.getNumAttentionHeads());
        assertEquals(8, metadata.getNumKVHeads());
        assertEquals(4096, metadata.getContextLength());
    }

    @Test
    @DisplayName("Test GGMLTensorInfo")
    void testGGMLTensorInfo() {
        long[] shape = new long[]{4096, 4096};
        GGMLTensorInfo tensorInfo = GGMLTensorInfo.builder()
                .name("model.layers.0.attention.wq.weight")
                .dataType(GGMLDataType.GGML_TYPE_F16)
                .shape(shape)
                .dataOffset(1024)
                .build();

        assertEquals("model.layers.0.attention.wq.weight", tensorInfo.getName());
        assertEquals(GGMLDataType.GGML_TYPE_F16, tensorInfo.getDataType());
        assertArrayEquals(shape, tensorInfo.getShape());
        assertEquals(1024, tensorInfo.getDataOffset());
        assertEquals(4096 * 4096, tensorInfo.getNumElements());
    }

    @Test
    @DisplayName("Test GGMLTensorInfo shape string")
    void testGGMLTensorInfoShapeString() {
        GGMLTensorInfo tensorInfo = GGMLTensorInfo.builder()
                .name("test")
                .dataType(GGMLDataType.GGML_TYPE_F32)
                .shape(new long[]{10, 20, 30})
                .dataOffset(0)
                .build();

        String shapeString = tensorInfo.getShapeString();
        assertTrue(shapeString.contains("10"));
        assertTrue(shapeString.contains("20"));
        assertTrue(shapeString.contains("30"));
    }

    @Test
    @DisplayName("Test total parameters calculation")
    void testTotalParametersCalculation() {
        GGMLTensorInfo tensor1 = GGMLTensorInfo.builder()
                .name("tensor1")
                .dataType(GGMLDataType.GGML_TYPE_F32)
                .shape(new long[]{100, 100})
                .dataOffset(0)
                .build();

        GGMLTensorInfo tensor2 = GGMLTensorInfo.builder()
                .name("tensor2")
                .dataType(GGMLDataType.GGML_TYPE_F32)
                .shape(new long[]{50, 50})
                .dataOffset(40000)
                .build();

        GGMLMetadata metadata = GGMLMetadata.builder()
                .architecture("test")
                .build();
        metadata.getTensors().add(tensor1);
        metadata.getTensors().add(tensor2);

        assertEquals(12500, metadata.getTotalParameters()); // 10000 + 2500
    }

    private File createMinimalGGUFFile() throws IOException {
        File file = tempDir.resolve("minimal.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            // GGUF magic
            buffer.putInt(0x46554747);
            // Version
            buffer.putInt(3);
            // Tensor count
            buffer.putLong(0);
            // Metadata KV count
            buffer.putLong(0);
            fos.write(buffer.array());
        }
        return file;
    }

    private File createGGUFWithVersion(int version) throws IOException {
        File file = tempDir.resolve("versioned.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(0x46554747);
            buffer.putInt(version);
            buffer.putLong(0);
            buffer.putLong(0);
            fos.write(buffer.array());
        }
        return file;
    }

    private File createGGUFWithMetadata() throws IOException {
        File file = tempDir.resolve("metadata.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            // Calculate sizes
            String keyName = "general.architecture";
            String keyValue = "llama";
            byte[] keyBytes = keyName.getBytes(StandardCharsets.UTF_8);
            byte[] valueBytes = keyValue.getBytes(StandardCharsets.UTF_8);

            // Header (24 bytes) + key length (8) + key + value type (4) + value length (8) + value
            int totalSize = 24 + 8 + keyBytes.length + 4 + 8 + valueBytes.length;
            ByteBuffer buffer = ByteBuffer.allocate(totalSize);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // Header
            buffer.putInt(0x46554747); // magic
            buffer.putInt(3);          // version
            buffer.putLong(0);         // tensor count
            buffer.putLong(1);         // metadata KV count

            // Key-value pair
            buffer.putLong(keyBytes.length);   // key length
            buffer.put(keyBytes);              // key
            buffer.putInt(8);                  // value type (8 = string)
            buffer.putLong(valueBytes.length); // value length
            buffer.put(valueBytes);            // value

            fos.write(buffer.array());
        }
        return file;
    }
}
