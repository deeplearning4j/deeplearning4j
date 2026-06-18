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

package org.eclipse.deeplearning4j.ggml;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLFormat;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GGMLModelImport Test")
@Tag(TagNames.SMOKE)
class GGMLModelImportTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("Test isGGMLFile returns true for valid GGUF file")
    void testIsGGMLFileValidGGUF() throws IOException {
        File ggufFile = createMinimalGGUFFile();
        assertTrue(GGMLModelImport.isGGMLFile(ggufFile));
    }

    @Test
    @DisplayName("Test isGGMLFile returns true for valid legacy GGML file")
    void testIsGGMLFileValidLegacy() throws IOException {
        File ggmlFile = createLegacyGGMLFile();
        assertTrue(GGMLModelImport.isGGMLFile(ggmlFile));
    }

    @Test
    @DisplayName("Test isGGMLFile returns false for invalid file")
    void testIsGGMLFileInvalid() throws IOException {
        File invalidFile = tempDir.resolve("invalid.bin").toFile();
        try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
            fos.write(new byte[]{0x00, 0x00, 0x00, 0x00});
        }
        assertFalse(GGMLModelImport.isGGMLFile(invalidFile));
    }

    @Test
    @DisplayName("Test isGGMLFile returns false for non-existent file")
    void testIsGGMLFileNonExistent() {
        assertFalse(GGMLModelImport.isGGMLFile("/non/existent/file.gguf"));
    }

    @Test
    @DisplayName("Test detectFormat for GGUF file")
    void testDetectFormatGGUF() throws Exception {
        File ggufFile = createMinimalGGUFFile();
        GGMLFormat format = GGMLModelImport.detectFormat(ggufFile);
        assertEquals(GGMLFormat.GGUF, format);
    }

    @Test
    @DisplayName("Test detectFormat for legacy GGML file")
    void testDetectFormatLegacy() throws Exception {
        File ggmlFile = createLegacyGGMLFile();
        GGMLFormat format = GGMLModelImport.detectFormat(ggmlFile);
        assertEquals(GGMLFormat.GGML, format);
    }

    @Test
    @DisplayName("Test detectFormat throws for invalid file")
    void testDetectFormatInvalid() throws IOException {
        File invalidFile = tempDir.resolve("invalid.bin").toFile();
        try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
            fos.write(new byte[]{0x00, 0x00, 0x00, 0x00});
        }
        assertThrows(GGMLImportException.class, () -> GGMLModelImport.detectFormat(invalidFile));
    }

    @Test
    @DisplayName("Test importModel throws for non-existent file")
    void testImportModelNonExistent() {
        assertThrows(GGMLImportException.class,
                () -> GGMLModelImport.importModel("/non/existent/file.gguf"));
    }

    @Test
    @DisplayName("Test importModel with file path string")
    void testImportModelStringPath() throws IOException, GGMLImportException {
        File ggufFile = createMinimalGGUFFile();
        // A minimal GGUF file with no tensors should still import successfully
        // using the GenericArchitecture fallback (returns empty graph)
        var sd = GGMLModelImport.importModel(ggufFile.getAbsolutePath());
        assertNotNull(sd);
    }

    @Test
    @DisplayName("Test ConversionOptions builder defaults")
    void testConversionOptionsDefaults() {
        ConversionOptions options = ConversionOptions.builder().build();

        assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16,
                options.getQuantizationMode());
        assertEquals(DataType.FLOAT16, options.getTargetDataType());
        assertTrue(options.isPreserveTokenizerInfo());
        assertFalse(options.isForTraining());
        assertTrue(options.isUseMemoryMapping());
        assertEquals(10, options.getTensorBatchSize());
        assertNull(options.getArchitectureOverride());
        assertEquals(0, options.getMaxFileSize());
    }

    @Test
    @DisplayName("Test ConversionOptions forInference")
    void testConversionOptionsForInference() {
        ConversionOptions options = ConversionOptions.forInference();

        assertFalse(options.isForTraining());
        // forInference() uses FLOAT32 to avoid cumulative precision loss from
        // FP16 cast ops over many transformer layers (intentional design choice)
        assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32,
                options.getQuantizationMode());
    }

    @Test
    @DisplayName("Test ConversionOptions forTraining")
    void testConversionOptionsForTraining() {
        ConversionOptions options = ConversionOptions.forTraining();

        assertTrue(options.isForTraining());
        assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32,
                options.getQuantizationMode());
    }

    @Test
    @DisplayName("Test ConversionOptions fp16")
    void testConversionOptionsFp16() {
        ConversionOptions options = ConversionOptions.fp16();

        assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16,
                options.getQuantizationMode());
        assertEquals(DataType.HALF, options.getTargetDataType());
    }

    @Test
    @DisplayName("Test ConversionOptions preserveQuantization")
    void testConversionOptionsPreserveQuantization() {
        ConversionOptions options = ConversionOptions.preserveQuantization();

        assertEquals(ConversionOptions.QuantizationMode.PRESERVE_QUANTIZATION,
                options.getQuantizationMode());
    }

    @Test
    @DisplayName("Test ConversionOptions custom settings")
    void testConversionOptionsCustom() {
        ConversionOptions options = ConversionOptions.builder()
                .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_BFLOAT16)
                .targetDataType(DataType.BFLOAT16)
                .forTraining(true)
                .preserveTokenizerInfo(false)
                .useMemoryMapping(false)
                .tensorBatchSize(20)
                .architectureOverride("llama")
                .maxFileSize(1024 * 1024 * 1024L)
                .build();

        assertEquals(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_BFLOAT16,
                options.getQuantizationMode());
        assertEquals(DataType.BFLOAT16, options.getTargetDataType());
        assertTrue(options.isForTraining());
        assertFalse(options.isPreserveTokenizerInfo());
        assertFalse(options.isUseMemoryMapping());
        assertEquals(20, options.getTensorBatchSize());
        assertEquals("llama", options.getArchitectureOverride());
        assertEquals(1024 * 1024 * 1024L, options.getMaxFileSize());
    }

    @Test
    @DisplayName("Test GGMLImportException with message")
    void testGGMLImportExceptionMessage() {
        GGMLImportException ex = new GGMLImportException("Test error");
        assertEquals("Test error", ex.getMessage());
    }

    @Test
    @DisplayName("Test GGMLImportException with cause")
    void testGGMLImportExceptionWithCause() {
        IOException cause = new IOException("IO error");
        GGMLImportException ex = new GGMLImportException("Test error", cause);
        assertEquals("Test error", ex.getMessage());
        assertEquals(cause, ex.getCause());
    }

    @Test
    @DisplayName("Test maxFileSize validation")
    void testMaxFileSizeValidation() throws IOException {
        File ggufFile = createMinimalGGUFFile();

        ConversionOptions options = ConversionOptions.builder()
                .maxFileSize(1) // 1 byte max
                .build();

        assertThrows(GGMLImportException.class,
                () -> GGMLModelImport.importModel(ggufFile, options));
    }

    @Test
    @DisplayName("HALF tensors load with exact GGUF shape (rank-1 [1] stays rank-1)")
    void testHalfTensorShapePreservation() throws IOException, GGMLImportException {
        // Regression test for SameDiffSerializer.saveAutoShard crash:
        //   "createOpaqueNDArray: Shape info was empty but buffer was not null!"
        // Root cause: the old HALF path used Nd4j.create(buf) which defaults to
        // shape {1, buf.length()}, then conditionally reshaped only when shape.length > 1.
        // That left rank-1 [1] GGUF tensors as {1, 1} 2D arrays; downstream reshape
        // to rank-0 then produced a malformed array with isEmpty=true and length=1.
        // The fix constructs each array with the exact GGUF shape up front.
        File ggufFile = createGGUFFileWithHalfTensors();

        ConversionOptions options = ConversionOptions.builder()
                .architectureOverride("generic")
                .build();

        SameDiff sd = GGMLModelImport.importModel(ggufFile, options);
        assertNotNull(sd);

        // Shape [1] HALF — the exact case that previously produced [1, 1] 2D.
        assertHalfConstantShape(sd, "scalar_weight", new long[]{1});
        // Shape [4] HALF — rank-1 multi-element.
        assertHalfConstantShape(sd, "vec_weight", new long[]{4});
        // Shape [3, 3] HALF — rank-2 (symmetric under GGUF column-major reversal).
        assertHalfConstantShape(sd, "mat_weight", new long[]{3, 3});
    }

    private void assertHalfConstantShape(SameDiff sd, String varName, long[] expectedShape) {
        SDVariable var = sd.getVariable(varName);
        assertNotNull(var, "Variable not found: " + varName);
        INDArray arr = var.getArr();
        assertNotNull(arr, "Array not materialized for: " + varName);
        assertEquals(DataType.HALF, arr.dataType(), "Wrong dtype for " + varName);
        assertArrayEquals(expectedShape, arr.shape(), "Wrong shape for " + varName);
        assertEquals(expectedShape.length, arr.rank(), "Wrong rank for " + varName);
        assertFalse(arr.isEmpty(), "Array should not be empty: " + varName);

        // Regression: this dup() is the exact call that crashed in
        // SameDiffSerializer.saveSharded → associateArrayWithVariable(arr.dup(), ...)
        INDArray duped = arr.dup();
        assertNotNull(duped);
        assertArrayEquals(expectedShape, duped.shape(), "dup() changed shape for " + varName);
        assertFalse(duped.isEmpty(), "duped array should not be empty: " + varName);
    }

    private File createGGUFFileWithHalfTensors() throws IOException {
        File file = tempDir.resolve("half_shapes.gguf").toFile();
        try (GGUFWriter writer = new GGUFWriter(file, 3)) {
            writer.addMetadataString("general.architecture", "generic");

            writer.registerTensor("scalar_weight", new long[]{1}, GGMLDataType.GGML_TYPE_F16);
            writer.registerTensor("vec_weight", new long[]{4}, GGMLDataType.GGML_TYPE_F16);
            writer.registerTensor("mat_weight", new long[]{3, 3}, GGMLDataType.GGML_TYPE_F16);

            writer.writeHeader();

            // F16 = 2 bytes per element. Use zeroed bytes — values don't matter for
            // the shape/dup regression, only the storage size.
            writer.writeTensorData("scalar_weight", new byte[1 * 2]);
            writer.writeTensorData("vec_weight", new byte[4 * 2]);
            writer.writeTensorData("mat_weight", new byte[3 * 3 * 2]);

            writer.finalizeFile();
        }
        return file;
    }

    private File createMinimalGGUFFile() throws IOException {
        File file = tempDir.resolve("minimal.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(0x46554747); // GGUF magic
            buffer.putInt(3);          // version
            buffer.putLong(0);         // tensor count
            buffer.putLong(0);         // metadata KV count
            fos.write(buffer.array());
        }
        return file;
    }

    private File createLegacyGGMLFile() throws IOException {
        File file = tempDir.resolve("legacy.ggml").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(8);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(0x67676D6C); // 'ggml' magic
            buffer.putInt(1);          // version
            fos.write(buffer.array());
        }
        return file;
    }
}
