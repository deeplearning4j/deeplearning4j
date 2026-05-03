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

package org.nd4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.ggml.format.*;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Main entry point for GGML/GGUF model import.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Import a GGUF model
 * SameDiff model = GGMLModelImport.importModel("llama-7b.gguf");
 *
 * // Import with options
 * ConversionOptions options = ConversionOptions.builder()
 *     .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
 *     .forTraining(true)
 *     .build();
 * SameDiff model = GGMLModelImport.importModel("llama-7b.gguf", options);
 *
 * // Convert directly to SDZ format
 * GGMLModelImport.convertToSDZ("llama-7b.gguf", "llama-7b.sdz");
 *
 * // Inspect model metadata without full import
 * GGMLMetadata metadata = GGMLModelImport.inspectModel("llama-7b.gguf");
 * System.out.println("Architecture: " + metadata.getArchitecture());
 * System.out.println("Parameters: " + metadata.getTotalParameters());
 * }</pre>
 */
@Slf4j
public class GGMLModelImport {

    private GGMLModelImport() {
        // Utility class
    }

    // ========== Import Methods ==========

    /**
     * Import a GGML/GGUF model file to SameDiff
     *
     * @param filePath path to the GGML/GGUF file
     * @return SameDiff graph containing the model
     * @throws GGMLImportException if import fails
     */
    public static SameDiff importModel(String filePath) throws GGMLImportException {
        return importModel(new File(filePath));
    }

    /**
     * Import a GGML/GGUF model file to SameDiff with options
     *
     * @param filePath path to the GGML/GGUF file
     * @param options  conversion options
     * @return SameDiff graph containing the model
     * @throws GGMLImportException if import fails
     */
    public static SameDiff importModel(String filePath, ConversionOptions options) throws GGMLImportException {
        return importModel(new File(filePath), options);
    }

    /**
     * Import a GGML/GGUF model file to SameDiff
     *
     * @param file the GGML/GGUF file
     * @return SameDiff graph containing the model
     * @throws GGMLImportException if import fails
     */
    public static SameDiff importModel(File file) throws GGMLImportException {
        return importModel(file, ConversionOptions.forInference());
    }

    /**
     * Import a GGML/GGUF model file to SameDiff with options
     *
     * @param file    the GGML/GGUF file
     * @param options conversion options
     * @return SameDiff graph containing the model
     * @throws GGMLImportException if import fails
     */
    public static SameDiff importModel(File file, ConversionOptions options) throws GGMLImportException {
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(options);
        return converter.convert(file);
    }

    /**
     * Import a GGML/GGUF model from an input stream
     *
     * @param inputStream the input stream
     * @return SameDiff graph containing the model
     * @throws GGMLImportException if import fails
     */
    public static SameDiff importModel(InputStream inputStream) throws GGMLImportException {
        return importModel(inputStream, ConversionOptions.forInference());
    }

    /**
     * Import a GGML/GGUF model from an input stream with options
     *
     * @param inputStream the input stream
     * @param options     conversion options
     * @return SameDiff graph containing the model
     * @throws GGMLImportException if import fails
     */
    public static SameDiff importModel(InputStream inputStream, ConversionOptions options)
            throws GGMLImportException {
        try {
            // Write stream to temp file for memory-mapped access
            Path tempFile = Files.createTempFile("ggml_import_", ".gguf");
            try {
                Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
                return importModel(tempFile.toFile(), options);
            } finally {
                Files.deleteIfExists(tempFile);
            }
        } catch (IOException e) {
            throw new GGMLImportException("Failed to read from input stream", e);
        }
    }

    // ========== Conversion Methods ==========

    /**
     * Convert a GGML/GGUF file directly to SDZ format
     *
     * @param ggmlPath path to the GGML/GGUF file
     * @param sdzPath  path for the output SDZ file
     * @throws GGMLImportException if conversion fails
     */
    public static void convertToSDZ(String ggmlPath, String sdzPath) throws GGMLImportException {
        convertToSDZ(new File(ggmlPath), new File(sdzPath));
    }

    /**
     * Convert a GGML/GGUF file directly to SDZ format with options
     *
     * @param ggmlPath path to the GGML/GGUF file
     * @param sdzPath  path for the output SDZ file
     * @param options  conversion options
     * @throws GGMLImportException if conversion fails
     */
    public static void convertToSDZ(String ggmlPath, String sdzPath, ConversionOptions options)
            throws GGMLImportException {
        convertToSDZ(new File(ggmlPath), new File(sdzPath), options);
    }

    /**
     * Convert a GGML/GGUF file directly to SDZ format
     *
     * @param ggmlFile the GGML/GGUF file
     * @param sdzFile  the output SDZ file
     * @throws GGMLImportException if conversion fails
     */
    public static void convertToSDZ(File ggmlFile, File sdzFile) throws GGMLImportException {
        convertToSDZ(ggmlFile, sdzFile, ConversionOptions.builder().build());
    }

    /**
     * Convert a GGML/GGUF file directly to SDZ format with options
     *
     * @param ggmlFile the GGML/GGUF file
     * @param sdzFile  the output SDZ file
     * @param options  conversion options
     * @throws GGMLImportException if conversion fails
     */
    public static void convertToSDZ(File ggmlFile, File sdzFile, ConversionOptions options)
            throws GGMLImportException {
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(options);
        converter.convertToSDZ(ggmlFile, sdzFile);
    }

    // ========== Inspection Methods ==========

    /**
     * Inspect model metadata without performing full import
     *
     * @param filePath path to the GGML/GGUF file
     * @return model metadata
     * @throws GGMLImportException if inspection fails
     */
    public static GGMLMetadata inspectModel(String filePath) throws GGMLImportException {
        return inspectModel(new File(filePath));
    }

    /**
     * Inspect model metadata without performing full import
     *
     * @param file the GGML/GGUF file
     * @return model metadata
     * @throws GGMLImportException if inspection fails
     */
    public static GGMLMetadata inspectModel(File file) throws GGMLImportException {
        try {
            GGMLFormat format = GGMLFormatDetector.detect(file);

            if (format == GGMLFormat.GGUF) {
                try (GGUFReader reader = new GGUFReader(file)) {
                    return reader.getMetadata();
                }
            } else {
                try (GGMLReader reader = new GGMLReader(file)) {
                    return reader.getMetadata();
                }
            }
        } catch (IOException e) {
            throw new GGMLImportException("Failed to inspect model: " + file, e);
        }
    }

    /**
     * Detect the format of a GGML/GGUF file
     *
     * @param filePath path to the file
     * @return detected format
     * @throws GGMLImportException if format detection fails
     */
    public static GGMLFormat detectFormat(String filePath) throws GGMLImportException {
        return detectFormat(new File(filePath));
    }

    /**
     * Detect the format of a GGML/GGUF file
     *
     * @param file the file
     * @return detected format
     * @throws GGMLImportException if format detection fails
     */
    public static GGMLFormat detectFormat(File file) throws GGMLImportException {
        try {
            return GGMLFormatDetector.detect(file);
        } catch (IOException e) {
            throw new GGMLImportException("Failed to detect format: " + file, e);
        }
    }

    /**
     * Check if a file appears to be a valid GGML/GGUF file
     *
     * @param filePath path to the file
     * @return true if the file appears to be a valid GGML/GGUF file
     */
    public static boolean isGGMLFile(String filePath) {
        return isGGMLFile(new File(filePath));
    }

    /**
     * Check if a file appears to be a valid GGML/GGUF file
     *
     * @param file the file
     * @return true if the file appears to be a valid GGML/GGUF file
     */
    public static boolean isGGMLFile(File file) {
        try {
            GGMLFormatDetector.detect(file);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
