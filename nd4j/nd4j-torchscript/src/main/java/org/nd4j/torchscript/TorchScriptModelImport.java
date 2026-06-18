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

package org.nd4j.torchscript;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.torchscript.convert.ConversionOptions;
import org.nd4j.torchscript.convert.TorchScriptToSameDiffConverter;
import org.nd4j.torchscript.format.*;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Main entry point for TorchScript and Safetensors model import.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Import a Safetensors model
 * SameDiff model = TorchScriptModelImport.importModel("resnet50.safetensors");
 *
 * // Import a TorchScript model
 * SameDiff model = TorchScriptModelImport.importModel("model.pt");
 *
 * // Import with options
 * ConversionOptions options = ConversionOptions.builder()
 *     .targetDataType(DataType.FLOAT16)
 *     .forTraining(false)
 *     .build();
 * SameDiff model = TorchScriptModelImport.importModel("model.safetensors", options);
 *
 * // Convert directly to SDZ format
 * TorchScriptModelImport.convertToSDZ("model.safetensors", "model.sdz");
 *
 * // Inspect model metadata without full import
 * TorchScriptMetadata metadata = TorchScriptModelImport.inspectModel("model.safetensors");
 * System.out.println("Architecture: " + metadata.getArchitecture());
 * System.out.println("Parameters: " + metadata.getTotalParameters());
 * }</pre>
 */
@Slf4j
public class TorchScriptModelImport {

    private TorchScriptModelImport() {
        // Utility class
    }

    // ========== Import Methods ==========

    /**
     * Import a TorchScript or Safetensors model file to SameDiff.
     *
     * @param filePath path to the model file
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if import fails
     */
    public static SameDiff importModel(String filePath) throws TorchScriptImportException {
        return importModel(new File(filePath));
    }

    /**
     * Import a TorchScript or Safetensors model file to SameDiff with options.
     *
     * @param filePath path to the model file
     * @param options  conversion options
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if import fails
     */
    public static SameDiff importModel(String filePath, ConversionOptions options)
            throws TorchScriptImportException {
        return importModel(new File(filePath), options);
    }

    /**
     * Import a TorchScript or Safetensors model file to SameDiff.
     *
     * @param file the model file
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if import fails
     */
    public static SameDiff importModel(File file) throws TorchScriptImportException {
        return importModel(file, ConversionOptions.builder().build());
    }

    /**
     * Import a TorchScript or Safetensors model file to SameDiff with options.
     *
     * @param file    the model file
     * @param options conversion options
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if import fails
     */
    public static SameDiff importModel(File file, ConversionOptions options)
            throws TorchScriptImportException {
        TorchScriptToSameDiffConverter converter = new TorchScriptToSameDiffConverter(options);
        return converter.convert(file);
    }

    /**
     * Import a model from an input stream.
     *
     * @param inputStream the input stream
     * @param format      the expected format (for file extension hint)
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if import fails
     */
    public static SameDiff importModel(InputStream inputStream, TorchScriptFormat format)
            throws TorchScriptImportException {
        return importModel(inputStream, format, ConversionOptions.builder().build());
    }

    /**
     * Import a model from an input stream with options.
     *
     * @param inputStream the input stream
     * @param format      the expected format (for file extension hint)
     * @param options     conversion options
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if import fails
     */
    public static SameDiff importModel(InputStream inputStream, TorchScriptFormat format,
                                        ConversionOptions options) throws TorchScriptImportException {
        try {
            // Write stream to temp file for memory-mapped access
            String extension = TorchScriptFormatDetector.getExtension(format);
            Path tempFile = Files.createTempFile("torchscript_import_", extension);
            try {
                Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
                return importModel(tempFile.toFile(), options);
            } finally {
                Files.deleteIfExists(tempFile);
            }
        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to read from input stream", e);
        }
    }

    // ========== Conversion Methods ==========

    /**
     * Convert a model file directly to SDZ format.
     *
     * @param modelPath path to the model file
     * @param sdzPath   path for the output SDZ file
     * @throws TorchScriptImportException if conversion fails
     */
    public static void convertToSDZ(String modelPath, String sdzPath)
            throws TorchScriptImportException {
        convertToSDZ(new File(modelPath), new File(sdzPath));
    }

    /**
     * Convert a model file directly to SDZ format with options.
     *
     * @param modelPath path to the model file
     * @param sdzPath   path for the output SDZ file
     * @param options   conversion options
     * @throws TorchScriptImportException if conversion fails
     */
    public static void convertToSDZ(String modelPath, String sdzPath, ConversionOptions options)
            throws TorchScriptImportException {
        convertToSDZ(new File(modelPath), new File(sdzPath), options);
    }

    /**
     * Convert a model file directly to SDZ format.
     *
     * @param modelFile the model file
     * @param sdzFile   the output SDZ file
     * @throws TorchScriptImportException if conversion fails
     */
    public static void convertToSDZ(File modelFile, File sdzFile)
            throws TorchScriptImportException {
        convertToSDZ(modelFile, sdzFile, ConversionOptions.builder().build());
    }

    /**
     * Convert a model file directly to SDZ format with options.
     *
     * @param modelFile the model file
     * @param sdzFile   the output SDZ file
     * @param options   conversion options
     * @throws TorchScriptImportException if conversion fails
     */
    public static void convertToSDZ(File modelFile, File sdzFile, ConversionOptions options)
            throws TorchScriptImportException {
        TorchScriptToSameDiffConverter converter = new TorchScriptToSameDiffConverter(options);
        converter.convertToSDZ(modelFile, sdzFile);
    }

    // ========== Inspection Methods ==========

    /**
     * Inspect model metadata without performing full import.
     *
     * @param filePath path to the model file
     * @return model metadata
     * @throws TorchScriptImportException if inspection fails
     */
    public static TorchScriptMetadata inspectModel(String filePath)
            throws TorchScriptImportException {
        return inspectModel(new File(filePath));
    }

    /**
     * Inspect model metadata without performing full import.
     *
     * @param file the model file
     * @return model metadata
     * @throws TorchScriptImportException if inspection fails
     */
    public static TorchScriptMetadata inspectModel(File file) throws TorchScriptImportException {
        try {
            TorchScriptFormat format = TorchScriptFormatDetector.detect(file);

            switch (format) {
                case SAFETENSORS:
                    try (SafetensorsReader reader = new SafetensorsReader(file)) {
                        return reader.getMetadata();
                    }
                case TORCHSCRIPT_PT:
                    try (TorchScriptReader reader = new TorchScriptReader(file)) {
                        return reader.getMetadata();
                    }
                default:
                    throw new TorchScriptImportException("Unknown format for file: " + file);
            }
        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to inspect model: " + file, e);
        }
    }

    /**
     * Detect the format of a model file.
     *
     * @param filePath path to the file
     * @return detected format
     * @throws TorchScriptImportException if format detection fails
     */
    public static TorchScriptFormat detectFormat(String filePath)
            throws TorchScriptImportException {
        return detectFormat(new File(filePath));
    }

    /**
     * Detect the format of a model file.
     *
     * @param file the file
     * @return detected format
     * @throws TorchScriptImportException if format detection fails
     */
    public static TorchScriptFormat detectFormat(File file) throws TorchScriptImportException {
        try {
            return TorchScriptFormatDetector.detect(file);
        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to detect format: " + file, e);
        }
    }

    /**
     * Check if a file appears to be a valid TorchScript or Safetensors file.
     *
     * @param filePath path to the file
     * @return true if the file appears to be a valid model file
     */
    public static boolean isSupportedFile(String filePath) {
        return isSupportedFile(new File(filePath));
    }

    /**
     * Check if a file appears to be a valid TorchScript or Safetensors file.
     *
     * @param file the file
     * @return true if the file appears to be a valid model file
     */
    public static boolean isSupportedFile(File file) {
        try {
            TorchScriptFormat format = TorchScriptFormatDetector.detect(file);
            return format != TorchScriptFormat.UNKNOWN;
        } catch (Exception e) {
            return false;
        }
    }
}
