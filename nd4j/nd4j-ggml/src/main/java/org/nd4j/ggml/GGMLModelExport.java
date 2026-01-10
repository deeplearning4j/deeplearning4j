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
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.ggml.architecture.ExportArchitecture;
import org.nd4j.ggml.architecture.ExportArchitectureRegistry;
import org.nd4j.ggml.export.ExportOptions;
import org.nd4j.ggml.export.SameDiffToGGMLConverter;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Main entry point for GGML/GGUF model export.
 * This is the reverse of {@link GGMLModelImport}.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Export SameDiff to GGUF
 * GGMLModelExport.exportModel(sameDiff, "model.gguf");
 *
 * // Export with quantization options
 * ExportOptions options = ExportOptions.builder()
 *     .quantizationType(ExportOptions.QuantizationType.Q4_K)
 *     .modelName("my-llama-model")
 *     .build();
 * GGMLModelExport.exportModel(sameDiff, "model-q4k.gguf", options);
 *
 * // Convert SDZ directly to GGUF
 * GGMLModelExport.convertSDZToGGUF("model.sdz", "model.gguf");
 *
 * // Re-quantize existing GGUF
 * GGMLModelExport.requantize("model-f16.gguf", "model-q4k.gguf",
 *                            ExportOptions.QuantizationType.Q4_K);
 * }</pre>
 */
@Slf4j
public class GGMLModelExport {

    private GGMLModelExport() {
        // Utility class
    }

    // ========== Export from SameDiff ==========

    /**
     * Export a SameDiff model to GGUF format with default options
     *
     * @param model      the SameDiff model
     * @param outputPath path for the output GGUF file
     * @throws GGMLExportException if export fails
     */
    public static void exportModel(SameDiff model, String outputPath) throws GGMLExportException {
        exportModel(model, new File(outputPath));
    }

    /**
     * Export a SameDiff model to GGUF format with default options
     *
     * @param model      the SameDiff model
     * @param outputFile the output GGUF file
     * @throws GGMLExportException if export fails
     */
    public static void exportModel(SameDiff model, File outputFile) throws GGMLExportException {
        exportModel(model, outputFile, ExportOptions.builder().build());
    }

    /**
     * Export a SameDiff model to GGUF format with specified options
     *
     * @param model      the SameDiff model
     * @param outputPath path for the output GGUF file
     * @param options    export options
     * @throws GGMLExportException if export fails
     */
    public static void exportModel(SameDiff model, String outputPath, ExportOptions options)
            throws GGMLExportException {
        exportModel(model, new File(outputPath), options);
    }

    /**
     * Export a SameDiff model to GGUF format with specified options
     *
     * @param model      the SameDiff model
     * @param outputFile the output GGUF file
     * @param options    export options
     * @throws GGMLExportException if export fails
     */
    public static void exportModel(SameDiff model, File outputFile, ExportOptions options)
            throws GGMLExportException {
        SameDiffToGGMLConverter converter = new SameDiffToGGMLConverter(options);
        converter.convert(model, outputFile);
    }

    // ========== Convert from SDZ ==========

    /**
     * Convert an SDZ file directly to GGUF format
     *
     * @param sdzPath  path to the input SDZ file
     * @param ggufPath path for the output GGUF file
     * @throws GGMLExportException if conversion fails
     */
    public static void convertSDZToGGUF(String sdzPath, String ggufPath) throws GGMLExportException {
        convertSDZToGGUF(new File(sdzPath), new File(ggufPath));
    }

    /**
     * Convert an SDZ file directly to GGUF format
     *
     * @param sdzFile  the input SDZ file
     * @param ggufFile the output GGUF file
     * @throws GGMLExportException if conversion fails
     */
    public static void convertSDZToGGUF(File sdzFile, File ggufFile) throws GGMLExportException {
        convertSDZToGGUF(sdzFile, ggufFile, ExportOptions.builder().build());
    }

    /**
     * Convert an SDZ file directly to GGUF format with specified options
     *
     * @param sdzPath  path to the input SDZ file
     * @param ggufPath path for the output GGUF file
     * @param options  export options
     * @throws GGMLExportException if conversion fails
     */
    public static void convertSDZToGGUF(String sdzPath, String ggufPath, ExportOptions options)
            throws GGMLExportException {
        convertSDZToGGUF(new File(sdzPath), new File(ggufPath), options);
    }

    /**
     * Convert an SDZ file directly to GGUF format with specified options
     *
     * @param sdzFile  the input SDZ file
     * @param ggufFile the output GGUF file
     * @param options  export options
     * @throws GGMLExportException if conversion fails
     */
    public static void convertSDZToGGUF(File sdzFile, File ggufFile, ExportOptions options)
            throws GGMLExportException {
        try {
            log.info("Loading SDZ model from: {}", sdzFile.getAbsolutePath());
            SameDiff model = SDZSerializer.load(sdzFile, false);
            exportModel(model, ggufFile, options);
        } catch (IOException e) {
            throw new GGMLExportException("Failed to load SDZ file: " + sdzFile, e);
        }
    }

    // ========== Re-quantization ==========

    /**
     * Re-quantize an existing GGUF file with a different quantization type.
     * This loads the GGUF, converts to SameDiff, then exports with new quantization.
     *
     * @param inputGguf   the input GGUF file
     * @param outputGguf  the output GGUF file
     * @param targetQuant the target quantization type
     * @throws GGMLExportException if re-quantization fails
     */
    public static void requantize(String inputGguf, String outputGguf,
                                   ExportOptions.QuantizationType targetQuant) throws GGMLExportException {
        requantize(new File(inputGguf), new File(outputGguf), targetQuant);
    }

    /**
     * Re-quantize an existing GGUF file with a different quantization type.
     *
     * @param inputGguf   the input GGUF file
     * @param outputGguf  the output GGUF file
     * @param targetQuant the target quantization type
     * @throws GGMLExportException if re-quantization fails
     */
    public static void requantize(File inputGguf, File outputGguf,
                                   ExportOptions.QuantizationType targetQuant) throws GGMLExportException {
        try {
            log.info("Re-quantizing {} to {} quantization", inputGguf.getName(), targetQuant);

            // Import the GGUF model
            SameDiff model = GGMLModelImport.importModel(inputGguf);

            // Export with new quantization
            ExportOptions options = ExportOptions.builder()
                    .quantizationType(targetQuant)
                    .build();
            exportModel(model, outputGguf, options);

            log.info("Successfully re-quantized to: {}", outputGguf.getAbsolutePath());

        } catch (GGMLImportException e) {
            throw new GGMLExportException("Failed to import GGUF for re-quantization", e);
        }
    }

    // ========== Inspection ==========

    /**
     * Get the list of supported quantization types
     */
    public static List<String> getSupportedQuantizationTypes() {
        return Arrays.stream(ExportOptions.QuantizationType.values())
                .map(Enum::name)
                .collect(Collectors.toList());
    }

    /**
     * Get the list of supported architectures
     */
    public static Set<String> getSupportedArchitectures() {
        return ExportArchitectureRegistry.getRegisteredArchitectures();
    }

    /**
     * Check if a SameDiff model can be exported to GGML format
     *
     * @param model the model to check
     * @return true if the model can be exported
     */
    public static boolean canExport(SameDiff model) {
        return ExportArchitectureRegistry.detect(model) != null;
    }

    /**
     * Detect the architecture of a SameDiff model
     *
     * @param model the model to analyze
     * @return the detected architecture name, or null if unknown
     */
    public static String detectArchitecture(SameDiff model) {
        ExportArchitecture arch = ExportArchitectureRegistry.detect(model);
        return arch != null ? arch.getArchitectureName() : null;
    }

    /**
     * Validate a SameDiff model for export
     *
     * @param model the model to validate
     * @return list of validation errors (empty if valid)
     */
    public static List<String> validateForExport(SameDiff model) {
        ExportArchitecture arch = ExportArchitectureRegistry.detect(model);
        if (arch == null) {
            return List.of("Could not detect architecture for model");
        }
        return arch.validateForExport(model);
    }
}
