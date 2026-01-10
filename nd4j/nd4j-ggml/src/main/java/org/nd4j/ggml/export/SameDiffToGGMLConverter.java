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

package org.nd4j.ggml.export;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.ggml.GGMLExportException;
import org.nd4j.ggml.architecture.ExportArchitecture;
import org.nd4j.ggml.architecture.ExportArchitectureRegistry;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLFormat;
import org.nd4j.ggml.format.GGUFWriter;
import org.nd4j.ggml.quantization.Quantizer;
import org.nd4j.ggml.quantization.QuantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

/**
 * Main converter class for SameDiff to GGML/GGUF conversion.
 * This is the reverse of {@link org.nd4j.ggml.convert.GGMLToSameDiffConverter}.
 *
 * <p>Example usage:
 * <pre>{@code
 * SameDiffToGGMLConverter converter = new SameDiffToGGMLConverter(
 *     ExportOptions.builder()
 *         .quantizationType(ExportOptions.QuantizationType.Q4_K)
 *         .modelName("my-llama-model")
 *         .build()
 * );
 * converter.convert(sameDiff, new File("model.gguf"));
 * }</pre>
 */
@Slf4j
public class SameDiffToGGMLConverter {

    private final ExportOptions options;

    public SameDiffToGGMLConverter() {
        this(ExportOptions.builder().build());
    }

    public SameDiffToGGMLConverter(ExportOptions options) {
        this.options = options;
    }

    /**
     * Convert a SameDiff model to GGUF format
     *
     * @param model      the SameDiff model to convert
     * @param outputFile the output GGUF file
     * @throws GGMLExportException if conversion fails
     */
    public void convert(SameDiff model, File outputFile) throws GGMLExportException {
        if (options.getOutputFormat() == GGMLFormat.GGUF) {
            convertToGGUF(model, outputFile);
        } else {
            throw new GGMLExportException("Legacy GGML format export not yet implemented");
        }
    }

    private void convertToGGUF(SameDiff model, File outputFile) throws GGMLExportException {
        // 1. Detect or use specified architecture
        ExportArchitecture architecture = ExportArchitectureRegistry.getOrDetect(
                model, options.getArchitectureHint());

        if (architecture == null) {
            throw new GGMLExportException("Could not detect architecture for model. " +
                    "Specify an architecture hint in ExportOptions.");
        }

        log.info("Using architecture: {} for export", architecture.getArchitectureName());

        // 2. Validate model if requested
        if (options.isValidateBeforeExport()) {
            List<String> errors = architecture.validateForExport(model);
            if (!errors.isEmpty()) {
                throw new GGMLExportException("Model validation failed:\n" + String.join("\n", errors));
            }
        }

        // 3. Collect tensors for export
        List<TensorExportInfo> tensors = collectTensorsForExport(model, architecture);

        if (tensors.isEmpty()) {
            throw new GGMLExportException("No tensors found for export");
        }

        log.info("Exporting {} tensors to GGUF format", tensors.size());

        // 4. Build metadata
        Map<String, Object> metadata = architecture.buildMetadata(model, options);

        // 5. Write GGUF file
        try (GGUFWriter writer = new GGUFWriter(outputFile, options.getGgufVersion())) {
            writer.setAlignment(options.getAlignment());

            // Write metadata
            for (Map.Entry<String, Object> entry : metadata.entrySet()) {
                writer.addMetadata(entry.getKey(), entry.getValue());
            }

            // Register tensors
            for (TensorExportInfo tensor : tensors) {
                writer.registerTensor(tensor.getGgmlName(), tensor.getShape(), tensor.getTargetDataType());
            }

            // Write header
            writer.writeHeader();

            // Write tensor data (quantizing as needed)
            long totalBytes = 0;
            for (TensorExportInfo tensor : tensors) {
                byte[] data = prepareTensorData(tensor);
                writer.writeTensorData(tensor.getGgmlName(), data);
                totalBytes += data.length;

                if (log.isDebugEnabled()) {
                    log.debug("Wrote tensor: {} shape={} type={} bytes={}",
                            tensor.getGgmlName(), tensor.getShapeString(),
                            tensor.getTargetDataType(), data.length);
                }
            }

            writer.finalizeFile();

            log.info("Successfully exported model to: {} ({} bytes)",
                    outputFile.getAbsolutePath(), totalBytes);

        } catch (IOException e) {
            throw new GGMLExportException("Failed to write GGUF file: " + outputFile, e);
        }
    }

    private List<TensorExportInfo> collectTensorsForExport(SameDiff model, ExportArchitecture architecture) {
        List<TensorExportInfo> tensors = new ArrayList<>();

        for (SDVariable var : model.variables()) {
            // Only export VARIABLE and CONSTANT types (not placeholders or computed arrays)
            if (var.getVariableType() != VariableType.VARIABLE &&
                var.getVariableType() != VariableType.CONSTANT) {
                continue;
            }

            INDArray arr = var.getArr();
            if (arr == null) {
                log.warn("Variable {} has no array data, skipping", var.name());
                continue;
            }

            // Check for custom name mapping
            String ggmlName = null;
            if (options.getTensorNameMapping() != null) {
                ggmlName = options.getTensorNameMapping().get(var.name());
            }

            // Use architecture mapping if no custom mapping
            if (ggmlName == null) {
                ggmlName = architecture.mapVariableName(var.name());
            }

            if (ggmlName == null) {
                log.debug("No GGML name mapping for variable: {}, skipping", var.name());
                continue;
            }

            // Determine target data type
            GGMLDataType targetType = determineTargetType(ggmlName, architecture);

            tensors.add(TensorExportInfo.builder()
                    .sdVarName(var.name())
                    .ggmlName(ggmlName)
                    .array(arr)
                    .shape(arr.shape())
                    .targetDataType(targetType)
                    .build());
        }

        // Sort by GGML name for consistent output
        tensors.sort(Comparator.comparing(TensorExportInfo::getGgmlName));

        return tensors;
    }

    private GGMLDataType determineTargetType(String ggmlName, ExportArchitecture architecture) {
        // Check for layer-specific overrides
        if (options.getLayerQuantizationOverrides() != null) {
            for (Map.Entry<String, ExportOptions.QuantizationType> entry :
                    options.getLayerQuantizationOverrides().entrySet()) {
                if (ggmlName.matches(entry.getKey())) {
                    return entry.getValue().toGGMLDataType();
                }
            }
        }

        // Use architecture recommendation
        return architecture.getRecommendedQuantType(ggmlName, options);
    }

    private byte[] prepareTensorData(TensorExportInfo tensor) throws GGMLExportException {
        GGMLDataType targetType = tensor.getTargetDataType();
        INDArray arr = tensor.getArray();

        if (targetType.isQuantized()) {
            // Quantize the tensor
            if (!QuantizerFactory.hasQuantizer(targetType)) {
                throw new GGMLExportException("No quantizer available for type: " + targetType);
            }
            Quantizer quantizer = QuantizerFactory.getQuantizer(targetType);
            return quantizer.quantize(arr);
        } else {
            // Convert to target non-quantized type
            return convertToBytes(arr, targetType);
        }
    }

    private byte[] convertToBytes(INDArray arr, GGMLDataType targetType) throws GGMLExportException {
        DataType nd4jType = targetType.getNd4jType();
        if (nd4jType == null) {
            throw new GGMLExportException("No ND4J type mapping for: " + targetType);
        }

        // Cast to target type if needed
        if (arr.dataType() != nd4jType) {
            arr = arr.castTo(nd4jType);
        }

        // Convert to bytes (little-endian)
        long numBytes = arr.length() * nd4jType.width();
        if (numBytes > Integer.MAX_VALUE) {
            throw new GGMLExportException("Tensor too large: " + numBytes + " bytes");
        }

        ByteBuffer buffer = ByteBuffer.allocate((int) numBytes).order(ByteOrder.LITTLE_ENDIAN);

        switch (nd4jType) {
            case FLOAT:
                buffer.asFloatBuffer().put(arr.toFloatVector());
                break;
            case HALF:
                short[] fp16Data = convertToFp16(arr);
                buffer.asShortBuffer().put(fp16Data);
                break;
            case BFLOAT16:
                short[] bf16Data = convertToBf16(arr);
                buffer.asShortBuffer().put(bf16Data);
                break;
            case DOUBLE:
                buffer.asDoubleBuffer().put(arr.toDoubleVector());
                break;
            case INT:
                buffer.asIntBuffer().put(arr.toIntVector());
                break;
            case LONG:
                buffer.asLongBuffer().put(arr.toLongVector());
                break;
            case BYTE:
                buffer.put(arr.data().asBytes());
                break;
            default:
                throw new GGMLExportException("Unsupported data type: " + nd4jType);
        }

        return buffer.array();
    }

    private short[] convertToFp16(INDArray arr) {
        float[] floats = arr.toFloatVector();
        short[] result = new short[floats.length];
        for (int i = 0; i < floats.length; i++) {
            result[i] = floatToFp16(floats[i]);
        }
        return result;
    }

    private short[] convertToBf16(INDArray arr) {
        float[] floats = arr.toFloatVector();
        short[] result = new short[floats.length];
        for (int i = 0; i < floats.length; i++) {
            // BFloat16: truncate mantissa of float
            int bits = Float.floatToIntBits(floats[i]);
            result[i] = (short) (bits >> 16);
        }
        return result;
    }

    private static short floatToFp16(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >> 16) & 0x8000;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        int mantissa = bits & 0x7FFFFF;

        if (exponent <= 0) {
            if (exponent < -10) {
                return (short) sign;
            }
            mantissa |= 0x800000;
            int shift = 1 - exponent;
            mantissa = mantissa >> shift;
            return (short) (sign | (mantissa >> 13));
        } else if (exponent >= 31) {
            if ((bits & 0x7FFFFFFF) > 0x7F800000) {
                return (short) (sign | 0x7E00);
            }
            return (short) (sign | 0x7C00);
        }

        return (short) (sign | (exponent << 10) | (mantissa >> 13));
    }
}
