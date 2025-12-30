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

package org.nd4j.ggml.convert;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.ggml.format.*;
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Main converter class for GGML/GGUF to SameDiff/SDZ conversion.
 */
@Slf4j
public class GGMLToSameDiffConverter {

    private final ConversionOptions options;

    public GGMLToSameDiffConverter() {
        this(ConversionOptions.builder().build());
    }

    public GGMLToSameDiffConverter(ConversionOptions options) {
        this.options = options;
    }

    /**
     * Convert a GGML/GGUF file to SameDiff graph
     */
    public SameDiff convert(File ggmlFile) throws GGMLImportException {
        try {
            // Validate file
            if (!ggmlFile.exists()) {
                throw new GGMLImportException("File not found: " + ggmlFile);
            }

            if (options.getMaxFileSize() > 0 && ggmlFile.length() > options.getMaxFileSize()) {
                throw new GGMLImportException("File too large: " + ggmlFile.length() +
                        " bytes (max: " + options.getMaxFileSize() + ")");
            }

            // Detect format
            GGMLFormat format = GGMLFormatDetector.detect(ggmlFile);
            log.info("Detected format: {} for file: {}", format, ggmlFile.getName());

            // Read metadata and tensors based on format
            GGMLMetadata metadata;
            Map<String, INDArray> weights;

            if (format == GGMLFormat.GGUF) {
                try (GGUFReader reader = new GGUFReader(ggmlFile)) {
                    metadata = reader.getMetadata();
                    weights = loadWeightsFromGGUF(reader, metadata.getTensors());
                }
            } else {
                try (GGMLReader reader = new GGMLReader(ggmlFile)) {
                    metadata = reader.getMetadata();
                    weights = loadWeightsFromGGML(reader, metadata.getTensors());
                }
            }

            log.info("Model: {} ({} tensors, {} parameters)",
                    metadata.getModelName() != null ? metadata.getModelName() : metadata.getArchitecture(),
                    metadata.getTensors().size(),
                    metadata.getTotalParameters());

            // Detect or override architecture
            ModelArchitecture architecture;
            if (options.getArchitectureOverride() != null) {
                architecture = ArchitectureRegistry.getArchitecture(options.getArchitectureOverride());
                if (architecture == null) {
                    throw new GGMLImportException("Unknown architecture: " + options.getArchitectureOverride());
                }
            } else {
                architecture = ArchitectureRegistry.detectArchitecture(metadata);
            }

            if (architecture == null) {
                throw new GGMLImportException("Could not detect architecture for model");
            }

            log.info("Using architecture: {}", architecture.getName());

            // Build SameDiff graph
            SameDiff sd = architecture.buildGraph(metadata, weights, options);

            // Add metadata
            addMetadataToGraph(sd, metadata);

            return sd;

        } catch (IOException e) {
            throw new GGMLImportException("Failed to read GGML file: " + ggmlFile, e);
        }
    }

    /**
     * Convert a GGML/GGUF file and save directly to SDZ format
     */
    public void convertToSDZ(File ggmlFile, File sdzFile) throws GGMLImportException {
        SameDiff sd = convert(ggmlFile);

        try {
            Map<String, String> metadata = new HashMap<>();
            metadata.put("source_format", "ggml");
            metadata.put("source_file", ggmlFile.getName());
            metadata.put("conversion_timestamp", String.valueOf(System.currentTimeMillis()));

            SDZSerializer.save(sd, sdzFile, options.isForTraining(), metadata);
            log.info("Saved SDZ model to: {}", sdzFile.getAbsolutePath());

        } catch (IOException e) {
            throw new GGMLImportException("Failed to save SDZ file: " + sdzFile, e);
        }
    }

    private Map<String, INDArray> loadWeightsFromGGUF(GGUFReader reader, List<GGMLTensorInfo> tensorInfos)
            throws IOException {
        Map<String, INDArray> weights = new LinkedHashMap<>();

        for (GGMLTensorInfo info : tensorInfos) {
            byte[] data = reader.readTensorData(info);
            INDArray array = convertTensorData(data, info);
            weights.put(info.getName(), array);

            if (log.isDebugEnabled()) {
                log.debug("Loaded tensor: {} shape={} type={}",
                        info.getName(), info.getShapeString(), info.getDataType());
            }
        }

        return weights;
    }

    private Map<String, INDArray> loadWeightsFromGGML(GGMLReader reader, List<GGMLTensorInfo> tensorInfos)
            throws IOException {
        Map<String, INDArray> weights = new LinkedHashMap<>();

        for (GGMLTensorInfo info : tensorInfos) {
            byte[] data = reader.readTensorData(info);
            INDArray array = convertTensorData(data, info);
            weights.put(info.getName(), array);
        }

        return weights;
    }

    private INDArray convertTensorData(byte[] data, GGMLTensorInfo info) {
        GGMLDataType dataType = info.getDataType();
        long[] shape = info.getShape();

        if (dataType.isQuantized()) {
            return handleQuantizedTensor(data, dataType, shape);
        } else {
            return handleNonQuantizedTensor(data, dataType, shape);
        }
    }

    private INDArray handleQuantizedTensor(byte[] data, GGMLDataType dataType, long[] shape) {
        DataType targetType = getTargetDataType();

        switch (options.getQuantizationMode()) {
            case PRESERVE_QUANTIZATION:
                // Store raw quantized data - caller needs to handle dequantization
                log.debug("Preserving quantized data for type: {}", dataType);
                return Nd4j.create(data, new long[]{data.length}, DataType.INT8);

            case DEQUANTIZE_TO_FLOAT32:
            case DEQUANTIZE_TO_FLOAT16:
            case DEQUANTIZE_TO_BFLOAT16:
            default:
                if (DequantizerFactory.hasDequantizer(dataType)) {
                    return DequantizerFactory.dequantizeToArray(data, dataType, shape, targetType);
                } else {
                    log.warn("No dequantizer for type {}, storing as raw bytes", dataType);
                    return Nd4j.create(data, new long[]{data.length}, DataType.INT8);
                }
        }
    }

    private INDArray handleNonQuantizedTensor(byte[] data, GGMLDataType dataType, long[] shape) {
        DataType nd4jType = dataType.getNd4jType();
        if (nd4jType == null) {
            throw new IllegalStateException("No ND4J type mapping for: " + dataType);
        }

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        INDArray array;

        switch (nd4jType) {
            case FLOAT:
                float[] floatData = new float[data.length / 4];
                buffer.asFloatBuffer().get(floatData);
                array = Nd4j.create(floatData, shape);
                break;

            case HALF:
                // Read as shorts, ND4J will handle FP16 conversion
                short[] fp16Data = new short[data.length / 2];
                buffer.asShortBuffer().get(fp16Data);
                array = Nd4j.create(fp16Data, shape, DataType.HALF);
                break;

            case BFLOAT16:
                short[] bf16Data = new short[data.length / 2];
                buffer.asShortBuffer().get(bf16Data);
                array = Nd4j.create(bf16Data, shape, DataType.BFLOAT16);
                break;

            case DOUBLE:
                double[] doubleData = new double[data.length / 8];
                buffer.asDoubleBuffer().get(doubleData);
                array = Nd4j.create(doubleData, shape);
                break;

            case BYTE:
                array = Nd4j.create(data, shape, DataType.BYTE);
                break;

            case SHORT:
                short[] shortData = new short[data.length / 2];
                buffer.asShortBuffer().get(shortData);
                array = Nd4j.create(shortData, shape, DataType.SHORT);
                break;

            case INT:
                int[] intData = new int[data.length / 4];
                buffer.asIntBuffer().get(intData);
                array = Nd4j.create(intData, shape, DataType.INT);
                break;

            case LONG:
                long[] longData = new long[data.length / 8];
                buffer.asLongBuffer().get(longData);
                array = Nd4j.create(longData, shape, DataType.LONG);
                break;

            default:
                throw new IllegalStateException("Unsupported data type: " + nd4jType);
        }

        // Convert to target type if needed
        DataType targetType = getTargetDataType();
        if (array.dataType() != targetType) {
            array = array.castTo(targetType);
        }

        return array;
    }

    private DataType getTargetDataType() {
        switch (options.getQuantizationMode()) {
            case DEQUANTIZE_TO_FLOAT16:
                return DataType.HALF;
            case DEQUANTIZE_TO_BFLOAT16:
                return DataType.BFLOAT16;
            default:
                return options.getTargetDataType();
        }
    }

    private void addMetadataToGraph(SameDiff sd, GGMLMetadata metadata) {
        // Model metadata is stored when saving to SDZ format
        // SameDiff doesn't have a direct property API, so we log the metadata
        log.info("Model metadata - name: {}, architecture: {}, layers: {}, hidden: {}, heads: {}, vocab: {}",
                metadata.getModelName(),
                metadata.getArchitecture(),
                metadata.getNumLayers(),
                metadata.getHiddenSize(),
                metadata.getNumAttentionHeads(),
                metadata.getVocabSize());

        if (options.isPreserveTokenizerInfo() && metadata.getTokenizerInfo() != null) {
            GGMLMetadata.TokenizerInfo tokInfo = metadata.getTokenizerInfo();
            log.info("Tokenizer - model: {}, bos_id: {}, eos_id: {}",
                    tokInfo.getModel(),
                    tokInfo.getBosTokenId(),
                    tokInfo.getEosTokenId());
        }
    }
}
