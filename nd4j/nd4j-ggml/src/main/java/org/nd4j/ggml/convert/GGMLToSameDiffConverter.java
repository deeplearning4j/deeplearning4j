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
import org.nd4j.autodiff.samediff.serde.ModelLoadingContext;
import org.nd4j.autodiff.samediff.serde.ModelSizeInfo;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.common.primitives.Pair;
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
     * Convert a GGML/GGUF file to SameDiff graph.
     * This method creates a temporary ModelLoadingContext for batch GPU transfers.
     */
    public SameDiff convert(File ggmlFile) throws GGMLImportException {
        // Create a context for optimized batch loading
        ModelSizeInfo sizeInfo = estimateModelSize(ggmlFile);
        try (ModelLoadingContext context = ModelLoadingContext.builder()
                .sizeInfo(sizeInfo)
                .asyncEnabled(true)
                .useBatchedNativeTransfer(true)
                .build()) {
            return convert(ggmlFile, context);
        }
    }

    /**
     * Convert a GGML/GGUF file to SameDiff graph using the provided loading context.
     * This enables batch GPU transfers to avoid frequent GPU I/O.
     *
     * @param ggmlFile the GGML/GGUF file to convert
     * @param context the loading context for batch GPU transfers (can be null for no batching)
     * @return the converted SameDiff graph
     * @throws GGMLImportException if conversion fails
     */
    public SameDiff convert(File ggmlFile, ModelLoadingContext context) throws GGMLImportException {
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
                    weights = loadWeightsFromGGUF(reader, metadata.getTensors(), context);
                }
            } else {
                try (GGMLReader reader = new GGMLReader(ggmlFile)) {
                    metadata = reader.getMetadata();
                    weights = loadWeightsFromGGML(reader, metadata.getTensors(), context);
                }
            }

            // Wait for all batch GPU transfers to complete
            if (context != null) {
                log.info("Waiting for batch GPU transfers to complete...");
                context.awaitTransfers();
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
     * Estimate the model size from tensor information for pre-allocation.
     */
    private ModelSizeInfo estimateModelSize(File ggmlFile) {
        try {
            GGMLFormat format = GGMLFormatDetector.detect(ggmlFile);
            GGMLMetadata metadata;

            if (format == GGMLFormat.GGUF) {
                try (GGUFReader reader = new GGUFReader(ggmlFile)) {
                    metadata = reader.getMetadata();
                }
            } else {
                try (GGMLReader reader = new GGMLReader(ggmlFile)) {
                    metadata = reader.getMetadata();
                }
            }

            // Build manifest from tensor info
            Map<String, Pair<Long, Long>> manifest = new LinkedHashMap<>();
            long offset = 0;
            for (GGMLTensorInfo info : metadata.getTensors()) {
                long tensorBytes = (long) (info.getNumElements() * info.getDataType().getBytesPerElement());
                manifest.put(info.getName(), Pair.of(offset, tensorBytes));
                offset += tensorBytes;
            }

            return ModelSizeInfo.fromManifest(manifest);
        } catch (Exception e) {
            log.warn("Failed to estimate model size, using file size as fallback: {}", e.getMessage());
            return ModelSizeInfo.builder()
                    .totalBytes(ggmlFile.length())
                    .arrayCount(0)
                    .largestArrayBytes(0)
                    .arraySizes(new LinkedHashMap<>())
                    .build();
        }
    }

    /**
     * Convert a GGML/GGUF file and save directly to SDZ format.
     * Uses batch GPU transfers for optimized loading.
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

    /**
     * Convert a GGML/GGUF file and save directly to SDZ format using the provided loading context.
     *
     * @param ggmlFile the GGML/GGUF file to convert
     * @param sdzFile the output SDZ file
     * @param context the loading context for batch GPU transfers (can be null)
     * @throws GGMLImportException if conversion fails
     */
    public void convertToSDZ(File ggmlFile, File sdzFile, ModelLoadingContext context) throws GGMLImportException {
        SameDiff sd = convert(ggmlFile, context);

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

    /**
     * Create a ModelLoadingContext optimized for a GGML/GGUF file.
     * This analyzes the file to estimate model size and configure optimal batch transfer settings.
     *
     * @param ggmlFile the GGML/GGUF file to analyze
     * @return a configured ModelLoadingContext for optimized batch loading
     */
    public ModelLoadingContext createLoadingContext(File ggmlFile) {
        ModelSizeInfo sizeInfo = estimateModelSize(ggmlFile);
        return ModelLoadingContext.builder()
                .sizeInfo(sizeInfo)
                .asyncEnabled(true)
                .useBatchedNativeTransfer(true)
                .parallelTransfers(ModelLoadingContext.DEFAULT_PARALLEL_TRANSFERS)
                .build();
    }

    /**
     * Load weights from a GGUF file with optional batch GPU transfer support.
     *
     * @param reader the GGUF reader
     * @param tensorInfos list of tensor information
     * @param context optional loading context for batch GPU transfers (can be null)
     * @return map of tensor names to INDArray weights
     */
    private Map<String, INDArray> loadWeightsFromGGUF(GGUFReader reader, List<GGMLTensorInfo> tensorInfos,
                                                       ModelLoadingContext context) throws IOException {
        Map<String, INDArray> weights = new LinkedHashMap<>();
        int totalTensors = tensorInfos.size();
        int loadedCount = 0;

        log.info("Loading {} tensors from GGUF file (batch GPU transfer: {})",
                totalTensors, context != null ? "enabled" : "disabled");

        for (GGMLTensorInfo info : tensorInfos) {
            byte[] data = reader.readTensorData(info);
            INDArray array = convertTensorData(data, info);
            weights.put(info.getName(), array);

            // Register with loading context for batch GPU transfer
            if (context != null) {
                context.onArrayLoaded(array);
                context.scheduleTransfer(array);
            }

            loadedCount++;
            if (log.isDebugEnabled()) {
                log.debug("Loaded tensor {}/{}: {} shape={} type={}",
                        loadedCount, totalTensors, info.getName(), info.getShapeString(), info.getDataType());
            } else if (loadedCount % 50 == 0 || loadedCount == totalTensors) {
                log.info("Loading progress: {}/{} tensors ({}%)",
                        loadedCount, totalTensors, (loadedCount * 100) / totalTensors);
            }
        }

        return weights;
    }

    /**
     * Load weights from a legacy GGML file with optional batch GPU transfer support.
     *
     * @param reader the GGML reader
     * @param tensorInfos list of tensor information
     * @param context optional loading context for batch GPU transfers (can be null)
     * @return map of tensor names to INDArray weights
     */
    private Map<String, INDArray> loadWeightsFromGGML(GGMLReader reader, List<GGMLTensorInfo> tensorInfos,
                                                       ModelLoadingContext context) throws IOException {
        Map<String, INDArray> weights = new LinkedHashMap<>();
        int totalTensors = tensorInfos.size();
        int loadedCount = 0;

        log.info("Loading {} tensors from GGML file (batch GPU transfer: {})",
                totalTensors, context != null ? "enabled" : "disabled");

        for (GGMLTensorInfo info : tensorInfos) {
            byte[] data = reader.readTensorData(info);
            INDArray array = convertTensorData(data, info);
            weights.put(info.getName(), array);

            // Register with loading context for batch GPU transfer
            if (context != null) {
                context.onArrayLoaded(array);
                context.scheduleTransfer(array);
            }

            loadedCount++;
            if (loadedCount % 50 == 0 || loadedCount == totalTensors) {
                log.info("Loading progress: {}/{} tensors ({}%)",
                        loadedCount, totalTensors, (loadedCount * 100) / totalTensors);
            }
        }

        return weights;
    }

    private INDArray convertTensorData(byte[] data, GGMLTensorInfo info) {
        GGMLDataType dataType = info.getDataType();
        long[] ggufShape = info.getShape();

        // GGUF stores dimensions in column-major order: dimension 0 is the innermost
        // (contiguous) dimension.  ND4J uses C-order (row-major) where the LAST
        // dimension is contiguous.  Reverse the shape so the C-order interpretation
        // matches the actual data layout.
        // Example: GGUF token_embd.weight [hidden=1024, vocab=248320]
        //       -> ND4J [vocab=248320, hidden=1024]  (standard [vocabSize, hiddenSize])
        long[] shape = reverseShape(ggufShape);

        if (dataType.isQuantized()) {
            return handleQuantizedTensor(data, dataType, shape);
        } else {
            return handleNonQuantizedTensor(data, dataType, shape);
        }
    }

    /**
     * Reverse shape dimensions to convert from GGUF column-major dimension order
     * to ND4J C-order (row-major) dimension order.
     */
    private static long[] reverseShape(long[] shape) {
        if (shape == null || shape.length <= 1) {
            return shape;
        }
        long[] reversed = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            reversed[i] = shape[shape.length - 1 - i];
        }
        return reversed;
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

        long numElements = 1;
        for (long dim : shape) {
            numElements *= dim;
        }
        if (numElements < 1) numElements = 1;

        // Use type-specific array creation to ensure reliable host-to-device transfer.
        // Direct ByteBuffer creation via Nd4j.createBuffer(ByteBuffer, DataType, int) can
        // leave CUDA device buffers uninitialized when the affinity manager doesn't detect
        // that the host buffer was written. Using Nd4j.create(primitive[], shape) goes through
        // a well-tested path that correctly marks host data as authoritative.
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        switch (nd4jType) {
            case FLOAT: {
                float[] floatData = new float[(int) numElements];
                bb.asFloatBuffer().get(floatData);
                return Nd4j.create(floatData, shape, 'c');
            }
            case HALF: {
                // F16: raw bytes are IEEE 754 half-precision bit patterns.
                // Use direct buffer approach with explicit host sync for CUDA compatibility.
                ByteBuffer directBuffer = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
                directBuffer.put(data);
                directBuffer.rewind();
                org.nd4j.linalg.api.buffer.DataBuffer buf = Nd4j.createBuffer(directBuffer, DataType.HALF, (int) numElements);
                INDArray array = Nd4j.create(buf);
                if (shape.length > 1) {
                    array = array.reshape('c', shape);
                }
                // Force device→host sync so host has correct data for later host→device transfers
                Nd4j.getAffinityManager().ensureLocation(array,
                        org.nd4j.linalg.api.concurrency.AffinityManager.Location.HOST);
                return array;
            }
            case DOUBLE: {
                double[] doubleData = new double[(int) numElements];
                bb.asDoubleBuffer().get(doubleData);
                return Nd4j.create(doubleData, shape, 'c');
            }
            default: {
                // Fallback for other types: use direct buffer approach
                ByteBuffer directBuffer = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
                directBuffer.put(data);
                directBuffer.rewind();
                org.nd4j.linalg.api.buffer.DataBuffer rawDataBuffer = Nd4j.createBuffer(directBuffer, nd4jType, (int) numElements);
                INDArray array = Nd4j.create(rawDataBuffer);
                if (shape.length > 0 && rawDataBuffer.length() > 0) {
                    array = array.reshape('c', shape);
                }
                return array;
            }
        }
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
