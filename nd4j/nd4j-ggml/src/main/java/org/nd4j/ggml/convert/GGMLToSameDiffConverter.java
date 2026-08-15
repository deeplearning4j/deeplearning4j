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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
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

            // Build SameDiff graph. Keep this phase distinct from tensor loading so importer
            // failures identify the architecture configuration that reached graph construction.
            SameDiff sd;
            try {
                sd = architecture.buildGraph(metadata, weights, options);
            } catch (RuntimeException failure) {
                throw new GGMLImportException("Could not build " + architecture.getName()
                        + " SameDiff graph after loading " + weights.size() + " entries"
                        + " (modelArchitecture=" + metadata.getArchitecture()
                        + ", layers=" + metadata.getNumLayers()
                        + ", hiddenSize=" + metadata.getHiddenSize()
                        + ", attentionHeads=" + metadata.getNumAttentionHeads()
                        + ", kvHeads=" + metadata.getNumKVHeads()
                        + ", quantizationMode=" + options.getQuantizationMode() + ")", failure);
            }

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
            try {
                validateRuntimeWeightPolicy(info);
                INDArray array;
                if (info.getDataType().isQuantized()) {
                    // Quantized tensors need dequantization or runtime-packed matmul storage.
                    byte[] data = reader.readTensorData(info);
                    boolean runtimePackedMatmul = shouldUseRuntimeQuantizedMatmul(info);
                    array = convertTensorData(data, info, runtimePackedMatmul);
                    // For RUNTIME_QUANTIZED_MATMUL: store companion metadata so the
                    // architecture builder can emit ggml_qmatmul instead of normal mmul.
                    // Key: tensorName + ".__q__"  Value: long[3] = [ggmlQuantType, N, K]
                    if (runtimePackedMatmul) {
                        long[] shape = reverseShape(info.getShape());  // ND4J C-order shape
                        if (shape != null && shape.length == 2) {
                            long N = shape[0];
                            long K = shape[1];
                            int ggmlQuantType = info.getDataType().toGgmlQuantType();
                            INDArray meta = Nd4j.createFromArray(new long[]{ggmlQuantType, N, K});
                            weights.put(info.getName() + ".__q__", meta);
                        }
                    }
                } else {
                    // Non-quantized tensors: use direct ByteBuffer to avoid heap copies
                    ByteBuffer directData = reader.readTensorDataDirect(info);
                    array = convertTensorDataDirect(directData, info);
                }
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
            } catch (IOException failure) {
                throw new IOException(tensorLoadFailure(loadedCount, totalTensors, info), failure);
            } catch (RuntimeException failure) {
                throw new IllegalStateException(
                        tensorLoadFailure(loadedCount, totalTensors, info), failure);
            }
        }

        return weights;
    }

    private static String tensorLoadFailure(int loadedCount, int totalTensors, GGMLTensorInfo info) {
        return "Could not load GGUF tensor " + (loadedCount + 1) + "/" + totalTensors
                + " " + info.getName() + " shape=" + info.getShapeString()
                + " type=" + info.getDataType() + " storageBytes=" + info.getDataSize();
    }

    private boolean shouldUseRuntimeQuantizedMatmul(GGMLTensorInfo info) {
        return options.isRuntimeQuantizedMatmul()
                && isLinearWeight(info)
                && isRuntimeQuantizationTypeAllowed(info.getDataType())
                && hasRuntimePackedRowLayout(info);
    }

    private void validateRuntimeWeightPolicy(GGMLTensorInfo info) {
        if (!options.isRuntimeQuantizedMatmul() || !isLinearWeight(info)) {
            return;
        }
        if (!info.getDataType().isQuantized()) {
            if (options.getQuantizationMode()
                    == ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL) {
                log.info("Using dense runtime fallback for linear weight {} shape={} type={}",
                        info.getName(), info.getShapeString(), info.getDataType());
                return;
            }
            throw new IllegalStateException("Packed " + options.getQuantizationMode()
                    + " requested, but linear weight " + info.getName()
                    + " is stored as " + info.getDataType());
        }
        if (!isRuntimeQuantizationTypeAllowed(info.getDataType())) {
            throw new IllegalStateException("Packed " + options.getQuantizationMode()
                    + " requested, but linear weight " + info.getName()
                    + " uses unsupported GGUF type " + info.getDataType());
        }
        if (!hasRuntimePackedRowLayout(info)
                && options.getQuantizationMode()
                != ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL) {
            throw new IllegalStateException("Packed " + options.getQuantizationMode()
                    + " requested, but linear weight " + info.getName()
                    + " has GGUF row width " + info.getShape()[0]
                    + " which is not divisible by " + info.getDataType().getBlockSize()
                    + " for " + info.getDataType());
        }
    }

    private static boolean hasRuntimePackedRowLayout(GGMLTensorInfo info) {
        long[] shape = info.getShape();
        int blockSize = info.getDataType().getBlockSize();
        return shape != null && shape.length == 2 && shape[0] > 0 && blockSize > 0
                && shape[0] % blockSize == 0;
    }

    private boolean isRuntimeQuantizationTypeAllowed(GGMLDataType dataType) {
        switch (options.getQuantizationMode()) {
            case RUNTIME_QUANTIZED_INT8:
                return dataType == GGMLDataType.GGML_TYPE_Q8_0;
            case RUNTIME_QUANTIZED_INT4:
                return dataType == GGMLDataType.GGML_TYPE_Q4_K
                        || dataType == GGMLDataType.GGML_TYPE_Q6_K;
            case RUNTIME_QUANTIZED_MATMUL:
                return dataType.isRuntimeQMatMulSupported();
            default:
                return false;
        }
    }

    private static boolean isLinearWeight(GGMLTensorInfo info) {
        String name = info.getName();
        long[] shape = info.getShape();
        if (name == null || shape == null || shape.length != 2) {
            return false;
        }
        String normalizedName = name.toLowerCase(Locale.ROOT);
        // Embedding tables are consumed by gather rather than matrix multiplication.
        return !normalizedName.contains("embd") && !normalizedName.contains("embedding");
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
            validateRuntimeWeightPolicy(info);
            byte[] data = reader.readTensorData(info);
            boolean runtimePackedMatmul = shouldUseRuntimeQuantizedMatmul(info);
            INDArray array = convertTensorData(data, info, runtimePackedMatmul);
            if (runtimePackedMatmul) {
                long[] shape = reverseShape(info.getShape());
                long N = shape[0];
                long K = shape[1];
                int ggmlQuantType = info.getDataType().toGgmlQuantType();
                weights.put(info.getName() + ".__q__",
                        Nd4j.createFromArray(new long[]{ggmlQuantType, N, K}));
            }
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

    private INDArray convertTensorData(byte[] data, GGMLTensorInfo info, boolean allowRuntimePackedMatmul) {
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
            return handleQuantizedTensor(data, dataType, shape, allowRuntimePackedMatmul);
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

    private INDArray handleQuantizedTensor(byte[] data, GGMLDataType dataType, long[] shape,
                                          boolean allowRuntimePackedMatmul) {
        DataType targetType = getTargetDataType();

        switch (options.getQuantizationMode()) {
            case PRESERVE_QUANTIZATION:
                log.debug("Preserving quantized data for type: {}", dataType);
                return Nd4j.create(data, new long[]{data.length}, DataType.INT8);

            case RUNTIME_QUANTIZED_MATMUL:
            case RUNTIME_QUANTIZED_INT8:
            case RUNTIME_QUANTIZED_INT4:
                if (allowRuntimePackedMatmul) {
                    log.debug("Runtime quantized matmul: preserving raw bytes for {}", dataType);
                    return Nd4j.create(data, new long[]{data.length}, DataType.INT8);
                }
                // Embeddings and other non-matmul tensors need their logical dense shape.
                return dequantizeRequired(data, dataType, shape, targetType);

            case DEQUANTIZE_TO_FLOAT32:
            case DEQUANTIZE_TO_FLOAT16:
            case DEQUANTIZE_TO_BFLOAT16:
            case DEQUANTIZE_TO_FLOAT8_E4M3:
            case DEQUANTIZE_TO_FLOAT8_E5M2:
            case HYBRID:
            default:
                return dequantizeRequired(data, dataType, shape, targetType);
        }
    }

    private static INDArray dequantizeRequired(byte[] data, GGMLDataType dataType,
                                               long[] shape, DataType targetType) {
        if (!DequantizerFactory.hasDequantizer(dataType)) {
            throw new IllegalStateException("No dequantizer is registered for GGUF tensor type " + dataType);
        }
        return DequantizerFactory.dequantizeToArray(data, dataType, shape, targetType);
    }

    /**
     * Convert tensor data from a direct ByteBuffer, avoiding the intermediate byte[] heap copy.
     * Used for non-quantized tensors where the raw bytes can be used directly.
     */
    private INDArray convertTensorDataDirect(ByteBuffer directData, GGMLTensorInfo info) {
        long[] shape = reverseShape(info.getShape());
        GGMLDataType dataType = info.getDataType();
        DataType nd4jType = dataType.getNd4jType();
        if (nd4jType == null) {
            throw new IllegalStateException("No ND4J type mapping for: " + dataType);
        }

        long numElements = 1;
        for (long dim : shape) numElements *= dim;
        if (numElements < 1) numElements = 1;

        DataType targetType = getTargetDataType();

        // directData is already a direct ByteBuffer in little-endian order.
        // Create ND4J DataBuffer directly from it — no heap copy needed.
        DataBuffer buf = Nd4j.createBuffer(directData, nd4jType, numElements);
        INDArray array = Nd4j.create(buf, shape, Nd4j.getStrides(shape, 'c'), 0, 'c');

        // Ensure host has correct data for later transfers
        Nd4j.getAffinityManager().ensureLocation(array,
                AffinityManager.Location.HOST);

        return castFloatingTarget(array, nd4jType, targetType);
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
        // Direct ByteBuffer creation via Nd4j.createBuffer(ByteBuffer, DataType, long) can
        // leave CUDA device buffers uninitialized when the affinity manager doesn't detect
        // that the host buffer was written. Using Nd4j.create(primitive[], shape) goes through
        // a well-tested path that correctly marks host data as authoritative.
        ByteBuffer bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        // Cast to target dtype if needed — non-quantized tensors (e.g. F16 norm weights
        // in Q8_0 files) must match the target dtype to avoid mixed-precision errors
        // in the SameDiff graph (e.g. FLOAT / HALF mismatches in RMSNorm).
        DataType targetType = getTargetDataType();

        switch (nd4jType) {
            case FLOAT: {
                float[] floatData = new float[Math.toIntExact(numElements)];
                bb.asFloatBuffer().get(floatData);
                return castFloatingTarget(Nd4j.create(floatData, shape, 'c'), nd4jType, targetType);
            }
            case HALF: {
                // F16: raw bytes are IEEE 754 half-precision bit patterns.
                // Use direct buffer approach with explicit host sync for CUDA compatibility.
                ByteBuffer directBuffer = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
                directBuffer.put(data);
                directBuffer.rewind();
                DataBuffer buf = Nd4j.createBuffer(directBuffer, DataType.HALF, numElements);
                // Construct with the exact GGUF shape so rank-0 scalars, rank-1 vectors, and N-D
                // tensors all produce consistent shape info. The previous "create(buf) + conditional
                // reshape" pattern left rank-0 tensors as malformed 1D [1] arrays that later paths
                // (e.g. SameDiffSerializer.saveAutoShard → dup) mistakenly flagged as empty.
                INDArray array = Nd4j.create(buf, shape, Nd4j.getStrides(shape, 'c'), 0, 'c');
                // Force device→host sync so host has correct data for later host→device transfers
                Nd4j.getAffinityManager().ensureLocation(array,
                        AffinityManager.Location.HOST);
                return castFloatingTarget(array, nd4jType, targetType);
            }
            case DOUBLE: {
                double[] doubleData = new double[Math.toIntExact(numElements)];
                bb.asDoubleBuffer().get(doubleData);
                return castFloatingTarget(Nd4j.create(doubleData, shape, 'c'), nd4jType, targetType);
            }
            default: {
                // Other primitive types use the direct buffer path.
                // Construct with the exact GGUF shape so rank-0 scalars, rank-1 vectors, and N-D
                // tensors all produce consistent shape info (see HALF case above).
                ByteBuffer directBuffer = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
                directBuffer.put(data);
                directBuffer.rewind();
                DataBuffer rawDataBuffer = Nd4j.createBuffer(directBuffer, nd4jType, numElements);
                return Nd4j.create(rawDataBuffer, shape, Nd4j.getStrides(shape, 'c'), 0, 'c');
            }
        }
    }

    private static INDArray castFloatingTarget(INDArray array, DataType sourceType, DataType targetType) {
        if (sourceType.isFPType() && targetType != null && targetType != sourceType) {
            return array.castTo(targetType);
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
