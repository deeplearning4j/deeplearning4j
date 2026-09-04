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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
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
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
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
        validateSourceFile(ggmlFile);
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
     * Convert a GGML/GGUF file to SameDiff graph on an explicitly selected device.
     *
     * <p>This overload is required by pooled model servers: their execution lane owns a stable
     * device affinity, so model loading must not independently choose a different "best" device.
     * The caller remains responsible for selecting the device dynamically.</p>
     *
     * @param ggmlFile the GGML/GGUF file to convert
     * @param targetDevice the device that must own imported model weights
     * @return the converted SameDiff graph
     * @throws GGMLImportException if conversion fails
     */
    public SameDiff convert(File ggmlFile, DeviceDescriptor targetDevice)
            throws GGMLImportException {
        if (targetDevice == null) {
            throw new IllegalArgumentException("targetDevice must not be null");
        }
        validateSourceFile(ggmlFile);
        ModelSizeInfo sizeInfo = estimateModelSize(ggmlFile);
        try (ModelLoadingContext context = ModelLoadingContext.builder()
                .sizeInfo(sizeInfo)
                .targetDevice(targetDevice)
                .asyncEnabled(true)
                .useBatchedNativeTransfer(true)
                .build()) {
            return convert(ggmlFile, context);
        }
    }

    /**
     * Fail-fast source validation shared by every convert entry point. A missing or
     * unreadable source must surface as {@link GGMLImportException} at the import
     * boundary instead of an admission-time IllegalStateException.
     */
    private static void validateSourceFile(File ggmlFile) throws GGMLImportException {
        if (ggmlFile == null) {
            throw new GGMLImportException("GGML file must not be null");
        }
        if (!ggmlFile.exists()) {
            throw new GGMLImportException("GGML file does not exist: " + ggmlFile);
        }
        if (!ggmlFile.isFile()) {
            throw new GGMLImportException("GGML path is not a regular file: " + ggmlFile);
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
        validateSourceFile(ggmlFile);
        try {

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
     * Estimate the destination-array size produced by this converter's current policy.
     * Compressed GGUF storage bytes are not a valid device-admission estimate when tensors are
     * dequantized to FLOAT/HALF/BFLOAT16 during import.
     */
    public ModelSizeInfo estimateModelSize(File ggmlFile) {
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

            List<GGMLTensorInfo> tensorInfos = metadata.getTensors();
            boolean compactTokenEmbedding = shouldUseCompactTokenEmbedding(tensorInfos);
            Map<String, Pair<Long, Long>> manifest = new LinkedHashMap<>();
            long offset = 0;
            for (GGMLTensorInfo info : tensorInfos) {
                long tensorBytes = estimateDestinationBytes(info, compactTokenEmbedding);
                manifest.put(info.getName(), Pair.of(offset, tensorBytes));
                offset = Math.addExact(offset, tensorBytes);
            }

            return ModelSizeInfo.fromManifest(manifest);
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Failed to estimate destination model size for " + ggmlFile, e);
        }
    }

    long estimateDestinationBytes(GGMLTensorInfo info, boolean compactTokenEmbedding) {
        if (info.getDataType().isQuantized()) {
            if (options.getQuantizationMode() == ConversionOptions.QuantizationMode.PRESERVE_QUANTIZATION
                    || shouldUseRuntimeQuantizedMatmul(info)) {
                return info.getDataSize();
            }
            return Math.multiplyExact(info.getNumElements(),
                    (long) getTargetDataType(info, compactTokenEmbedding).width());
        }

        DataType sourceType = info.getDataType().getNd4jType();
        if (sourceType == null) {
            throw new IllegalStateException("No ND4J type mapping for: " + info.getDataType());
        }
        DataType targetType = getTargetDataType(info, compactTokenEmbedding);
        DataType materializedType = sourceType.isFPType() && targetType != null
                ? targetType : sourceType;
        return Math.multiplyExact(info.getNumElements(), (long) materializedType.width());
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
        boolean compactTokenEmbedding = shouldUseCompactTokenEmbedding(tensorInfos);

        log.info("Loading {} tensors from GGUF file (batch GPU transfer: {})",
                totalTensors, context != null ? "enabled" : "disabled");

        for (GGMLTensorInfo info : tensorInfos) {
            try {
                validateRuntimeWeightPolicy(info);
                INDArray array;
                if (info.getDataType().isQuantized()) {
                    // Quantized tensors need dequantization or runtime-packed matmul storage.
                    boolean runtimePackedMatmul = shouldUseRuntimeQuantizedMatmul(info);
                    if (runtimePackedMatmul || options.getQuantizationMode()
                            == ConversionOptions.QuantizationMode.PRESERVE_QUANTIZATION) {
                        // Packed storage keeps the complete payload: read it in one pass.
                        byte[] data = reader.readTensorData(info);
                        array = convertTensorData(data, info, runtimePackedMatmul,
                                getTargetDataType(info, compactTokenEmbedding));
                    } else {
                        // Dense dequantization streams the payload in bounded, block-aligned
                        // chunks so the full packed bytes never coexist with the full output.
                        array = dequantizeTensorStreaming(reader, info,
                                getTargetDataType(info, compactTokenEmbedding));
                    }
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
                    array = convertTensorDataDirect(directData, info,
                            getTargetDataType(info, compactTokenEmbedding));
                }
                weights.put(info.getName(), array);

                // Register with loading context for batch GPU transfer
                if (context != null) {
                    context.onArrayLoaded(array);
                    context.scheduleTransfer(array);
                }

                loadedCount++;
                if (MEMORY_TRACE) {
                    log.info("MEM_TRACE tensor {}/{} {} bytes={} rssKb={}",
                            loadedCount, totalTensors, info.getName(), info.getDataSize(), currentRssKb());
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
            // Non-quantized linear weights (e.g. a lone BF16 projection inside a Q4_K_M
            // GGUF) never violate the packed contract: they simply execute densely, the
            // same way RUNTIME_QUANTIZED_MATMUL's fallback handles them. Throwing here
            // would reject an otherwise fully-packed model over one small dense tensor.
            log.info("Executing linear weight {} densely ({} storage) under packed policy {}",
                    info.getName(), info.getDataType(), options.getQuantizationMode());
            return;
        }
        if (!isRuntimeQuantizationTypeAllowed(info.getDataType())) {
            if (options.getQuantizationMode()
                    == ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL) {
                log.info("Using dense runtime fallback for unsupported quantized linear weight "
                                + "{} shape={} type={}",
                        info.getName(), info.getShapeString(), info.getDataType());
                return;
            }
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
        boolean compactTokenEmbedding = shouldUseCompactTokenEmbedding(tensorInfos);

        log.info("Loading {} tensors from GGML file (batch GPU transfer: {})",
                totalTensors, context != null ? "enabled" : "disabled");

        for (GGMLTensorInfo info : tensorInfos) {
            validateRuntimeWeightPolicy(info);
            byte[] data = reader.readTensorData(info);
            boolean runtimePackedMatmul = shouldUseRuntimeQuantizedMatmul(info);
            INDArray array = convertTensorData(data, info, runtimePackedMatmul,
                    getTargetDataType(info, compactTokenEmbedding));
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

    private INDArray convertTensorData(byte[] data, GGMLTensorInfo info,
                                       boolean allowRuntimePackedMatmul, DataType targetType) {
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
            return handleQuantizedTensor(data, dataType, shape, allowRuntimePackedMatmul, targetType);
        } else {
            return handleNonQuantizedTensor(data, dataType, shape, targetType);
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
                                          boolean allowRuntimePackedMatmul, DataType targetType) {

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
     * Streaming chunk size for dense dequantization, in elements. Chunks are rounded up
     * to whole quantization blocks; 2^23 elements bounds one FLOAT32 chunk at 32 MiB.
     */
    private static final int STREAM_DEQUANT_CHUNK_ELEMENTS = 1 << 23;

    /**
     * Dense dequantization streams only tensors large enough for the transient bound to
     * matter; smaller tensors keep the original whole-tensor semantics (and exotic
     * target dtypes keep it unconditionally, since interval assignment is only
     * supported for the common float storage types).
     */
    private static final long STREAM_DEQUANT_MIN_TENSOR_BYTES = 16L * 1024 * 1024;

    /** Opt-in per-tensor RSS trace: -Dnd4j.ggml.memoryTrace=true (used for import peak attribution). */
    private static final boolean MEMORY_TRACE =
            Boolean.getBoolean("nd4j.ggml.memoryTrace");

    private static long currentRssKb() {
        if (!MEMORY_TRACE) {
            return 0L;
        }
        try {
            for (String line : java.nio.file.Files.readAllLines(
                    java.nio.file.Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("\\D+", ""));
                }
            }
        } catch (Exception ignored) {
        }
        return -1L;
    }

    /**
     * Dequantize a quantized GGUF tensor by streaming block-aligned chunks through the
     * canonical dequantizer into one preallocated dense output. Peak transient memory is
     * bounded to the current chunk (packed bytes + native chunk copy + dequantized chunk)
     * plus the final output, regardless of tensor size.
     */
    private INDArray dequantizeTensorStreaming(GGUFReader reader, GGMLTensorInfo info,
                                               DataType targetType) throws IOException {
        GGMLDataType dataType = info.getDataType();
        if (!DequantizerFactory.hasDequantizer(dataType)) {
            throw new IllegalStateException("No dequantizer is registered for GGUF tensor type " + dataType);
        }
        long[] shape = reverseShape(info.getShape());
        boolean streamableTarget = targetType == DataType.FLOAT || targetType == DataType.HALF
                || targetType == DataType.BFLOAT16;
        if (MEMORY_TRACE) {
            log.info("MEM_TRACE dequant_route tensor={} dataSize={} target={} streamable={}",
                    info.getName(), info.getDataSize(), targetType, streamableTarget);
        }
        if (!streamableTarget || info.getDataSize() < STREAM_DEQUANT_MIN_TENSOR_BYTES) {
            return dequantizeRequired(reader.readTensorData(info), dataType, shape, targetType);
        }
        int blockSize = Math.max(1, dataType.getBlockSize());
        long numElements = 1;
        for (long dim : shape) {
            numElements *= dim;
        }
        long tensorBytes = info.getDataSize();
        if (numElements < 1 || numElements % blockSize != 0
                || tensorBytes <= 0 || tensorBytes > Integer.MAX_VALUE) {
            return dequantizeRequired(reader.readTensorData(info), dataType, shape, targetType);
        }
        long totalBlocks = numElements / blockSize;
        long bytesPerBlock = tensorBytes / totalBlocks;
        if (bytesPerBlock <= 0 || totalBlocks * bytesPerBlock != tensorBytes) {
            // Irregular block geometry: keep the complete-tensor read semantics.
            return dequantizeRequired(reader.readTensorData(info), dataType, shape, targetType);
        }
        long chunkBlocks = Math.max(1, STREAM_DEQUANT_CHUNK_ELEMENTS / blockSize);
        long chunkBytes = chunkBlocks * bytesPerBlock;
        byte[] chunk = new byte[(int) Math.min(tensorBytes, chunkBytes)];

        // Give every chunk its own cycle in one reusable workspace. A single scope around
        // the complete tensor does not reuse memory: each chunk advances the workspace
        // offset, and all external spills remain live until that scope exits. On the Qwen
        // token embedding that retained more than 4 GiB of transient allocations.
        WorkspaceConfiguration streamWsCfg = WorkspaceConfiguration.builder()
                .initialSize(chunkBytes * 4L)
                .maxSize(chunkBytes * 8L)
                .overallocationLimit(0.0)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .build();
        // The chunk loop speaks flat element offsets. Allocate the TENSOR-SHAPED array
        // directly (weights must not be view-backed arrays downstream — DSP capture and
        // constant serialization assume native allocations), and merge chunks through a
        // contiguous 1-D view of that same allocation. reshape on a freshly allocated
        // C-order array is guaranteed to return a view, so writes through it land in
        // the output buffer.
        INDArray output = Nd4j.zeros(targetType, shape);
        INDArray flatOutput = output.reshape(new long[]{numElements});
        MemoryWorkspace streamWorkspace = Nd4j.getWorkspaceManager()
                .createNewWorkspace(streamWsCfg, "gguf-stream-dequant");
        try {
            boolean nativeInto = DequantizerFactory.supportsNativeDequantization(dataType);
            long tailElements = numElements % chunkBlocks;
            // Host staging via one reusable DIRECT ByteBuffer (off-heap, no JVM heap churn);
            // the reader fills it straight from the file channel and Nd4j.createBuffer wraps
            // it as the packed input for the native dequant op. Sized to the full chunk;
            // the tail chunk reads a bounded limit().
            ByteBuffer packedHost = nativeInto
                    ? ByteBuffer.allocateDirect((int) Math.min(tensorBytes, chunkBytes))
                            .order(ByteOrder.LITTLE_ENDIAN) : null;
            try (INDArray fullChunk = nativeInto
                         ? Nd4j.createUninitialized(targetType, chunkBlocks) : null;
                 INDArray tailChunk = nativeInto && tailElements != 0
                         ? Nd4j.createUninitialized(targetType, tailElements) : null) {
                if (MEMORY_TRACE) {
                    log.info("MEM_TRACE stream_alloc output_mb={} rssKb={}",
                            numElements * targetType.width() / (1024 * 1024), currentRssKb());
                }
                long copied = 0;
                while (copied < numElements) {
                    long elements = Math.min(numElements - copied, chunkBlocks);
                    int bytes = (int) ((elements / blockSize) * bytesPerBlock);
                    INDArray chunkArray = null;
                    boolean closeChunkArray = false;
                    try (MemoryWorkspace ignored = streamWorkspace.notifyScopeEntered()) {
                        if (nativeInto) {
                            // Device-aware path: the file channel reads straight into the
                            // direct ByteBuffer (off-heap), Nd4j.createBuffer wraps it as a
                            // host DataBuffer, and each GGMLDequantize execution performs its
                            // own implicit H2D of that input and runs the dequant kernel
                            // device→device into chunkArray. The chunk merge is one assign op
                            // (device→device on CUDA).
                            //
                            // CORRECTNESS: packedHost is REUSED across chunks. The
                            // dequantizeInto H2D of chunk N is asynchronous — overwriting
                            // packedHost with chunk N+1's bytes before that H2D retires would
                            // feed chunk N+1's bytes to chunk N's dequant (silently wrong
                            // weights). A commit() per chunk orders H2D before the host
                            // rewrite. Import remains ~2 minutes (vs 45+ minutes for the old
                            // sync-storm path); the commit is the price of staging reuse.
                            packedHost.clear();
                            packedHost.limit(bytes);
                            reader.readTensorDataRange(info,
                                    (copied / blockSize) * bytesPerBlock,
                                    packedHost);
                            packedHost.rewind();
                            DataBuffer packedBuffer = Nd4j.createBuffer(
                                    packedHost, DataType.INT8, bytes);
                            INDArray rawBytes = Nd4j.create(
                                    packedBuffer, new long[]{bytes},
                                    new long[]{1}, 0, 'c');
                            chunkArray = elements == chunkBlocks ? fullChunk : tailChunk;
                            DequantizerFactory.dequantizeInto(
                                    rawBytes, dataType, new long[]{elements}, chunkArray);
                            INDArray target = flatOutput.get(
                                    NDArrayIndex.interval(copied, copied + elements));
                            target.assign(chunkArray);
                            // Order before the next packedHost rewrite (see comment above).
                            Nd4j.getExecutioner().commit();
                        } else {
                            reader.readTensorDataRange(info,
                                    (copied / blockSize) * bytesPerBlock,
                                    chunk, 0, bytes);
                            byte[] packedChunk = bytes == chunk.length
                                    ? chunk : Arrays.copyOf(chunk, bytes);
                            chunkArray = DequantizerFactory.dequantizeToArray(
                                    packedChunk, dataType, new long[]{elements}, targetType);
                            closeChunkArray = true;
                            // CPU dequantizer fallback: genuinely host-side buffers, host copy.
                            chunkArray.data().copyAtStride(
                                    output.data(), elements, 1, 1, 0, copied);
                        }
                    } finally {
                        if (closeChunkArray && chunkArray != null) {
                            chunkArray.close();
                        }
                    }
                    copied += elements;
                    if (MEMORY_TRACE
                            && copied % (4L * STREAM_DEQUANT_CHUNK_ELEMENTS) == 0) {
                        log.info("MEM_TRACE stream_chunk copied_mb={} rssKb={}",
                                copied * targetType.width() / (1024 * 1024), currentRssKb());
                    }
                }
                return output;
            }
        } catch (RuntimeException | Error failure) {
            output.close();
            throw failure;
        } finally {
            Nd4j.getWorkspaceManager().destroyWorkspace(streamWorkspace);
        }
    }

    /**
     * Convert tensor data from a direct ByteBuffer, avoiding the intermediate byte[] heap copy.
     * Used for non-quantized tensors where the raw bytes can be used directly.
     */
    private INDArray convertTensorDataDirect(ByteBuffer directData, GGMLTensorInfo info,
                                             DataType targetType) {
        // The direct-buffer DataBuffer constructor is not a complete host-authoritative
        // materialization path on every backend. In particular, it can leave the source
        // bytes unregistered before the first device transfer, producing all-zero or
        // non-finite weights. Copy the bounded tensor bytes and use the same canonical
        // type-specific materializer as the regular reader path.
        ByteBuffer readable = directData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        byte[] data = new byte[readable.remaining()];
        readable.get(data);
        return handleNonQuantizedTensor(data, info.getDataType(),
                reverseShape(info.getShape()), targetType);
    }

    private INDArray handleNonQuantizedTensor(byte[] data, GGMLDataType dataType, long[] shape,
                                              DataType targetType) {
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
        switch (nd4jType) {
            case FLOAT: {
                float[] floatData = new float[Math.toIntExact(numElements)];
                bb.asFloatBuffer().get(floatData);
                return castFloatingTarget(Nd4j.create(floatData, shape, 'c'), nd4jType, targetType);
            }
            case HALF: {
                // F16 stores IEEE-754 binary16 bit patterns. Decode to float on the host,
                // then cast through ND4J's normal authoritative host-array path.
                short[] raw = new short[Math.toIntExact(numElements)];
                bb.asShortBuffer().get(raw);
                float[] floatData = new float[raw.length];
                for (int i = 0; i < raw.length; i++) {
                    floatData[i] = halfToFloat(raw[i]);
                }
                return castFloatingTarget(Nd4j.create(floatData, shape, 'c'),
                        DataType.FLOAT, targetType);
            }
            case BFLOAT16: {
                // BF16 is the high 16 bits of an IEEE-754 float. Materialize through FLOAT
                // so the bytes are copied and marked host-authoritative before any backend
                // transfer; casting the resulting array preserves the model's BF16 dtype.
                short[] raw = new short[Math.toIntExact(numElements)];
                bb.asShortBuffer().get(raw);
                float[] floatData = new float[raw.length];
                for (int i = 0; i < raw.length; i++) {
                    floatData[i] = Float.intBitsToFloat((raw[i] & 0xffff) << 16);
                }
                return castFloatingTarget(Nd4j.create(floatData, shape, 'c'),
                        DataType.FLOAT, targetType);
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

    private static float halfToFloat(short bits) {
        int value = bits & 0xffff;
        int sign = (value >>> 15) & 1;
        int exponent = (value >>> 10) & 0x1f;
        int fraction = value & 0x3ff;
        int signBits = sign << 31;

        if (exponent == 0) {
            if (fraction == 0) {
                return Float.intBitsToFloat(signBits);
            }
            while ((fraction & 0x400) == 0) {
                fraction <<= 1;
                exponent--;
            }
            exponent++;
            fraction &= 0x3ff;
        } else if (exponent == 0x1f) {
            return Float.intBitsToFloat(signBits | 0x7f800000 | (fraction << 13));
        }

        int floatExponent = exponent + (127 - 15);
        return Float.intBitsToFloat(signBits | (floatExponent << 23) | (fraction << 13));
    }

    private static INDArray castFloatingTarget(INDArray array, DataType sourceType, DataType targetType) {
        if (sourceType.isFPType() && targetType != null && targetType != sourceType) {
            return array.castTo(targetType);
        }
        return array;
    }

    private boolean shouldUseCompactTokenEmbedding(List<GGMLTensorInfo> tensorInfos) {
        DataType embeddingDataType = options.getEmbeddingDataType();
        if (embeddingDataType == null) {
            return false;
        }
        if (!embeddingDataType.isFPType()) {
            throw new IllegalArgumentException("embeddingDataType must be floating point, but was "
                    + embeddingDataType);
        }

        // A tied input/output table must retain the graph compute dtype: otherwise the final
        // projection would cast the entire vocabulary table back to that dtype at runtime.
        boolean hasDedicatedOutputProjection = tensorInfos.stream()
                .anyMatch(info -> "output.weight".equals(info.getName()));
        if (hasDedicatedOutputProjection) {
            log.info("Storing token_embd.weight as {} and casting only gathered rows", embeddingDataType);
        }
        return hasDedicatedOutputProjection;
    }

    private DataType getTargetDataType(GGMLTensorInfo info, boolean compactTokenEmbedding) {
        if (compactTokenEmbedding && "token_embd.weight".equals(info.getName())) {
            return options.getEmbeddingDataType();
        }
        return getTargetDataType();
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
