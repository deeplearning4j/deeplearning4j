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

package org.nd4j.torchscript.convert;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.torchscript.TorchScriptImportException;
import org.nd4j.torchscript.architecture.ArchitectureRegistry;
import org.nd4j.torchscript.architecture.VisionArchitecture;
import org.nd4j.torchscript.format.*;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Main converter class for TorchScript/Safetensors to SameDiff conversion.
 */
@Slf4j
public class TorchScriptToSameDiffConverter {

    private final ConversionOptions options;

    public TorchScriptToSameDiffConverter() {
        this(ConversionOptions.builder().build());
    }

    public TorchScriptToSameDiffConverter(ConversionOptions options) {
        this.options = options;
    }

    /**
     * Convert a TorchScript/Safetensors file to SameDiff graph.
     *
     * @param file the model file
     * @return SameDiff graph containing the model
     * @throws TorchScriptImportException if conversion fails
     */
    public SameDiff convert(File file) throws TorchScriptImportException {
        try {
            // Validate file
            if (!file.exists()) {
                throw new TorchScriptImportException("File not found: " + file);
            }

            if (options.getMaxFileSize() > 0 && file.length() > options.getMaxFileSize()) {
                throw new TorchScriptImportException("File too large: " + file.length() +
                        " bytes (max: " + options.getMaxFileSize() + ")");
            }

            // Detect format
            TorchScriptFormat format = TorchScriptFormatDetector.detect(file);
            log.info("Detected format: {} for file: {}", format, file.getName());

            // Convert based on format
            switch (format) {
                case SAFETENSORS:
                    return convertFromSafetensors(file);
                case TORCHSCRIPT_PT:
                    return convertFromTorchScript(file);
                case PYTORCH_BIN:
                    return convertFromPytorchBin(file);
                default:
                    throw new TorchScriptImportException("Unknown or unsupported format: " + format);
            }

        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to read file: " + file, e);
        }
    }

    /**
     * Convert a model file and save directly to SDZ format.
     *
     * @param modelFile the model file
     * @param sdzFile   the output SDZ file
     * @throws TorchScriptImportException if conversion fails
     */
    public void convertToSDZ(File modelFile, File sdzFile) throws TorchScriptImportException {
        SameDiff sd = convert(modelFile);

        try {
            Map<String, String> metadata = new HashMap<>();
            metadata.put("source_format", "torchscript");
            metadata.put("source_file", modelFile.getName());
            metadata.put("conversion_timestamp", String.valueOf(System.currentTimeMillis()));

            SDZSerializer.save(sd, sdzFile, options.isForTraining(), metadata);
            log.info("Saved SDZ model to: {}", sdzFile.getAbsolutePath());

        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to save SDZ file: " + sdzFile, e);
        }
    }

    private SameDiff convertFromSafetensors(File file) throws TorchScriptImportException, IOException {
        try (SafetensorsReader reader = new SafetensorsReader(file)) {
            TorchScriptMetadata metadata = reader.getMetadata();
            Map<String, INDArray> weights = loadWeightsFromSafetensors(reader, metadata.getTensors());

            return buildGraph(metadata, weights);
        }
    }

    private SameDiff convertFromTorchScript(File file) throws TorchScriptImportException, IOException {
        try (TorchScriptReader reader = new TorchScriptReader(file)) {
            TorchScriptMetadata metadata = reader.getMetadata();
            Map<String, INDArray> weights = loadWeightsFromTorchScript(reader, metadata.getTensors());

            return buildGraph(metadata, weights);
        }
    }

    private SameDiff convertFromPytorchBin(File file) throws TorchScriptImportException {
        // PyTorch .bin files are just pickle files with tensors
        // For now, treat similarly to TorchScript
        throw new TorchScriptImportException("PyTorch .bin format not yet implemented");
    }

    private SameDiff buildGraph(TorchScriptMetadata metadata, Map<String, INDArray> weights)
            throws TorchScriptImportException {

        log.info("Building graph: {} tensors, {} parameters",
                metadata.getTensors().size(), metadata.getTotalParameters());

        // Detect or override architecture
        VisionArchitecture architecture;
        if (options.getArchitectureOverride() != null) {
            architecture = ArchitectureRegistry.getArchitecture(options.getArchitectureOverride());
            if (architecture == null) {
                throw new TorchScriptImportException("Unknown architecture: " + options.getArchitectureOverride());
            }
        } else {
            architecture = ArchitectureRegistry.detectArchitecture(metadata);
        }

        if (architecture == null) {
            throw new TorchScriptImportException("Could not detect architecture for model");
        }

        log.info("Using architecture: {}", architecture.getName());

        // Build SameDiff graph
        SameDiff sd = architecture.buildGraph(metadata, weights, options);

        return sd;
    }

    private Map<String, INDArray> loadWeightsFromSafetensors(SafetensorsReader reader,
                                                              List<TorchScriptTensorInfo> tensorInfos)
            throws IOException, TorchScriptImportException {

        Map<String, INDArray> weights = new LinkedHashMap<>();

        for (TorchScriptTensorInfo info : tensorInfos) {
            byte[] data = reader.readTensorData(info);
            INDArray array = convertTensorData(data, info);

            // Apply name mapping
            String name = TensorNameMapper.mapName(info.getName(), options.getTensorNameMapping());
            weights.put(name, array);

            if (log.isDebugEnabled()) {
                log.debug("Loaded tensor: {} shape={} type={}",
                        info.getName(), info.getShapeString(), info.getDataType());
            }
        }

        return weights;
    }

    private Map<String, INDArray> loadWeightsFromTorchScript(TorchScriptReader reader,
                                                              List<TorchScriptTensorInfo> tensorInfos)
            throws IOException, TorchScriptImportException {

        Map<String, INDArray> weights = new LinkedHashMap<>();

        for (TorchScriptTensorInfo info : tensorInfos) {
            try {
                byte[] data = reader.readTensorData(info);
                INDArray array = convertTensorData(data, info);

                String name = TensorNameMapper.mapName(info.getName(), options.getTensorNameMapping());
                weights.put(name, array);
            } catch (IOException e) {
                log.warn("Failed to load tensor: {}", info.getName(), e);
            }
        }

        return weights;
    }

    private INDArray convertTensorData(byte[] data, TorchScriptTensorInfo info)
            throws TorchScriptImportException {

        TorchScriptDataType dataType = info.getDataType();
        long[] shape = info.getShape();

        if (dataType == null || !dataType.isSupported()) {
            throw new TorchScriptImportException("Unsupported data type: " + dataType);
        }

        DataType nd4jType = dataType.getNd4jType();
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        INDArray array;

        switch (nd4jType) {
            case FLOAT:
                float[] floatData = new float[data.length / 4];
                buffer.asFloatBuffer().get(floatData);
                array = Nd4j.create(floatData, shape);
                break;

            case HALF:
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

            case BYTE:  // INT8
                array = Nd4j.create(data, shape, DataType.INT8);
                break;

            case SHORT:  // INT16
                short[] shortData = new short[data.length / 2];
                buffer.asShortBuffer().get(shortData);
                array = Nd4j.create(shortData, shape, DataType.INT16);
                break;

            case INT:  // INT32
                int[] intData = new int[data.length / 4];
                buffer.asIntBuffer().get(intData);
                array = Nd4j.create(intData, shape, DataType.INT32);
                break;

            case LONG:  // INT64
                long[] longData = new long[data.length / 8];
                buffer.asLongBuffer().get(longData);
                array = Nd4j.create(longData, shape, DataType.INT64);
                break;

            case UBYTE:  // UINT8
                array = Nd4j.create(data, shape, DataType.UINT8);
                break;

            case BOOL:
                array = Nd4j.create(data, shape, DataType.BOOL);
                break;

            default:
                throw new TorchScriptImportException("Unsupported data type: " + nd4jType);
        }

        // Apply weight transformations if needed
        if (options.isTransformWeights()) {
            array = transformWeights(array, info);
        }

        // Convert to target data type if needed
        DataType targetType = options.getTargetDataType();
        if (array.dataType() != targetType) {
            array = array.castTo(targetType);
        }

        return array;
    }

    /**
     * Transform weights from PyTorch format to ND4J format.
     */
    private INDArray transformWeights(INDArray array, TorchScriptTensorInfo info) {
        String name = info.getName();
        long[] shape = info.getShape();

        // Conv2D weights: PyTorch [out, in, kH, kW] -> ND4J [kH, kW, in, out]
        if (TensorNameMapper.inferLayerType(name).equals("conv2d") &&
                TensorNameMapper.isWeightTensor(name) && shape.length == 4) {
            return array.permute(2, 3, 1, 0);
        }

        // Conv1D weights: PyTorch [out, in, kW] -> ND4J [kW, in, out]
        if (TensorNameMapper.inferLayerType(name).equals("conv1d") &&
                TensorNameMapper.isWeightTensor(name) && shape.length == 3) {
            return array.permute(2, 1, 0);
        }

        // Linear weights: PyTorch [out, in] -> ND4J [in, out] (transpose)
        if (TensorNameMapper.inferLayerType(name).equals("linear") &&
                TensorNameMapper.isWeightTensor(name) && shape.length == 2) {
            return array.transpose();
        }

        return array;
    }
}
