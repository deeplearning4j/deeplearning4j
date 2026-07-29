/*
 *  ******************************************************************************
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

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.pipeline.ModelFormat;
import org.eclipse.deeplearning4j.pipeline.ModelManifest;
import org.eclipse.deeplearning4j.pipeline.PipelineLoader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGUFHeader;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.linalg.api.buffer.DataType;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Pipeline loader for GGUF models. Model parsing and graph construction are
 * delegated to {@link GGMLModelImport}, the canonical nd4j-ggml importer.
 */
@Slf4j
public class GGMLPipelineLoader implements PipelineLoader {

    @Override
    public String getName() {
        return "GGML";
    }

    @Override
    public ModelFormat getFormat() {
        return ModelFormat.GGUF;
    }

    @Override
    public boolean supports(ModelFormat format) {
        return format == ModelFormat.GGUF;
    }

    @Override
    public boolean convertsToSdz() {
        return true;
    }

    @Override
    public SameDiff loadModel(ModelManifest manifest, LoadConfig config) throws IOException {
        List<File> weightFiles = manifest.getWeightFiles();
        if (weightFiles == null || weightFiles.isEmpty()) {
            throw new IOException("No GGUF weight files found in manifest");
        }
        if (weightFiles.size() > 1) {
            log.warn("Multiple GGUF files found, loading first: {}", weightFiles.get(0));
        }
        return loadModel(weightFiles.get(0), config);
    }

    @Override
    public SameDiff loadModel(File file, LoadConfig config) throws IOException {
        if (!file.isFile()) {
            throw new IOException("GGUF file not found: " + file);
        }

        try {
            return GGMLModelImport.importModel(file, conversionOptions(config));
        } catch (GGMLImportException e) {
            throw new IOException("Failed to import GGUF model: " + file, e);
        }
    }

    @Override
    public Map<String, SameDiff> loadPipeline(ModelManifest manifest, LoadConfig config) throws IOException {
        Map<String, SameDiff> components = new LinkedHashMap<>();
        components.put("model", loadModel(manifest, config));
        return components;
    }

    /**
     * Inspect a GGUF file without loading its tensor data.
     */
    public static GGUFHeader inspectFile(File file) throws IOException {
        try (GGUFReader reader = new GGUFReader(file)) {
            return reader.getHeader();
        } catch (GGMLImportException e) {
            throw new IOException("Failed to inspect GGUF file: " + file, e);
        }
    }

    private static ConversionOptions conversionOptions(LoadConfig config) throws IOException {
        ConversionOptions.ConversionOptionsBuilder builder = ConversionOptions.builder()
                .useMemoryMapping(config.useMmap());

        if (!config.dequantize()) {
            return builder
                    .quantizationMode(ConversionOptions.QuantizationMode.PRESERVE_QUANTIZATION)
                    .build();
        }

        String requestedType = config.getDataType() == null
                ? "float32"
                : config.getDataType().toLowerCase(Locale.ROOT).replace("_", "").replace("-", "");
        if (config.convertToFloat32() || "float".equals(requestedType)
                || "float32".equals(requestedType) || "fp32".equals(requestedType)) {
            return builder
                    .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT32)
                    .targetDataType(DataType.FLOAT)
                    .build();
        }
        if ("half".equals(requestedType) || "float16".equals(requestedType) || "fp16".equals(requestedType)) {
            return builder
                    .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_FLOAT16)
                    .targetDataType(DataType.HALF)
                    .build();
        }
        if ("bfloat16".equals(requestedType) || "bf16".equals(requestedType)) {
            return builder
                    .quantizationMode(ConversionOptions.QuantizationMode.DEQUANTIZE_TO_BFLOAT16)
                    .targetDataType(DataType.BFLOAT16)
                    .build();
        }

        throw new IOException("Unsupported GGUF target data type: " + config.getDataType());
    }
}
