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
import org.eclipse.deeplearning4j.pipeline.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Pipeline loader for GGUF/GGML format.
 * Loads llama.cpp compatible models and converts weights to SameDiff variables.
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
        if (!file.exists()) {
            throw new IOException("File not found: " + file);
        }

        log.info("Loading GGUF model from: {}", file.getName());
        SameDiff sd = SameDiff.create();

        try (GGUFReader reader = GGUFReader.open(file)) {
            GGUFHeader header = reader.getHeader();

            storeMetadataInSameDiff(sd, header);

            Map<String, INDArray> tensors = reader.readAllTensors(config.dequantize());
            for (Map.Entry<String, INDArray> entry : tensors.entrySet()) {
                String name = normalizeVariableName(entry.getKey());
                INDArray value = entry.getValue();

                if (config.convertToFloat32() && value.dataType() != org.nd4j.linalg.api.buffer.DataType.FLOAT) {
                    value = value.castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
                }

                sd.constant(name, value);
                log.trace("Added variable: {} with shape {}", name, Arrays.toString(value.shape()));
            }

            log.info("Loaded {} variables from GGUF (architecture: {})",
                    sd.variables().size(), header.getArchitecture());
        }

        return sd;
    }

    @Override
    public Map<String, SameDiff> loadPipeline(ModelManifest manifest, LoadConfig config) throws IOException {
        Map<String, SameDiff> components = new LinkedHashMap<>();
        components.put("model", loadModel(manifest, config));
        return components;
    }

    private void storeMetadataInSameDiff(SameDiff sd, GGUFHeader header) {
        // Log GGUF metadata for reference
        log.debug("GGUF Metadata - Architecture: {}, Model: {}, Context: {}, Embedding: {}, Blocks: {}",
                header.getArchitecture(),
                header.getModelName(),
                header.getContextLength(),
                header.getEmbeddingLength(),
                header.getBlockCount());
    }

    private String normalizeVariableName(String name) {
        return name.replace("/", "_").replace(".", "_").replace(":", "_");
    }

    public static GGUFHeader inspectFile(File file) throws IOException {
        return GGUFHeader.fromFile(file);
    }

    public static Map<String, INDArray> loadWeights(File file, boolean dequantize) throws IOException {
        try (GGUFReader reader = GGUFReader.open(file)) {
            return reader.readAllTensors(dequantize);
        }
    }

    public static Map<String, Object> loadMetadata(File file) throws IOException {
        try (GGUFReader reader = GGUFReader.open(file)) {
            return reader.getMetadata();
        }
    }
}
