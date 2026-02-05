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

package org.eclipse.deeplearning4j.safetensors;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.pipeline.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Pipeline loader for SafeTensors format.
 * Loads HuggingFace SafeTensors models and converts weights to SameDiff variables.
 */
@Slf4j
public class SafeTensorsPipelineLoader implements PipelineLoader {

    @Override
    public String getName() {
        return "SafeTensors";
    }

    @Override
    public ModelFormat getFormat() {
        return ModelFormat.SAFETENSORS;
    }

    @Override
    public boolean supports(ModelFormat format) {
        return format == ModelFormat.SAFETENSORS;
    }

    @Override
    public boolean convertsToSdz() {
        return true;
    }

    @Override
    public SameDiff loadModel(ModelManifest manifest, LoadConfig config) throws IOException {
        List<File> weightFiles = manifest.getWeightFiles();
        if (weightFiles == null || weightFiles.isEmpty()) {
            throw new IOException("No SafeTensors weight files found in manifest");
        }

        log.info("Loading SafeTensors model from {} file(s)", weightFiles.size());
        SameDiff sd = SameDiff.create();

        for (File weightFile : weightFiles) {
            log.debug("Loading weights from: {}", weightFile.getName());
            loadWeightsIntoSameDiff(sd, weightFile, config);
        }

        log.info("Loaded {} variables from SafeTensors", sd.variables().size());
        return sd;
    }

    @Override
    public SameDiff loadModel(File file, LoadConfig config) throws IOException {
        if (!file.exists()) {
            throw new IOException("File not found: " + file);
        }

        SameDiff sd = SameDiff.create();
        loadWeightsIntoSameDiff(sd, file, config);
        return sd;
    }

    @Override
    public Map<String, SameDiff> loadPipeline(ModelManifest manifest, LoadConfig config) throws IOException {
        Map<String, SameDiff> components = new LinkedHashMap<>();

        if (!manifest.isPipeline()) {
            components.put("model", loadModel(manifest, config));
            return components;
        }

        ModelIndex modelIndex = manifest.getModelIndex();
        if (modelIndex == null) {
            throw new IOException("Pipeline manifest has no model index");
        }

        for (String componentName : modelIndex.getComponentNames()) {
            ModelManifest componentManifest = manifest.getComponent(componentName);
            if (componentManifest != null) {
                try {
                    log.info("Loading pipeline component: {}", componentName);
                    SameDiff componentModel = loadModel(componentManifest, config);
                    components.put(componentName, componentModel);
                } catch (Exception e) {
                    log.warn("Failed to load component '{}': {}", componentName, e.getMessage());
                }
            }
        }

        return components;
    }

    private void loadWeightsIntoSameDiff(SameDiff sd, File file, LoadConfig config) throws IOException {
        try (SafeTensorsReader reader = SafeTensorsReader.open(file)) {
            Map<String, INDArray> tensors = reader.readAllTensors();

            for (Map.Entry<String, INDArray> entry : tensors.entrySet()) {
                String name = normalizeVariableName(entry.getKey());
                INDArray value = entry.getValue();

                if (config.convertToFloat32() && value.dataType() != org.nd4j.linalg.api.buffer.DataType.FLOAT) {
                    value = value.castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT);
                }

                sd.constant(name, value);
                log.trace("Added variable: {} with shape {}", name, Arrays.toString(value.shape()));
            }
        }
    }

    private String normalizeVariableName(String name) {
        return name.replace("/", "_").replace(".", "_").replace(":", "_");
    }

    public static Map<String, INDArray> loadWeights(File file) throws IOException {
        return SafeTensorsReader.loadFile(file);
    }

    public static Map<String, INDArray> loadWeights(List<File> files) throws IOException {
        return SafeTensorsReader.loadFiles(files);
    }

    public static SafeTensorsHeader inspectFile(File file) throws IOException {
        return SafeTensorsHeader.fromFile(file);
    }
}
