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

package org.eclipse.deeplearning4j.onnx;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.pipeline.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Pipeline loader for ONNX format.
 * Wraps the existing ONNX importer to provide a unified pipeline API.
 */
@Slf4j
public class ONNXPipelineLoader implements PipelineLoader {

    private final OnnxFrameworkImporter importer;

    public ONNXPipelineLoader() {
        this.importer = new OnnxFrameworkImporter();
    }

    @Override
    public String getName() {
        return "ONNX";
    }

    @Override
    public ModelFormat getFormat() {
        return ModelFormat.ONNX;
    }

    @Override
    public boolean supports(ModelFormat format) {
        return format == ModelFormat.ONNX;
    }

    @Override
    public boolean convertsToSdz() {
        return true;
    }

    @Override
    public SameDiff loadModel(ModelManifest manifest, LoadConfig config) throws IOException {
        List<File> weightFiles = manifest.getWeightFiles();
        if (weightFiles == null || weightFiles.isEmpty()) {
            throw new IOException("No ONNX model files found in manifest");
        }

        if (weightFiles.size() > 1) {
            log.warn("Multiple ONNX files found, loading first: {}", weightFiles.get(0));
        }

        return loadModel(weightFiles.get(0), config);
    }

    @Override
    public SameDiff loadModel(File file, LoadConfig config) throws IOException {
        if (!file.exists()) {
            throw new IOException("File not found: " + file);
        }

        log.info("Loading ONNX model from: {}", file.getName());

        try {
            SameDiff sd = importer.runImport(file.getAbsolutePath(), Collections.emptyMap(), true, false);

            if (config.convertToFloat32()) {
                convertToFloat32(sd);
            }

            log.info("Loaded ONNX model with {} variables", sd.variables().size());
            return sd;

        } catch (Exception e) {
            throw new IOException("Failed to import ONNX model: " + e.getMessage(), e);
        }
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
            components.put("model", loadModel(manifest, config));
            return components;
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

    private void convertToFloat32(SameDiff sd) {
        for (var variable : sd.variables()) {
            if (variable.getArr() != null &&
                    variable.getArr().dataType() != org.nd4j.linalg.api.buffer.DataType.FLOAT) {
                try {
                    variable.setArray(variable.getArr().castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT));
                } catch (Exception e) {
                    log.trace("Could not convert variable {} to float32", variable.name());
                }
            }
        }
    }

    public static SameDiff importModel(File file) throws IOException {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();
        return loader.loadModel(file, LoadConfig.defaults());
    }

    public static SameDiff importModel(String path) throws IOException {
        return importModel(new File(path));
    }
}
