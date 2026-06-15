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

package org.eclipse.deeplearning4j.pipeline;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Auto-detecting model loader following HuggingFace's from_pretrained pattern.
 */
@Slf4j
public class AutoModel {

    private AutoModel() {}

    public static SameDiff fromPretrained(String path) throws IOException {
        return fromPretrained(new File(path), PipelineLoader.LoadConfig.defaults());
    }

    public static SameDiff fromPretrained(File path) throws IOException {
        return fromPretrained(path, PipelineLoader.LoadConfig.defaults());
    }

    public static SameDiff fromPretrained(String path, PipelineLoader.LoadConfig config) throws IOException {
        return fromPretrained(new File(path), config);
    }

    public static SameDiff fromPretrained(File path, PipelineLoader.LoadConfig config) throws IOException {
        log.info("Loading model from: {}", path.getAbsolutePath());

        if (path.isFile()) {
            return loadFromFile(path, config);
        }

        ModelManifest manifest = ModelManifest.fromDirectory(path);
        log.debug("Detected manifest: format={}, type={}", manifest.getFormat(), manifest.getType());

        if (config.cacheConvertedModel()) {
            Optional<SameDiff> cached = loadCachedModel(manifest, config);
            if (cached.isPresent()) {
                log.info("Loaded from cache");
                return cached.get();
            }
        }

        if (manifest.getFormat() == ModelFormat.SDZ) {
            return loadSdz(manifest);
        }

        PipelineLoader loader = PipelineLoaderRegistry.requireLoader(manifest.getFormat());
        log.info("Using loader: {}", loader.getName());

        SameDiff model = loader.loadModel(manifest, config);

        if (config.cacheConvertedModel() && loader.convertsToSdz()) {
            cacheModel(manifest, model, config);
        }

        return model;
    }

    public static Pipeline pipelineFromPretrained(String path) throws IOException {
        return pipelineFromPretrained(new File(path), PipelineLoader.LoadConfig.defaults());
    }

    public static Pipeline pipelineFromPretrained(File path) throws IOException {
        return pipelineFromPretrained(path, PipelineLoader.LoadConfig.defaults());
    }

    public static Pipeline pipelineFromPretrained(File path, PipelineLoader.LoadConfig config) throws IOException {
        log.info("Loading pipeline from: {}", path.getAbsolutePath());
        ModelManifest manifest = ModelManifest.fromDirectory(path);

        if (!manifest.isPipeline()) {
            SameDiff model = fromPretrained(path, config);
            Map<String, SameDiff> components = new LinkedHashMap<>();
            components.put("model", model);
            return Pipeline.builder().manifest(manifest).components(components).build();
        }

        Map<String, SameDiff> components = new LinkedHashMap<>();
        for (String name : manifest.getComponentNames()) {
            ModelManifest cm = manifest.getComponent(name);
            if (cm != null) {
                try {
                    log.debug("Loading component: {}", name);
                    components.put(name, fromPretrained(cm.getModelDirectory(), config));
                } catch (Exception e) {
                    log.warn("Failed to load component '{}': {}", name, e.getMessage());
                }
            }
        }

        return Pipeline.builder()
                .manifest(manifest)
                .modelIndex(manifest.getModelIndex())
                .components(components)
                .build();
    }

    private static SameDiff loadFromFile(File file, PipelineLoader.LoadConfig config) throws IOException {
        ModelFormat format = ModelFormat.fromFilename(file.getName());
        if (format == ModelFormat.SDZ) {
            return SDZSerializer.load(file, false);
        }
        PipelineLoader loader = PipelineLoaderRegistry.requireLoader(format);
        return loader.loadModel(file, config);
    }

    private static SameDiff loadSdz(ModelManifest manifest) throws IOException {
        File f = manifest.getPrimaryWeightFile();
        if (f == null) throw new IOException("No SDZ file found in: " + manifest.getModelDirectory());
        return SDZSerializer.load(f, false);
    }

    private static Optional<SameDiff> loadCachedModel(ModelManifest manifest, PipelineLoader.LoadConfig config) {
        File cacheFile = getCacheFile(manifest, config);
        if (cacheFile != null && cacheFile.exists()) {
            try {
                return Optional.of(SDZSerializer.load(cacheFile, false));
            } catch (IOException e) {
                log.debug("Failed to load cached model: {}", e.getMessage());
            }
        }
        return Optional.empty();
    }

    private static void cacheModel(ModelManifest manifest, SameDiff model, PipelineLoader.LoadConfig config) {
        File cacheFile = getCacheFile(manifest, config);
        if (cacheFile != null) {
            try {
                File cacheDir = cacheFile.getParentFile();
                if (!cacheDir.exists()) cacheDir.mkdirs();
                SDZSerializer.save(model, cacheFile, false, null);
                log.debug("Cached model to: {}", cacheFile);
            } catch (IOException e) {
                log.warn("Failed to cache model: {}", e.getMessage());
            }
        }
    }

    private static File getCacheFile(ModelManifest manifest, PipelineLoader.LoadConfig config) {
        File cacheDir = config.getCacheDirectory();
        if (cacheDir == null) return null;
        String name = manifest.getModelDirectory().getName();
        String hash = Integer.toHexString(manifest.getModelDirectory().getAbsolutePath().hashCode());
        return new File(cacheDir, name + "_" + hash + ".sdz");
    }

    public static ModelManifest inspect(String path) throws IOException {
        return ModelManifest.fromDirectory(new File(path));
    }

    public static ModelManifest inspect(File path) throws IOException {
        return ModelManifest.fromDirectory(path);
    }
}
