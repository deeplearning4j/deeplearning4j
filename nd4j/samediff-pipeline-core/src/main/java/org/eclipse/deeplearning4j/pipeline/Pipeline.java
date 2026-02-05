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

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * A multi-component model pipeline following HuggingFace patterns.
 */
@Data
@Builder
@Slf4j
public class Pipeline implements Closeable {

    private final ModelManifest manifest;
    private final ModelIndex modelIndex;
    private final Map<String, SameDiff> components;
    private final Map<String, Object> metadata;

    public SameDiff getComponent(String name) {
        return components != null ? components.get(name) : null;
    }

    public SameDiff requireComponent(String name) {
        SameDiff c = getComponent(name);
        if (c == null) throw new IllegalArgumentException("Component not found: " + name);
        return c;
    }

    public boolean hasComponent(String name) {
        return components != null && components.containsKey(name);
    }

    public Set<String> getComponentNames() {
        return components != null ? components.keySet() : Collections.emptySet();
    }

    public int getComponentCount() {
        return components != null ? components.size() : 0;
    }

    public SameDiff getVisionEncoder() {
        return getFirstAvailable("vision_encoder", "image_encoder", "visual_encoder", "encoder");
    }

    public SameDiff getTextEncoder() {
        return getFirstAvailable("text_encoder", "text_model");
    }

    public SameDiff getDecoder() {
        return getFirstAvailable("decoder", "decoder_model", "language_model", "model");
    }

    public SameDiff getUnet() {
        return getFirstAvailable("unet", "transformer");
    }

    public SameDiff getVae() {
        return getFirstAvailable("vae", "vae_encoder", "vae_decoder");
    }

    private SameDiff getFirstAvailable(String... names) {
        for (String n : names) {
            SameDiff c = getComponent(n);
            if (c != null) return c;
        }
        return null;
    }

    public void save(File outputDir) throws IOException {
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Could not create output directory: " + outputDir);
        }
        if (modelIndex != null) {
            modelIndex.save(new File(outputDir, ModelIndex.MODEL_INDEX_FILE));
        }
        for (Map.Entry<String, SameDiff> e : components.entrySet()) {
            File componentDir = new File(outputDir, e.getKey());
            if (!componentDir.exists()) componentDir.mkdirs();
            File sdzFile = new File(componentDir, "model.sdz");
            log.info("Saving component '{}' to: {}", e.getKey(), sdzFile);
            org.nd4j.autodiff.samediff.serde.SDZSerializer.save(e.getValue(), sdzFile, false, null);
        }
        log.info("Pipeline saved to: {}", outputDir);
    }

    public static PipelineBuilder builder() {
        return new PipelineBuilder().components(new LinkedHashMap<>()).metadata(new HashMap<>());
    }

    @Override
    public void close() throws IOException {
        if (components != null) components.clear();
    }
}
