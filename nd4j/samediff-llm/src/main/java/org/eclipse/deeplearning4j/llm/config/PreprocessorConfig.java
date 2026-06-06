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

package org.eclipse.deeplearning4j.llm.config;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.shade.jackson.annotation.JsonAnyGetter;
import org.nd4j.shade.jackson.annotation.JsonAnySetter;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonToken;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Canonical configuration for image preprocessing from HuggingFace preprocessor_config.json.
 *
 * <p>This is the single source of truth for preprocessing configuration across the project.
 * It merges fields from both VLM and pipeline-core preprocessing configs.</p>
 *
 * <p>Handles both integer and object forms of size/cropSize fields:</p>
 * <pre>{@code
 * // Integer form
 * {"size": 224}
 *
 * // Object form
 * {"size": {"height": 384, "width": 384}}
 * {"size": {"shortest_edge": 224}}
 * }</pre>
 *
 * Adam Gibson
 */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class PreprocessorConfig {

    public static final String PREPROCESSOR_CONFIG_FILE = "preprocessor_config.json";
    public static final String PROCESSOR_CONFIG_FILE = "processor_config.json";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("image_processor_type")
    private String imageProcessorType;

    @JsonProperty("processor_class")
    private String processorClass;

    @JsonProperty("auto_map")
    private String autoMap;

    @JsonProperty("do_resize")
    private boolean doResize = true;

    @JsonProperty("size")
    @JsonDeserialize(using = ImageSizeDeserializer.class)
    private ImageSize size;

    @JsonProperty("resample")
    private Integer resample;

    @JsonProperty("do_rescale")
    private boolean doRescale = true;

    @JsonProperty("rescale_factor")
    private double rescaleFactor = 1.0 / 255.0;

    @JsonProperty("do_normalize")
    private boolean doNormalize = true;

    @JsonProperty("image_mean")
    private double[] imageMean;

    @JsonProperty("image_std")
    private double[] imageStd;

    @JsonProperty("do_convert_rgb")
    private boolean doConvertRgb = true;

    @JsonProperty("do_center_crop")
    private boolean doCenterCrop = false;

    @JsonProperty("crop_size")
    @JsonDeserialize(using = ImageSizeDeserializer.class)
    private ImageSize cropSize;

    @JsonProperty("do_pad")
    private boolean doPad = false;

    @JsonProperty("pad_size")
    private ImageSize padSize;

    @JsonProperty("max_image_size")
    @JsonDeserialize(using = ImageSizeDeserializer.class)
    private ImageSize maxImageSize;

    @JsonProperty("do_image_splitting")
    private boolean doImageSplitting = false;

    @JsonProperty("padding_mode")
    private String paddingMode;

    @JsonProperty("image_format")
    private String imageFormat;

    // Vision transformer specific
    @JsonProperty("patch_size")
    private Integer patchSize;

    @JsonProperty("num_channels")
    private Integer numChannels = 3;

    // Catch-all for any additional/unknown fields
    private Map<String, Object> additionalProperties = new LinkedHashMap<>();

    @JsonAnySetter
    public void setAdditionalProperty(String key, Object value) {
        additionalProperties.put(key, value);
    }

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    // ==================== Loading ====================

    public static PreprocessorConfig fromFile(File file) throws IOException {
        return MAPPER.readValue(file, PreprocessorConfig.class);
    }

    public static PreprocessorConfig fromJson(String json) throws IOException {
        return MAPPER.readValue(json, PreprocessorConfig.class);
    }

    public static Optional<PreprocessorConfig> loadIfExists(File dir) {
        File preprocessorFile = new File(dir, PREPROCESSOR_CONFIG_FILE);
        if (preprocessorFile.exists()) {
            try {
                return Optional.of(fromFile(preprocessorFile));
            } catch (IOException e) {
                // Fall through
            }
        }
        File processorFile = new File(dir, PROCESSOR_CONFIG_FILE);
        if (processorFile.exists()) {
            try {
                return Optional.of(fromFile(processorFile));
            } catch (IOException e) {
                // Fall through
            }
        }
        return Optional.empty();
    }

    public static boolean exists(File dir) {
        return new File(dir, PREPROCESSOR_CONFIG_FILE).exists() ||
                new File(dir, PROCESSOR_CONFIG_FILE).exists();
    }

    // ==================== Loading from model directory ====================

    /**
     * Load the preprocessor config from a model directory.
     * Looks for preprocessor_config.json or processor_config.json.
     *
     * @param modelDir the model directory
     * @return the loaded config
     * @throws IOException if no config file found or parsing fails
     */
    public static PreprocessorConfig fromModelDirectory(File modelDir) throws IOException {
        return loadIfExists(modelDir)
                .orElseThrow(() -> new IOException(
                        "No preprocessor_config.json or processor_config.json found in: " +
                        modelDir.getAbsolutePath() +
                        ". Every VLM model must ship a preprocessor config file."));
    }

    // ==================== Factory Methods ====================

    /**
     * Creates a default PreprocessorConfig suitable for Vision Transformer (ViT) models.
     * Uses standard ImageNet normalization (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).
     * The size must be set separately via {@link #setSize(ImageSize)}.
     *
     * @return a new PreprocessorConfig with ViT defaults
     */
    public static PreprocessorConfig forViT() {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setDoResize(true);
        config.setDoNormalize(true);
        config.setDoRescale(true);
        // Standard ViT normalization
        config.setImageMean(new double[]{0.5, 0.5, 0.5});
        config.setImageStd(new double[]{0.5, 0.5, 0.5});
        return config;
    }

    // ==================== Convenience Methods ====================

    public int getTargetHeight() {
        // For VLMs with tiled image splitting, max_image_size is the tile size
        if (maxImageSize != null) {
            if (maxImageSize.getHeight() != null) return maxImageSize.getHeight();
            if (maxImageSize.getLongestEdge() != null) return maxImageSize.getLongestEdge();
            if (maxImageSize.getShortestEdge() != null) return maxImageSize.getShortestEdge();
        }
        if (size != null) {
            if (size.getHeight() != null) return size.getHeight();
            if (size.getShortestEdge() != null) return size.getShortestEdge();
            if (size.getLongestEdge() != null) return size.getLongestEdge();
        }
        throw new IllegalStateException(
                "PreprocessorConfig has no size set. Load from preprocessor_config.json via fromFile() or fromModelDirectory().");
    }

    public int getTargetWidth() {
        // For VLMs with tiled image splitting, max_image_size is the tile size
        if (maxImageSize != null) {
            if (maxImageSize.getWidth() != null) return maxImageSize.getWidth();
            if (maxImageSize.getLongestEdge() != null) return maxImageSize.getLongestEdge();
            if (maxImageSize.getShortestEdge() != null) return maxImageSize.getShortestEdge();
        }
        if (size != null) {
            if (size.getWidth() != null) return size.getWidth();
            if (size.getShortestEdge() != null) return size.getShortestEdge();
            if (size.getLongestEdge() != null) return size.getLongestEdge();
        }
        throw new IllegalStateException(
                "PreprocessorConfig has no size set. Load from preprocessor_config.json via fromFile() or fromModelDirectory().");
    }

    public double[] getMean() {
        return imageMean != null ? imageMean : new double[]{0.485, 0.456, 0.406};
    }

    public double[] getStd() {
        return imageStd != null ? imageStd : new double[]{0.229, 0.224, 0.225};
    }

    public int getEffectiveSize() {
        if (maxImageSize != null) {
            if (maxImageSize.getLongestEdge() != null) return maxImageSize.getLongestEdge();
            if (maxImageSize.getShortestEdge() != null) return maxImageSize.getShortestEdge();
            if (maxImageSize.getHeight() != null) return maxImageSize.getHeight();
        }
        if (size != null) {
            if (size.getShortestEdge() != null) return size.getShortestEdge();
            if (size.getLongestEdge() != null) return size.getLongestEdge();
            if (size.getHeight() != null) return size.getHeight();
        }
        throw new IllegalStateException(
                "PreprocessorConfig has no size set. Load from preprocessor_config.json via fromFile() or fromModelDirectory().");
    }

    public int getEffectiveCropSize() {
        if (cropSize != null) {
            if (cropSize.getHeight() != null) return cropSize.getHeight();
        }
        return getEffectiveSize();
    }

    public boolean shouldNormalize() {
        return doNormalize;
    }

    public boolean shouldRescale() {
        return doRescale;
    }

    // ==================== Inner Classes ====================

    @Data
    @NoArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class ImageSize {
        private Integer height;
        private Integer width;

        @JsonProperty("shortest_edge")
        private Integer shortestEdge;

        @JsonProperty("longest_edge")
        private Integer longestEdge;

        public ImageSize(int height, int width) {
            this.height = height;
            this.width = width;
        }
    }

    /**
     * Deserializer that handles both integer and object forms of size fields.
     * Integer form: {@code "size": 224} becomes ImageSize(224, 224).
     * Object form: {@code "size": {"height": 384, "width": 384}} is deserialized normally.
     */
    public static class ImageSizeDeserializer extends JsonDeserializer<ImageSize> {
        @Override
        public ImageSize deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
            if (p.currentToken() == JsonToken.VALUE_NUMBER_INT) {
                int val = p.getIntValue();
                return new ImageSize(val, val);
            }
            // Delegate to default object deserialization
            return p.readValueAs(ImageSize.class);
        }
    }
}
