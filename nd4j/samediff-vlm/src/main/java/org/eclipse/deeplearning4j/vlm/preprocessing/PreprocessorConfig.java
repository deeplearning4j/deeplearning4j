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

package org.eclipse.deeplearning4j.vlm.preprocessing;

import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Configuration for image preprocessing from HuggingFace preprocessor_config.json.
 *
 * This configuration specifies how to preprocess images for vision models:
 * - Image resizing parameters
 * - Normalization mean and std
 * - Rescaling factors
 * - Color conversion settings
 *
 * <p>Example preprocessor_config.json:</p>
 * <pre>{@code
 * {
 *   "do_resize": true,
 *   "size": {"height": 384, "width": 384},
 *   "do_rescale": true,
 *   "rescale_factor": 0.00392156862745098,
 *   "do_normalize": true,
 *   "image_mean": [0.48145466, 0.4578275, 0.40821073],
 *   "image_std": [0.26862954, 0.26130258, 0.27577711]
 * }
 * }</pre>
 *
 * Adam Gibson
 */
@Data
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class PreprocessorConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("image_processor_type")
    private String imageProcessorType;

    @JsonProperty("do_resize")
    private boolean doResize = true;

    @JsonProperty("size")
    private ImageSize size;

    @JsonProperty("resample")
    private Integer resample; // PIL resampling mode

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
    private ImageSize cropSize;

    @JsonProperty("do_pad")
    private boolean doPad = false;

    @JsonProperty("pad_size")
    private ImageSize padSize;

    // Vision transformer specific
    @JsonProperty("patch_size")
    private Integer patchSize;

    @JsonProperty("num_channels")
    private Integer numChannels = 3;

    /**
     * Load config from a preprocessor_config.json file.
     *
     * @param file the config file
     * @return the parsed configuration
     * @throws IOException if reading fails
     */
    public static PreprocessorConfig fromFile(File file) throws IOException {
        return MAPPER.readValue(file, PreprocessorConfig.class);
    }

    /**
     * Load config from a JSON string.
     *
     * @param json the JSON string
     * @return the parsed configuration
     * @throws IOException if parsing fails
     */
    public static PreprocessorConfig fromJson(String json) throws IOException {
        return MAPPER.readValue(json, PreprocessorConfig.class);
    }

    /**
     * Create default config for CLIP-style models.
     *
     * @return CLIP preprocessor config
     */
    public static PreprocessorConfig forClip() {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setDoResize(true);
        config.setSize(new ImageSize(224, 224));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.48145466, 0.4578275, 0.40821073});
        config.setImageStd(new double[]{0.26862954, 0.26130258, 0.27577711});
        return config;
    }

    /**
     * Create default config for ViT models.
     *
     * @return ViT preprocessor config
     */
    public static PreprocessorConfig forViT() {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setDoResize(true);
        config.setSize(new ImageSize(224, 224));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.5, 0.5, 0.5});
        config.setImageStd(new double[]{0.5, 0.5, 0.5});
        return config;
    }

    /**
     * Create config for SmolDocling.
     *
     * @return SmolDocling preprocessor config
     */
    public static PreprocessorConfig forSmolDocling() {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setDoResize(true);
        config.setSize(new ImageSize(512, 512)); // SmolDocling vision encoder expects 512x512
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.48145466, 0.4578275, 0.40821073});
        config.setImageStd(new double[]{0.26862954, 0.26130258, 0.27577711});
        return config;
    }

    /**
     * Get the target height for resizing.
     *
     * @return target height, or 224 as default
     */
    public int getTargetHeight() {
        if (size != null) {
            if (size.getHeight() != null) return size.getHeight();
            if (size.getShortestEdge() != null) return size.getShortestEdge();
        }
        return 224;
    }

    /**
     * Get the target width for resizing.
     *
     * @return target width, or 224 as default
     */
    public int getTargetWidth() {
        if (size != null) {
            if (size.getWidth() != null) return size.getWidth();
            if (size.getShortestEdge() != null) return size.getShortestEdge();
        }
        return 224;
    }

    /**
     * Get normalization mean values.
     *
     * @return mean values, or ImageNet defaults
     */
    public double[] getMean() {
        return imageMean != null ? imageMean : new double[]{0.485, 0.456, 0.406};
    }

    /**
     * Get normalization std values.
     *
     * @return std values, or ImageNet defaults
     */
    public double[] getStd() {
        return imageStd != null ? imageStd : new double[]{0.229, 0.224, 0.225};
    }

    /**
     * Image size specification.
     */
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
}
