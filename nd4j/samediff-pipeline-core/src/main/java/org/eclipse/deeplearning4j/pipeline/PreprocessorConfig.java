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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Represents preprocessor_config.json for image/feature preprocessing.
 * Used by vision models for image normalization, resizing, etc.
 */
@Data
@Builder
public class PreprocessorConfig {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String PREPROCESSOR_CONFIG_FILE = "preprocessor_config.json";
    public static final String PROCESSOR_CONFIG_FILE = "processor_config.json";

    // Processor/preprocessor class
    private String imageProcessorType;
    private String processorClass;
    private String autoMap;

    // Image size settings
    private Integer imageSize;
    private Integer size;
    private Map<String, Integer> sizeDict;
    private Integer cropSize;
    private Map<String, Integer> cropSizeDict;
    private Integer shortestEdge;

    // Normalization
    private List<Double> imageMean;
    private List<Double> imageStd;
    private Boolean doNormalize;
    private Double rescaleFactor;
    private Boolean doRescale;

    // Resize settings
    private Boolean doResize;
    private String resample;
    private Boolean doCenterCrop;

    // Convert settings
    private Boolean doConvertRgb;
    private String imageFormat;

    // Padding
    private Boolean doPad;
    private String paddingMode;

    // Raw JSON for any additional fields
    private JsonObject rawConfig;

    public static PreprocessorConfig fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static PreprocessorConfig fromJson(JsonObject json) {
        PreprocessorConfigBuilder builder = PreprocessorConfig.builder();
        builder.rawConfig(json);

        // Processor class info
        if (json.has("image_processor_type")) builder.imageProcessorType(json.get("image_processor_type").getAsString());
        if (json.has("processor_class")) builder.processorClass(json.get("processor_class").getAsString());
        if (json.has("auto_map")) builder.autoMap(json.get("auto_map").toString());

        // Size settings - handle various formats
        if (json.has("image_size")) builder.imageSize(json.get("image_size").getAsInt());
        if (json.has("size")) {
            JsonElement sizeEl = json.get("size");
            if (sizeEl.isJsonPrimitive()) {
                builder.size(sizeEl.getAsInt());
            } else if (sizeEl.isJsonObject()) {
                Map<String, Integer> sizeMap = new LinkedHashMap<>();
                for (Map.Entry<String, JsonElement> e : sizeEl.getAsJsonObject().entrySet()) {
                    sizeMap.put(e.getKey(), e.getValue().getAsInt());
                }
                builder.sizeDict(sizeMap);
                if (sizeMap.containsKey("shortest_edge")) {
                    builder.shortestEdge(sizeMap.get("shortest_edge"));
                } else if (sizeMap.containsKey("height") && sizeMap.containsKey("width")) {
                    builder.size(Math.min(sizeMap.get("height"), sizeMap.get("width")));
                }
            }
        }
        if (json.has("crop_size")) {
            JsonElement cropEl = json.get("crop_size");
            if (cropEl.isJsonPrimitive()) {
                builder.cropSize(cropEl.getAsInt());
            } else if (cropEl.isJsonObject()) {
                Map<String, Integer> cropMap = new LinkedHashMap<>();
                for (Map.Entry<String, JsonElement> e : cropEl.getAsJsonObject().entrySet()) {
                    cropMap.put(e.getKey(), e.getValue().getAsInt());
                }
                builder.cropSizeDict(cropMap);
            }
        }

        // Normalization
        if (json.has("image_mean")) {
            List<Double> mean = new ArrayList<>();
            for (JsonElement el : json.getAsJsonArray("image_mean")) {
                mean.add(el.getAsDouble());
            }
            builder.imageMean(mean);
        }
        if (json.has("image_std")) {
            List<Double> std = new ArrayList<>();
            for (JsonElement el : json.getAsJsonArray("image_std")) {
                std.add(el.getAsDouble());
            }
            builder.imageStd(std);
        }
        if (json.has("do_normalize")) builder.doNormalize(json.get("do_normalize").getAsBoolean());
        if (json.has("rescale_factor")) builder.rescaleFactor(json.get("rescale_factor").getAsDouble());
        if (json.has("do_rescale")) builder.doRescale(json.get("do_rescale").getAsBoolean());

        // Resize settings
        if (json.has("do_resize")) builder.doResize(json.get("do_resize").getAsBoolean());
        if (json.has("resample")) builder.resample(json.get("resample").getAsString());
        if (json.has("do_center_crop")) builder.doCenterCrop(json.get("do_center_crop").getAsBoolean());

        // Convert settings
        if (json.has("do_convert_rgb")) builder.doConvertRgb(json.get("do_convert_rgb").getAsBoolean());
        if (json.has("image_format")) builder.imageFormat(json.get("image_format").getAsString());

        // Padding
        if (json.has("do_pad")) builder.doPad(json.get("do_pad").getAsBoolean());
        if (json.has("padding_mode")) builder.paddingMode(json.get("padding_mode").getAsString());

        return builder.build();
    }

    public static Optional<PreprocessorConfig> loadIfExists(File dir) {
        // Try preprocessor_config.json first
        File preprocessorFile = new File(dir, PREPROCESSOR_CONFIG_FILE);
        if (preprocessorFile.exists()) {
            try {
                return Optional.of(fromFile(preprocessorFile));
            } catch (IOException e) {
                // Fall through
            }
        }
        // Try processor_config.json
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

    /**
     * Get the effective image size for resizing.
     */
    public int getEffectiveSize() {
        if (shortestEdge != null) return shortestEdge;
        if (size != null) return size;
        if (imageSize != null) return imageSize;
        if (sizeDict != null) {
            if (sizeDict.containsKey("shortest_edge")) return sizeDict.get("shortest_edge");
            if (sizeDict.containsKey("height")) return sizeDict.get("height");
        }
        return 224; // Common default
    }

    /**
     * Get the effective crop size.
     */
    public int getEffectiveCropSize() {
        if (cropSize != null) return cropSize;
        if (cropSizeDict != null && cropSizeDict.containsKey("height")) {
            return cropSizeDict.get("height");
        }
        return getEffectiveSize();
    }

    /**
     * Get image mean as array, with ImageNet defaults.
     */
    public double[] getImageMeanArray() {
        if (imageMean != null && !imageMean.isEmpty()) {
            return imageMean.stream().mapToDouble(Double::doubleValue).toArray();
        }
        // ImageNet defaults
        return new double[]{0.485, 0.456, 0.406};
    }

    /**
     * Get image std as array, with ImageNet defaults.
     */
    public double[] getImageStdArray() {
        if (imageStd != null && !imageStd.isEmpty()) {
            return imageStd.stream().mapToDouble(Double::doubleValue).toArray();
        }
        // ImageNet defaults
        return new double[]{0.229, 0.224, 0.225};
    }

    /**
     * Get the rescale factor, defaulting to 1/255.
     */
    public double getEffectiveRescaleFactor() {
        return rescaleFactor != null ? rescaleFactor : 1.0 / 255.0;
    }

    /**
     * Check if normalization should be applied.
     */
    public boolean shouldNormalize() {
        return doNormalize == null || doNormalize;
    }

    /**
     * Check if rescaling should be applied.
     */
    public boolean shouldRescale() {
        return doRescale == null || doRescale;
    }
}
