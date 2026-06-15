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

package org.eclipse.deeplearning4j.vlm.model.encoder;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.ArrayList;
import java.util.List;

/**
 * Configuration for vision encoder model input/output variable names.
 *
 * <p>Eliminates hardcoded variable name assumptions for the vision encoder.
 * Mirrors {@link org.eclipse.deeplearning4j.llm.generation.ModelIOConfig} which
 * does the same for decoder models.</p>
 *
 * <p>Provides a {@link #discover(SameDiff)} factory that auto-discovers names from
 * the SameDiff graph by inspecting input/output variable names.</p>
 *
 * <h3>Usage:</h3>
 * <pre>{@code
 * // Auto-discover from vision encoder graph
 * VisionEncoderIOConfig ioConfig = VisionEncoderIOConfig.discover(visionEncoder);
 *
 * // Use discovered names
 * Map<String, INDArray> inputs = new HashMap<>();
 * inputs.put(ioConfig.getPixelValuesName(), image);
 * Map<String, INDArray> outputs = visionEncoder.output(inputs, ioConfig.getOutputNames());
 * INDArray embeddings = outputs.get(ioConfig.getPrimaryOutputName());
 * }</pre>
 *
 * @see org.eclipse.deeplearning4j.llm.generation.ModelIOConfig
 */
@Slf4j
@Getter
@Builder
public class VisionEncoderIOConfig {

    /** Name of the pixel values input placeholder (e.g., "pixel_values"). */
    @Builder.Default
    private final String pixelValuesName = "pixel_values";

    /** Name of the pixel attention mask input (e.g., "pixel_attention_mask"). May be null. */
    private final String pixelAttentionMaskName;

    /** Primary output name from the vision encoder (e.g., "image_embeds", "last_hidden_state"). */
    @Builder.Default
    private final String primaryOutputName = "image_embeds";

    /** All output names to request from the vision encoder. */
    @Builder.Default
    private final String[] outputNames = new String[]{"image_embeds"};

    /**
     * Auto-discover a VisionEncoderIOConfig from a SameDiff vision encoder graph.
     *
     * <p>Inspects the model's input and output variable names to find:</p>
     * <ul>
     *   <li>Pixel values input: input containing "pixel_values" or "pixel"</li>
     *   <li>Pixel attention mask: input containing "pixel_attention_mask"</li>
     *   <li>Output names: registered outputs, or variables matching known patterns</li>
     * </ul>
     *
     * @param visionEncoder the SameDiff vision encoder model to inspect
     * @return a VisionEncoderIOConfig with discovered names
     */
    public static VisionEncoderIOConfig discover(SameDiff visionEncoder) {
        // --- Discover inputs ---
        String pixelValuesName = null;
        String pixelAttentionMaskName = null;

        for (String input : visionEncoder.inputs()) {
            String lower = input.toLowerCase();
            if (pixelValuesName == null && lower.contains("pixel_value")) {
                pixelValuesName = input;
            } else if (pixelValuesName == null && lower.contains("pixel") && !lower.contains("mask")) {
                pixelValuesName = input;
            }
            if (pixelAttentionMaskName == null && lower.contains("pixel_attention_mask")) {
                pixelAttentionMaskName = input;
            }
        }

        if (pixelValuesName == null) {
            pixelValuesName = "pixel_values";
        }

        // --- Discover outputs ---
        String primaryOutput = null;
        List<String> outputNames = new ArrayList<>();

        // 1. Try registered outputs
        List<String> registered = visionEncoder.outputs();
        if (registered != null && !registered.isEmpty()) {
            outputNames.addAll(registered);
            primaryOutput = registered.get(0);
            // Prefer last_hidden_state or image_embeds if among registered
            for (String name : registered) {
                String lower = name.toLowerCase();
                if (lower.contains("last_hidden_state") || lower.contains("image_embeds")) {
                    primaryOutput = name;
                    break;
                }
            }
        }

        // 2. Try well-known variable names if no registered outputs
        if (outputNames.isEmpty()) {
            String[] wellKnown = {
                    "image_embeds", "last_hidden_state", "pooler_output",
                    "encoder_output", "hidden_states", "visual_features"
            };
            for (String name : wellKnown) {
                if (visionEncoder.hasVariable(name)) {
                    outputNames.add(name);
                    if (primaryOutput == null) {
                        primaryOutput = name;
                    }
                }
            }
        }

        // 3. Search variables for common ONNX patterns
        if (outputNames.isEmpty()) {
            for (String varName : visionEncoder.variableMap().keySet()) {
                String lower = varName.toLowerCase();
                if (lower.contains("last_hidden_state") || lower.contains("image_embeds")
                        || lower.contains("pooler_output") || lower.contains("encoder_output")) {
                    outputNames.add(varName);
                    if (primaryOutput == null) {
                        primaryOutput = varName;
                    }
                }
            }
        }

        // 4. Final fallback
        if (outputNames.isEmpty()) {
            log.warn("Could not discover vision encoder output names, falling back to 'image_embeds'");
            primaryOutput = "image_embeds";
            outputNames.add("image_embeds");
        }

        VisionEncoderIOConfig config = VisionEncoderIOConfig.builder()
                .pixelValuesName(pixelValuesName)
                .pixelAttentionMaskName(pixelAttentionMaskName)
                .primaryOutputName(primaryOutput)
                .outputNames(outputNames.toArray(new String[0]))
                .build();

        log.info("VisionEncoderIOConfig.discover(): pixelValues={}, pixelAttentionMask={}, primaryOutput={}, outputs={}",
                config.pixelValuesName, config.pixelAttentionMaskName,
                config.primaryOutputName, outputNames);

        return config;
    }

    /**
     * Check if the vision encoder accepts a pixel attention mask input.
     */
    public boolean hasPixelAttentionMask() {
        return pixelAttentionMaskName != null;
    }
}
