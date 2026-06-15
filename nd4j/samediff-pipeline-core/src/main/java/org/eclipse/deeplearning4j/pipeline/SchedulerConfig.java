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
 * Represents scheduler_config.json for diffusion model schedulers.
 * Used by Stable Diffusion and similar models for noise scheduling.
 */
@Data
@Builder
public class SchedulerConfig {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String SCHEDULER_CONFIG_FILE = "scheduler_config.json";

    // Scheduler type
    private String schedulerClass;
    private String type;

    // Core parameters
    private Integer numTrainTimesteps;
    private Double betaStart;
    private Double betaEnd;
    private String betaSchedule;
    private String predictionType;

    // Timestep settings
    private String timestepSpacing;
    private Integer stepsOffset;
    private Boolean rescaleBetasZeroSnr;

    // Sampling settings
    private Boolean clipSample;
    private Double clipSampleRange;
    private Boolean setSampleToAlphasSigmasProd;
    private Boolean thresholding;
    private Double dynamicThresholdingRatio;
    private Double sampleMaxValue;

    // DDPM/DDIM specific
    private Double varianceType;
    private Boolean skipPrkSteps;

    // Euler specific
    private String interpolationType;
    private Boolean useKarrasSigmas;

    // Raw JSON for any additional fields
    private JsonObject rawConfig;

    public static SchedulerConfig fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static SchedulerConfig fromJson(JsonObject json) {
        SchedulerConfigBuilder builder = SchedulerConfig.builder();
        builder.rawConfig(json);

        // Scheduler type
        if (json.has("_class_name")) builder.schedulerClass(json.get("_class_name").getAsString());
        if (json.has("_diffusers_version")) builder.type(json.get("_diffusers_version").getAsString());

        // Core parameters
        if (json.has("num_train_timesteps")) builder.numTrainTimesteps(json.get("num_train_timesteps").getAsInt());
        if (json.has("beta_start")) builder.betaStart(json.get("beta_start").getAsDouble());
        if (json.has("beta_end")) builder.betaEnd(json.get("beta_end").getAsDouble());
        if (json.has("beta_schedule")) builder.betaSchedule(json.get("beta_schedule").getAsString());
        if (json.has("prediction_type")) builder.predictionType(json.get("prediction_type").getAsString());

        // Timestep settings
        if (json.has("timestep_spacing")) builder.timestepSpacing(json.get("timestep_spacing").getAsString());
        if (json.has("steps_offset")) builder.stepsOffset(json.get("steps_offset").getAsInt());
        if (json.has("rescale_betas_zero_snr")) builder.rescaleBetasZeroSnr(json.get("rescale_betas_zero_snr").getAsBoolean());

        // Sampling settings
        if (json.has("clip_sample")) builder.clipSample(json.get("clip_sample").getAsBoolean());
        if (json.has("clip_sample_range")) builder.clipSampleRange(json.get("clip_sample_range").getAsDouble());
        if (json.has("set_alpha_to_one")) builder.setSampleToAlphasSigmasProd(json.get("set_alpha_to_one").getAsBoolean());
        if (json.has("thresholding")) builder.thresholding(json.get("thresholding").getAsBoolean());
        if (json.has("dynamic_thresholding_ratio")) builder.dynamicThresholdingRatio(json.get("dynamic_thresholding_ratio").getAsDouble());
        if (json.has("sample_max_value")) builder.sampleMaxValue(json.get("sample_max_value").getAsDouble());

        // DDPM/DDIM specific
        if (json.has("skip_prk_steps")) builder.skipPrkSteps(json.get("skip_prk_steps").getAsBoolean());

        // Euler specific
        if (json.has("interpolation_type")) builder.interpolationType(json.get("interpolation_type").getAsString());
        if (json.has("use_karras_sigmas")) builder.useKarrasSigmas(json.get("use_karras_sigmas").getAsBoolean());

        return builder.build();
    }

    public static Optional<SchedulerConfig> loadIfExists(File dir) {
        // Check in scheduler subdirectory first (Diffusers pattern)
        File schedulerDir = new File(dir, "scheduler");
        if (schedulerDir.exists() && schedulerDir.isDirectory()) {
            File configFile = new File(schedulerDir, SCHEDULER_CONFIG_FILE);
            if (configFile.exists()) {
                try {
                    return Optional.of(fromFile(configFile));
                } catch (IOException e) {
                    // Fall through
                }
            }
        }

        // Check in root directory
        File configFile = new File(dir, SCHEDULER_CONFIG_FILE);
        if (configFile.exists()) {
            try {
                return Optional.of(fromFile(configFile));
            } catch (IOException e) {
                // Fall through
            }
        }

        return Optional.empty();
    }

    public static boolean exists(File dir) {
        File schedulerDir = new File(dir, "scheduler");
        return new File(schedulerDir, SCHEDULER_CONFIG_FILE).exists() ||
                new File(dir, SCHEDULER_CONFIG_FILE).exists();
    }

    /**
     * Get the scheduler class name.
     */
    public String getSchedulerClassName() {
        return schedulerClass != null ? schedulerClass : "DDPMScheduler";
    }

    /**
     * Get the number of training timesteps, defaulting to 1000.
     */
    public int getEffectiveNumTrainTimesteps() {
        return numTrainTimesteps != null ? numTrainTimesteps : 1000;
    }

    /**
     * Get beta start value, defaulting to DDPM default.
     */
    public double getEffectiveBetaStart() {
        return betaStart != null ? betaStart : 0.0001;
    }

    /**
     * Get beta end value, defaulting to DDPM default.
     */
    public double getEffectiveBetaEnd() {
        return betaEnd != null ? betaEnd : 0.02;
    }

    /**
     * Get the beta schedule type.
     */
    public String getEffectiveBetaSchedule() {
        return betaSchedule != null ? betaSchedule : "linear";
    }

    /**
     * Get the prediction type.
     */
    public String getEffectivePredictionType() {
        return predictionType != null ? predictionType : "epsilon";
    }

    /**
     * Check if this is an Euler-family scheduler.
     */
    public boolean isEulerScheduler() {
        return schedulerClass != null && schedulerClass.toLowerCase().contains("euler");
    }

    /**
     * Check if this is a DDPM/DDIM scheduler.
     */
    public boolean isDDPMScheduler() {
        return schedulerClass != null && (
                schedulerClass.contains("DDPM") ||
                        schedulerClass.contains("DDIM") ||
                        schedulerClass.contains("PNDM")
        );
    }
}
