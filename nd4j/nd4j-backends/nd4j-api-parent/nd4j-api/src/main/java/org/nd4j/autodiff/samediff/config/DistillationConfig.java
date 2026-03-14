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

package org.nd4j.autodiff.samediff.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.base.Preconditions;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for knowledge distillation training.
 * <p>
 * Supports multiple distillation strategies:
 * <ul>
 *   <li>LOGIT_KD: Standard Hinton-style logit-level KL divergence</li>
 *   <li>FEATURE_KD: Intermediate layer feature matching (TinyBERT/MiniLM)</li>
 *   <li>ATTENTION_KD: Attention map matching between teacher and student</li>
 *   <li>COMBINED: Weighted combination of multiple strategies</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DistillationConfig implements Serializable {

    /**
     * Distillation strategy type.
     */
    public enum DistillationType {
        LOGIT_KD,
        FEATURE_KD,
        ATTENTION_KD,
        COMBINED
    }

    /**
     * The distillation strategy to use.
     * Default: LOGIT_KD
     */
    @Builder.Default
    private DistillationType distillationType = DistillationType.LOGIT_KD;

    /**
     * Temperature for softmax in KL divergence distillation.
     * Higher values produce softer probability distributions.
     * Default: 4.0
     */
    @Builder.Default
    private double temperature = 4.0;

    /**
     * Weight for the distillation loss relative to hard-label loss.
     * L = alpha * distillation_loss + (1-alpha) * hard_label_loss
     * Default: 0.5
     */
    @Builder.Default
    private double alpha = 0.5;

    /**
     * Mapping of student layer names to teacher layer names for feature distillation.
     * Key: student layer output variable name
     * Value: teacher layer output variable name
     */
    @Builder.Default
    private Map<String, String> featureLayerMappings = new LinkedHashMap<>();

    /**
     * Mapping of student attention variable names to teacher attention variable names.
     * Key: student attention output variable name
     * Value: teacher attention output variable name
     */
    @Builder.Default
    private Map<String, String> attentionLayerMappings = new LinkedHashMap<>();

    /**
     * Weight for feature distillation loss (when using COMBINED mode).
     * Default: 1.0
     */
    @Builder.Default
    private double featureLossWeight = 1.0;

    /**
     * Weight for attention distillation loss (when using COMBINED mode).
     * Default: 1.0
     */
    @Builder.Default
    private double attentionLossWeight = 1.0;

    /**
     * Weight for logit distillation loss (when using COMBINED mode).
     * Default: 1.0
     */
    @Builder.Default
    private double logitLossWeight = 1.0;

    /**
     * Enable progressive distillation with temperature annealing.
     * Temperature will decrease from initialTemperature to 1.0 over the course of training.
     * Default: false
     */
    @Builder.Default
    private boolean temperatureAnnealing = false;

    /**
     * Starting temperature for temperature annealing.
     * Default: 10.0
     */
    @Builder.Default
    private double initialTemperature = 10.0;

    /**
     * Final temperature for temperature annealing.
     * Default: 1.0
     */
    @Builder.Default
    private double finalTemperature = 1.0;

    /**
     * Student logit output variable name.
     */
    private String studentLogitVariable;

    /**
     * Teacher logit output variable name.
     */
    private String teacherLogitVariable;

    /**
     * Get the effective temperature at a given training progress fraction.
     *
     * @param progress Training progress from 0.0 (start) to 1.0 (end)
     * @return The effective temperature
     */
    public double getEffectiveTemperature(double progress) {
        if (!temperatureAnnealing) {
            return temperature;
        }
        // Linear annealing from initialTemperature to finalTemperature
        return initialTemperature + (finalTemperature - initialTemperature) * progress;
    }

    /**
     * Validate the configuration.
     */
    public void validate() {
        Preconditions.checkState(temperature > 0, "Temperature must be positive. Got: %s", temperature);
        Preconditions.checkState(alpha >= 0 && alpha <= 1, "Alpha must be in [0, 1]. Got: %s", alpha);

        if (distillationType == DistillationType.FEATURE_KD || distillationType == DistillationType.COMBINED) {
            Preconditions.checkState(featureLayerMappings != null && !featureLayerMappings.isEmpty(),
                "Feature layer mappings must be provided for FEATURE_KD or COMBINED distillation");
        }

        if (distillationType == DistillationType.ATTENTION_KD || distillationType == DistillationType.COMBINED) {
            Preconditions.checkState(attentionLayerMappings != null && !attentionLayerMappings.isEmpty(),
                "Attention layer mappings must be provided for ATTENTION_KD or COMBINED distillation");
        }

        if (distillationType == DistillationType.LOGIT_KD || distillationType == DistillationType.COMBINED) {
            Preconditions.checkState(studentLogitVariable != null && !studentLogitVariable.isEmpty(),
                "Student logit variable must be specified for LOGIT_KD or COMBINED distillation");
            Preconditions.checkState(teacherLogitVariable != null && !teacherLogitVariable.isEmpty(),
                "Teacher logit variable must be specified for LOGIT_KD or COMBINED distillation");
        }
    }

    /**
     * Create a standard Hinton-style logit KD configuration.
     */
    public static DistillationConfig logitKD(String studentLogitVar, String teacherLogitVar,
                                               double temperature, double alpha) {
        return DistillationConfig.builder()
            .distillationType(DistillationType.LOGIT_KD)
            .studentLogitVariable(studentLogitVar)
            .teacherLogitVariable(teacherLogitVar)
            .temperature(temperature)
            .alpha(alpha)
            .build();
    }

    /**
     * Create a feature-level KD configuration.
     */
    public static DistillationConfig featureKD(Map<String, String> featureMappings) {
        return DistillationConfig.builder()
            .distillationType(DistillationType.FEATURE_KD)
            .featureLayerMappings(featureMappings)
            .build();
    }
}
