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

package org.eclipse.deeplearning4j.llm.pipeline;

import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

/**
 * Represents a single stage in a multi-model pipeline.
 *
 * <p>A stage defines which model to use and how to process its output.
 * Multiple stages can share the same model instance for different purposes.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * // Simple stage with default parameters
 * PipelineStage summarization = new PipelineStage("summarizer", "summary");
 *
 * // Stage with custom parameters
 * PipelineStage classification = new PipelineStage("classifier", "category",
 *     Map.of("categories", "spam, ham"));
 *
 * // Stage with structured input from previous stage
 * PipelineStage ner = PipelineStage.fromStructuredOutput("ner", "entities", "classification");
 * }</pre>
 */
@Getter
public class PipelineStage {

    /** The model ID to use for this stage (must be registered in the pipeline). */
    private final String modelId;

    /** The stage ID (unique identifier for this stage's output). */
    private final String stageId;

    /** Stage-specific parameters (overrides model type defaults). */
    private final Map<String, String> parameters;

    /** Whether to use structured output from the previous stage as input. */
    private final boolean useStructuredInput;

    /** The previous stage ID to take input from (null for first stage). */
    private final String inputFromStage;

    /**
     * Create a simple pipeline stage.
     *
     * @param modelId the model ID to use
     * @param stageId the stage ID for output reference
     */
    public PipelineStage(String modelId, String stageId) {
        this(modelId, stageId, new HashMap<>());
    }

    /**
     * Create a pipeline stage with custom parameters.
     *
     * @param modelId the model ID to use
     * @param stageId the stage ID for output reference
     * @param parameters stage-specific parameters
     */
    public PipelineStage(String modelId, String stageId, Map<String, String> parameters) {
        this.modelId = modelId;
        this.stageId = stageId;
        this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
        this.useStructuredInput = false;
        this.inputFromStage = null;
    }

    /**
     * Create a pipeline stage with all parameters.
     *
     * @param modelId the model ID to use
     * @param stageId the stage ID for output reference
     * @param parameters stage-specific parameters
     * @param useStructuredInput whether to use structured output from previous stage
     * @param inputFromStage the previous stage ID to take input from
     */
    public PipelineStage(
            String modelId,
            String stageId,
            Map<String, String> parameters,
            boolean useStructuredInput,
            String inputFromStage) {
        this.modelId = modelId;
        this.stageId = stageId;
        this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
        this.useStructuredInput = useStructuredInput;
        this.inputFromStage = inputFromStage;
    }

    /**
     * Create a stage that uses structured output from a previous stage.
     *
     * @param modelId the model ID to use
     * @param stageId the stage ID for output reference
     * @param inputFromStage the previous stage ID to take input from
     * @return a new stage configuration
     */
    public static PipelineStage fromStructuredOutput(
            String modelId, String stageId, String inputFromStage) {
        return new PipelineStage(modelId, stageId, new HashMap<>(), true, inputFromStage);
    }

    /**
     * Create a builder for PipelineStage.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for PipelineStage.
     */
    public static class Builder {
        private String modelId;
        private String stageId;
        private Map<String, String> parameters = new HashMap<>();
        private boolean useStructuredInput = false;
        private String inputFromStage;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder stageId(String stageId) {
            this.stageId = stageId;
            return this;
        }

        public Builder parameters(Map<String, String> parameters) {
            this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
            return this;
        }

        public Builder parameter(String key, String value) {
            if (this.parameters == null) {
                this.parameters = new HashMap<>();
            }
            this.parameters.put(key, value);
            return this;
        }

        public Builder useStructuredInput(boolean useStructuredInput) {
            this.useStructuredInput = useStructuredInput;
            return this;
        }

        public Builder inputFromStage(String inputFromStage) {
            this.inputFromStage = inputFromStage;
            return this;
        }

        public PipelineStage build() {
            return new PipelineStage(modelId, stageId, parameters, useStructuredInput, inputFromStage);
        }
    }
}
