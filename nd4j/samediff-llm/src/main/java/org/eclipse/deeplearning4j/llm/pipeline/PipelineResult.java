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
import lombok.Setter;

import java.util.*;

/**
 * Result from executing a multi-model pipeline.
 *
 * <p>Contains outputs from all stages, execution metrics, and error information.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * PipelineResult result = pipeline.execute(stages, "input text");
 *
 * // Get stage output
 * StageOutput summary = result.getStageOutput("summary");
 * String text = summary.getRawText();
 *
 * // Get all outputs in order
 * for (StageOutput output : result.getOutputsInOrder()) {
 *     System.out.println(output.getStageId() + ": " + output.getRawText());
 * }
 *
 * // Get structured data
 * List<ModelType.Entity> entities = (List<ModelType.Entity>) 
 *     result.getStageOutput("entities").getStructuredData();
 * }</pre>
 */
@Getter
public class PipelineResult {

    /** Stage outputs: stageId -> StageOutput */
    private final Map<String, StageOutput> stageOutputs;

    /** Execution order of stages (for ordered iteration). */
    private final List<String> executionOrder;

    /** Total pipeline execution duration in milliseconds. */
    @Setter
    private long totalDurationMs;

    /** Whether the pipeline completed successfully. */
    @Setter
    private boolean success;

    /** Error message if the pipeline failed. */
    @Setter
    private String errorMessage;

    /** Index of the stage that failed (if any). */
    @Setter
    private Integer failedStageIndex;

    /** Input text to the pipeline. */
    @Setter
    private String inputText;

    /** Final output text (from the last stage). */
    @Setter
    private String finalOutput;

    /**
     * Create a new pipeline result.
     */
    public PipelineResult() {
        this.stageOutputs = new LinkedHashMap<>();
        this.executionOrder = new ArrayList<>();
        this.success = false;
    }

    /**
     * Add a stage output to the result.
     *
     * @param stageId the stage ID
     * @param output the stage output
     */
    public void addStageOutput(String stageId, StageOutput output) {
        stageOutputs.put(stageId, output);
        executionOrder.add(stageId);
    }

    /**
     * Get the output from a specific stage.
     *
     * @param stageId the stage ID
     * @return the stage output, or null if not found
     */
    public StageOutput getStageOutput(String stageId) {
        return stageOutputs.get(stageId);
    }

    /**
     * Get all stage outputs in execution order.
     *
     * @return list of stage outputs in order
     */
    public List<StageOutput> getOutputsInOrder() {
        List<StageOutput> outputs = new ArrayList<>();
        for (String stageId : executionOrder) {
            outputs.add(stageOutputs.get(stageId));
        }
        return outputs;
    }

    /**
     * Get the text output from a specific stage.
     *
     * @param stageId the stage ID
     * @return the raw text output, or null if not found
     */
    public String getStageText(String stageId) {
        StageOutput output = stageOutputs.get(stageId);
        return output != null ? output.getRawText() : null;
    }

    /**
     * Get the structured output from a specific stage.
     *
     * @param stageId the stage ID
     * @return the structured data, or null if not found
     */
    public Object getStageStructuredData(String stageId) {
        StageOutput output = stageOutputs.get(stageId);
        return output != null ? output.getStructuredData() : null;
    }

    /**
     * Get the number of stages executed.
     *
     * @return the number of stages
     */
    public int getStageCount() {
        return stageOutputs.size();
    }

    /**
     * Get total tokens generated across all stages.
     *
     * @return total token count
     */
    public int getTotalTokenCount() {
        int total = 0;
        for (StageOutput output : stageOutputs.values()) {
            total += output.getTokenCount();
        }
        return total;
    }

    /**
     * Get a summary of the pipeline result.
     *
     * @return summary string
     */
    public String getSummary() {
        return String.format("PipelineResult{success=%s, stages=%d, totalTokens=%d, duration=%dms}",
                success, getStageCount(), getTotalTokenCount(), totalDurationMs);
    }

    @Override
    public String toString() {
        return getSummary();
    }
}
