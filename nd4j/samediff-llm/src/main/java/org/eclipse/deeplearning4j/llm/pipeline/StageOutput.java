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

import lombok.Builder;
import lombok.Getter;

/**
 * Output from a single pipeline stage.
 *
 * <p>Contains both raw text output and parsed structured data.</p>
 *
 * @see ModelType.ParsedOutput
 */
@Getter
@Builder
public class StageOutput {

    /** The stage ID that produced this output. */
    private final String stageId;

    /** The model ID used for this stage. */
    private final String modelId;

    /** The model role for this stage. */
    private final ModelRole modelRole;

    /** The raw text output from the model. */
    private final String rawText;

    /** The parsed structured output. */
    private final ModelType.ParsedOutput parsedOutput;

    /** Execution duration in milliseconds. */
    private final long durationMs;

    /** Number of tokens generated. */
    private final int tokenCount;

    /**
     * Get the structured data from the parsed output.
     *
     * @return the structured data (type depends on model role)
     */
    public Object getStructuredData() {
        return parsedOutput != null ? parsedOutput.getStructuredData() : null;
    }

    /**
     * Get metadata from the parsed output.
     *
     * @return the metadata map
     */
    public java.util.Map<String, Object> getMetadata() {
        return parsedOutput != null ? parsedOutput.getMetadata() : new java.util.HashMap<>();
    }

    /**
     * Get a specific metadata value.
     *
     * @param key the metadata key
     * @return the metadata value, or null if not present
     */
    public Object getMetadataValue(String key) {
        return getMetadata().get(key);
    }

    @Override
    public String toString() {
        return "StageOutput{" +
                "stageId='" + stageId + '\'' +
                ", modelId='" + modelId + '\'' +
                ", modelRole=" + modelRole +
                ", tokenCount=" + tokenCount +
                ", durationMs=" + durationMs +
                '}';
    }
}
