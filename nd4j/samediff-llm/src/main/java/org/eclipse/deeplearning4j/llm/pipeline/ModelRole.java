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

/**
 * Enumerates the roles a model can play in a multi-model pipeline.
 *
 * <p>Each role defines the model's purpose and expected input/output behavior.
 * The same model instance can be reused across multiple roles if compatible.</p>
 *
 * <p>Supported roles:</p>
 * <ul>
 *   <li>{@link #SUMMARIZATION} - Reduce long text to concise summaries</li>
 *   <li>{@link #CLASSIFICATION} - Categorize text into predefined classes</li>
 *   <li>{@link #SORTING} - Order items by criteria (relevance, priority, etc.)</li>
 *   <li>{@link #RELATION_EXTRACTION} - Extract relationships between entities</li>
 *   <li>{@link #LANGUAGE_CLASSIFICATION} - Detect the language of text</li>
 *   <li>{@link #NAMED_ENTITY_RECOGNITION} - Identify and classify named entities</li>
 *   <li>{@link #GENERAL} - General-purpose text generation</li>
 * </ul>
 */
public enum ModelRole {
    /**
     * Summarization: reduce long text to concise summaries.
     * Expected output: summarized text.
     */
    SUMMARIZATION("summarization"),

    /**
     * Classification: categorize text into predefined classes.
     * Expected output: class label or category name.
     */
    CLASSIFICATION("classification"),

    /**
     * Sorting: order items by criteria (relevance, priority, etc.).
     * Expected output: ordered list or ranked items.
     */
    SORTING("sorting"),

    /**
     * Relation extraction: extract relationships between entities.
     * Expected output: structured relations (subject, predicate, object).
     */
    RELATION_EXTRACTION("relation_extraction"),

    /**
     * Language classification: detect the language of text.
     * Expected output: language code (e.g., "en", "es", "fr").
     */
    LANGUAGE_CLASSIFICATION("language_classification"),

    /**
     * Named entity recognition: identify and classify named entities.
     * Expected output: entities with types (PERSON, ORGANIZATION, etc.).
     */
    NAMED_ENTITY_RECOGNITION("named_entity_recognition"),

    /**
     * General-purpose text generation.
     * Expected output: free-form text.
     */
    GENERAL("general");

    private final String roleName;

    ModelRole(String roleName) {
        this.roleName = roleName;
    }

    /**
     * Get the role name as a string identifier.
     *
     * @return the role name
     */
    public String getRoleName() {
        return roleName;
    }

    /**
     * Check if this role produces structured output (non-freeform text).
     *
     * @return true if the role produces structured output
     */
    public boolean isStructuredOutput() {
        switch (this) {
            case CLASSIFICATION:
            case SORTING:
            case RELATION_EXTRACTION:
            case LANGUAGE_CLASSIFICATION:
            case NAMED_ENTITY_RECOGNITION:
                return true;
            default:
                return false;
        }
    }
}
