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

package org.eclipse.deeplearning4j.llm.pipeline.resolution;

/**
 * Strategy for computing similarity between two entity mentions.
 * Different entity types (PERSON, ORGANIZATION, LOCATION, etc.) require
 * different matching semantics — name normalization for people,
 * abbreviation expansion for organizations, geocoding for locations.
 */
public interface EntityResolutionStrategy {

    /**
     * Compute a confidence score [0.0, 1.0] that two mentions refer to the same entity.
     * 0.0 = definitely different, 1.0 = definitely the same.
     *
     * @param mentionA first mention text
     * @param mentionB second mention text
     * @return confidence in [0.0, 1.0]
     */
    double computeSimilarity(String mentionA, String mentionB);

    /**
     * Normalize a mention into its canonical form for this entity type.
     * For example, "Dr. Barack H. Obama Jr." -> "barack obama" for PERSON.
     *
     * @param mention raw mention text
     * @return normalized canonical form
     */
    String normalize(String mention);

    /**
     * @return the entity type this strategy handles (e.g., "PERSON", "ORGANIZATION")
     */
    String entityType();

    /**
     * @return the default threshold above which two mentions are considered the same entity
     */
    double defaultThreshold();
}
