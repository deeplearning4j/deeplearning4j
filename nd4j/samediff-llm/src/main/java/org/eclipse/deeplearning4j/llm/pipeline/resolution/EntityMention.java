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

import lombok.Getter;
import lombok.Setter;

/**
 * A single mention of an entity as it appeared in source text.
 * Tracks the original surface form, source document, and confidence
 * that this mention belongs to its parent {@link ResolvedEntity}.
 */
@Getter
public class EntityMention {
    private final String text;
    private final String type;
    private final String sourceDocument;
    @Setter
    private double confidence;

    public EntityMention(String text, String type, String sourceDocument, double confidence) {
        this.text = text;
        this.type = type;
        this.sourceDocument = sourceDocument;
        this.confidence = confidence;
    }

    @Override
    public String toString() {
        return "Mention{" + text + " [" + type + "] conf=" + String.format("%.2f", confidence) + "}";
    }
}
