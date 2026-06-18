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

package org.eclipse.deeplearning4j.vlm.output;

import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Represents a parsed document structure from VLM output.
 *
 * Adam Gibson
 */
@Data
@Builder
public class DocumentStructure {

    private final List<DocTagsParser.DocumentElement> elements;
    private final String rawOutput;

    /**
     * Get all text content concatenated.
     *
     * @return the full text
     */
    public String getFullText() {
        return elements.stream()
                .map(DocTagsParser.DocumentElement::getContent)
                .filter(c -> c != null && !c.isEmpty())
                .collect(Collectors.joining(" "));
    }

    /**
     * Get elements of a specific type.
     *
     * @param tagType the tag type to filter
     * @return list of matching elements
     */
    public List<DocTagsParser.DocumentElement> getElementsByType(String tagType) {
        return elements.stream()
                .filter(e -> tagType.equals(e.getTagType()))
                .collect(Collectors.toList());
    }

    /**
     * Get all tables in the document.
     *
     * @return list of table elements
     */
    public List<DocTagsParser.DocumentElement> getTables() {
        return getElementsByType("table");
    }

    /**
     * Get all headers in the document.
     *
     * @return list of header elements
     */
    public List<DocTagsParser.DocumentElement> getHeaders() {
        return elements.stream()
                .filter(e -> e.getTagType().contains("header") || "title".equals(e.getTagType()))
                .collect(Collectors.toList());
    }

    /**
     * Get the number of elements.
     *
     * @return element count
     */
    public int getElementCount() {
        return elements != null ? elements.size() : 0;
    }
}
