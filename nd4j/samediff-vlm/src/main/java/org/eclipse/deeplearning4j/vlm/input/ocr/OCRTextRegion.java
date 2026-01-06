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

package org.eclipse.deeplearning4j.vlm.input.ocr;

import lombok.Builder;
import lombok.Data;
import org.eclipse.deeplearning4j.vlm.output.BoundingBox;

/**
 * Represents a detected text region from OCR.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
public class OCRTextRegion {

    /**
     * The detected text content.
     */
    private final String text;

    /**
     * Confidence score (0-1).
     */
    private final double confidence;

    /**
     * Bounding box of the text region.
     */
    private final BoundingBox boundingBox;

    /**
     * Language of the detected text (if available).
     */
    private final String language;

    /**
     * Text direction (ltr, rtl, ttb).
     */
    private final String direction;

    /**
     * Font size estimate (if available).
     */
    private final int fontSize;

    /**
     * Whether this is likely a word or a line.
     */
    private final TextLevel level;

    /**
     * Create a simple text region with just text and bounding box.
     *
     * @param text the text
     * @param boundingBox the bounding box
     * @return text region
     */
    public static OCRTextRegion of(String text, BoundingBox boundingBox) {
        return OCRTextRegion.builder()
                .text(text)
                .boundingBox(boundingBox)
                .confidence(1.0)
                .level(TextLevel.WORD)
                .build();
    }

    /**
     * Create a text region with confidence.
     *
     * @param text the text
     * @param boundingBox the bounding box
     * @param confidence the confidence score
     * @return text region
     */
    public static OCRTextRegion of(String text, BoundingBox boundingBox, double confidence) {
        return OCRTextRegion.builder()
                .text(text)
                .boundingBox(boundingBox)
                .confidence(confidence)
                .level(TextLevel.WORD)
                .build();
    }

    /**
     * Text level enumeration.
     */
    public enum TextLevel {
        CHARACTER,
        WORD,
        LINE,
        PARAGRAPH,
        BLOCK
    }
}
