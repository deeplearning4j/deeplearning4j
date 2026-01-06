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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Result of OCR recognition containing detected text regions.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder
public class OCRResult {

    /**
     * List of detected text regions.
     */
    private final List<OCRTextRegion> regions;

    /**
     * Overall confidence score (0-1).
     */
    private final double confidence;

    /**
     * Processing time in milliseconds.
     */
    private final long processingTimeMs;

    /**
     * Original image width.
     */
    private final int imageWidth;

    /**
     * Original image height.
     */
    private final int imageHeight;

    /**
     * Engine that produced this result.
     */
    private final String engineName;

    /**
     * Get the full text concatenated from all regions.
     *
     * @return full text
     */
    public String getFullText() {
        return regions.stream()
                .sorted(Comparator.comparingInt((OCRTextRegion r) -> r.getBoundingBox().getY1())
                        .thenComparingInt((OCRTextRegion r) -> r.getBoundingBox().getX1()))
                .map(OCRTextRegion::getText)
                .collect(Collectors.joining(" "));
    }

    /**
     * Get text organized by lines (regions with similar Y coordinates).
     *
     * @param lineThreshold maximum Y difference to consider same line
     * @return list of lines, each containing regions
     */
    public List<List<OCRTextRegion>> getTextByLines(int lineThreshold) {
        if (regions.isEmpty()) {
            return new ArrayList<>();
        }

        List<OCRTextRegion> sorted = regions.stream()
                .sorted(Comparator.comparingInt((OCRTextRegion r) -> r.getBoundingBox().getY1()))
                .collect(Collectors.toList());

        List<List<OCRTextRegion>> lines = new ArrayList<>();
        List<OCRTextRegion> currentLine = new ArrayList<>();
        int currentY = sorted.get(0).getBoundingBox().getY1();

        for (OCRTextRegion region : sorted) {
            int y = region.getBoundingBox().getY1();
            if (Math.abs(y - currentY) <= lineThreshold) {
                currentLine.add(region);
            } else {
                if (!currentLine.isEmpty()) {
                    currentLine.sort(Comparator.comparingInt((OCRTextRegion r) -> r.getBoundingBox().getX1()));
                    lines.add(currentLine);
                }
                currentLine = new ArrayList<>();
                currentLine.add(region);
                currentY = y;
            }
        }

        if (!currentLine.isEmpty()) {
            currentLine.sort(Comparator.comparingInt((OCRTextRegion r) -> r.getBoundingBox().getX1()));
            lines.add(currentLine);
        }

        return lines;
    }

    /**
     * Get text organized by lines as strings.
     *
     * @param lineThreshold maximum Y difference to consider same line
     * @return list of line strings
     */
    public List<String> getLines(int lineThreshold) {
        return getTextByLines(lineThreshold).stream()
                .map(line -> line.stream()
                        .map(OCRTextRegion::getText)
                        .collect(Collectors.joining(" ")))
                .collect(Collectors.toList());
    }

    /**
     * Get text with line breaks.
     *
     * @param lineThreshold maximum Y difference to consider same line
     * @return text with line breaks
     */
    public String getTextWithLineBreaks(int lineThreshold) {
        return String.join("\n", getLines(lineThreshold));
    }

    /**
     * Get regions above a confidence threshold.
     *
     * @param minConfidence minimum confidence (0-1)
     * @return filtered regions
     */
    public List<OCRTextRegion> getHighConfidenceRegions(double minConfidence) {
        return regions.stream()
                .filter(r -> r.getConfidence() >= minConfidence)
                .collect(Collectors.toList());
    }

    /**
     * Get regions within a bounding box.
     *
     * @param box the bounding box
     * @return regions within the box
     */
    public List<OCRTextRegion> getRegionsInArea(BoundingBox box) {
        return regions.stream()
                .filter(r -> box.overlaps(r.getBoundingBox()))
                .collect(Collectors.toList());
    }

    /**
     * Get the number of detected regions.
     *
     * @return region count
     */
    public int getRegionCount() {
        return regions != null ? regions.size() : 0;
    }

    /**
     * Check if any text was detected.
     *
     * @return true if text was found
     */
    public boolean hasText() {
        return regions != null && !regions.isEmpty();
    }

    /**
     * Convert to JSON format.
     *
     * @return JSON string
     */
    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"confidence\": ").append(confidence).append(",\n");
        sb.append("  \"processingTimeMs\": ").append(processingTimeMs).append(",\n");
        sb.append("  \"imageWidth\": ").append(imageWidth).append(",\n");
        sb.append("  \"imageHeight\": ").append(imageHeight).append(",\n");
        sb.append("  \"engineName\": \"").append(engineName).append("\",\n");
        sb.append("  \"regions\": [\n");

        for (int i = 0; i < regions.size(); i++) {
            OCRTextRegion r = regions.get(i);
            sb.append("    {\n");
            sb.append("      \"text\": \"").append(escapeJson(r.getText())).append("\",\n");
            sb.append("      \"confidence\": ").append(r.getConfidence()).append(",\n");
            sb.append("      \"boundingBox\": [")
                    .append(r.getBoundingBox().getX1()).append(", ")
                    .append(r.getBoundingBox().getY1()).append(", ")
                    .append(r.getBoundingBox().getX2()).append(", ")
                    .append(r.getBoundingBox().getY2()).append("]\n");
            sb.append("    }").append(i < regions.size() - 1 ? "," : "").append("\n");
        }

        sb.append("  ]\n");
        sb.append("}");
        return sb.toString();
    }

    private String escapeJson(String text) {
        return text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
