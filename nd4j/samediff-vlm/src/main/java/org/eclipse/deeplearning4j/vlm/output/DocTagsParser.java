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
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parser for SmolDocling's DocTags output format.
 *
 * DocTags is a structured format for representing document elements with
 * their bounding boxes and content. The format uses XML-like tags with
 * embedded location tokens.
 *
 * <p>DocTags format example:</p>
 * <pre>
 * &lt;page_header&gt;&lt;loc_123&gt;&lt;loc_456&gt;&lt;loc_789&gt;&lt;loc_012&gt;Header Text&lt;/page_header&gt;
 * &lt;text&gt;&lt;loc_100&gt;&lt;loc_200&gt;&lt;loc_300&gt;&lt;loc_400&gt;Paragraph content...&lt;/text&gt;
 * &lt;table&gt;&lt;loc_500&gt;&lt;loc_600&gt;&lt;loc_700&gt;&lt;loc_800&gt;
 *   &lt;tr&gt;&lt;td&gt;Cell 1&lt;/td&gt;&lt;td&gt;Cell 2&lt;/td&gt;&lt;/tr&gt;
 * &lt;/table&gt;
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class DocTagsParser {

    // Pattern to match DocTags elements
    private static final Pattern ELEMENT_PATTERN = Pattern.compile(
            "<(\\w+)>(?:<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>)?([^<]*?)(?:<[^>]+>.*?)?</\\1>",
            Pattern.DOTALL
    );

    // Pattern to match location tokens
    private static final Pattern LOC_PATTERN = Pattern.compile("<loc_(\\d+)>");

    // Pattern to match table rows
    private static final Pattern TR_PATTERN = Pattern.compile("<tr>(.*?)</tr>", Pattern.DOTALL);

    // Pattern to match table cells
    private static final Pattern TD_PATTERN = Pattern.compile("<td>(.*?)</td>", Pattern.DOTALL);

    /**
     * Parse DocTags output into a structured document.
     *
     * @param docTagsOutput the raw DocTags output from the model
     * @return parsed document structure
     */
    public DocumentStructure parse(String docTagsOutput) {
        List<DocumentElement> elements = new ArrayList<>();

        Matcher matcher = ELEMENT_PATTERN.matcher(docTagsOutput);
        while (matcher.find()) {
            String tagType = matcher.group(1);
            String content = matcher.group(6);

            // Extract bounding box if present
            BoundingBox bbox = null;
            if (matcher.group(2) != null) {
                bbox = new BoundingBox(
                        Integer.parseInt(matcher.group(2)),
                        Integer.parseInt(matcher.group(3)),
                        Integer.parseInt(matcher.group(4)),
                        Integer.parseInt(matcher.group(5))
                );
            }

            DocumentElement element = DocumentElement.builder()
                    .tagType(tagType)
                    .content(content != null ? content.trim() : "")
                    .boundingBox(bbox)
                    .build();

            // Parse nested content for tables
            if ("table".equals(tagType)) {
                element.setChildren(parseTableRows(matcher.group(0)));
            }

            elements.add(element);
        }

        return DocumentStructure.builder()
                .elements(elements)
                .rawOutput(docTagsOutput)
                .build();
    }

    /**
     * Parse table rows from table content.
     */
    private List<DocumentElement> parseTableRows(String tableContent) {
        List<DocumentElement> rows = new ArrayList<>();

        Matcher rowMatcher = TR_PATTERN.matcher(tableContent);
        while (rowMatcher.find()) {
            String rowContent = rowMatcher.group(1);
            List<DocumentElement> cells = new ArrayList<>();

            Matcher cellMatcher = TD_PATTERN.matcher(rowContent);
            while (cellMatcher.find()) {
                cells.add(DocumentElement.builder()
                        .tagType("td")
                        .content(cellMatcher.group(1).trim())
                        .build());
            }

            rows.add(DocumentElement.builder()
                    .tagType("tr")
                    .children(cells)
                    .build());
        }

        return rows;
    }

    /**
     * Convert parsed document to markdown.
     *
     * @param document the parsed document
     * @return markdown string
     */
    public String toMarkdown(DocumentStructure document) {
        StringBuilder sb = new StringBuilder();

        for (DocumentElement element : document.getElements()) {
            switch (element.getTagType()) {
                case "page_header":
                case "section_header":
                    sb.append("# ").append(element.getContent()).append("\n\n");
                    break;
                case "title":
                    sb.append("## ").append(element.getContent()).append("\n\n");
                    break;
                case "text":
                case "paragraph":
                    sb.append(element.getContent()).append("\n\n");
                    break;
                case "list_item":
                    sb.append("- ").append(element.getContent()).append("\n");
                    break;
                case "table":
                    sb.append(tableToMarkdown(element)).append("\n\n");
                    break;
                case "code":
                    sb.append("```\n").append(element.getContent()).append("\n```\n\n");
                    break;
                case "caption":
                    sb.append("*").append(element.getContent()).append("*\n\n");
                    break;
                default:
                    if (!element.getContent().isEmpty()) {
                        sb.append(element.getContent()).append("\n\n");
                    }
            }
        }

        return sb.toString().trim();
    }

    /**
     * Convert a table element to markdown.
     */
    private String tableToMarkdown(DocumentElement table) {
        if (table.getChildren() == null || table.getChildren().isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        boolean isHeader = true;

        for (DocumentElement row : table.getChildren()) {
            if (row.getChildren() != null) {
                sb.append("|");
                for (DocumentElement cell : row.getChildren()) {
                    sb.append(" ").append(cell.getContent()).append(" |");
                }
                sb.append("\n");

                // Add header separator after first row
                if (isHeader) {
                    sb.append("|");
                    for (int i = 0; i < row.getChildren().size(); i++) {
                        sb.append(" --- |");
                    }
                    sb.append("\n");
                    isHeader = false;
                }
            }
        }

        return sb.toString();
    }

    /**
     * Convert parsed document to HTML.
     *
     * @param document the parsed document
     * @return HTML string
     */
    public String toHtml(DocumentStructure document) {
        StringBuilder sb = new StringBuilder();
        sb.append("<div class=\"document\">\n");

        for (DocumentElement element : document.getElements()) {
            String bboxAttr = "";
            if (element.getBoundingBox() != null) {
                BoundingBox bb = element.getBoundingBox();
                bboxAttr = String.format(" data-bbox=\"%d,%d,%d,%d\"",
                        bb.getX1(), bb.getY1(), bb.getX2(), bb.getY2());
            }

            switch (element.getTagType()) {
                case "page_header":
                case "section_header":
                    sb.append(String.format("  <h1%s>%s</h1>\n", bboxAttr, escapeHtml(element.getContent())));
                    break;
                case "title":
                    sb.append(String.format("  <h2%s>%s</h2>\n", bboxAttr, escapeHtml(element.getContent())));
                    break;
                case "text":
                case "paragraph":
                    sb.append(String.format("  <p%s>%s</p>\n", bboxAttr, escapeHtml(element.getContent())));
                    break;
                case "list_item":
                    sb.append(String.format("  <li%s>%s</li>\n", bboxAttr, escapeHtml(element.getContent())));
                    break;
                case "table":
                    sb.append(tableToHtml(element, bboxAttr));
                    break;
                default:
                    if (!element.getContent().isEmpty()) {
                        sb.append(String.format("  <div class=\"%s\"%s>%s</div>\n",
                                element.getTagType(), bboxAttr, escapeHtml(element.getContent())));
                    }
            }
        }

        sb.append("</div>");
        return sb.toString();
    }

    /**
     * Convert a table element to HTML.
     */
    private String tableToHtml(DocumentElement table, String bboxAttr) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("  <table%s>\n", bboxAttr));

        if (table.getChildren() != null) {
            for (DocumentElement row : table.getChildren()) {
                sb.append("    <tr>\n");
                if (row.getChildren() != null) {
                    for (DocumentElement cell : row.getChildren()) {
                        sb.append(String.format("      <td>%s</td>\n", escapeHtml(cell.getContent())));
                    }
                }
                sb.append("    </tr>\n");
            }
        }

        sb.append("  </table>\n");
        return sb.toString();
    }

    private String escapeHtml(String text) {
        return text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;");
    }

    /**
     * Represents a document element.
     */
    @Data
    @Builder
    public static class DocumentElement {
        private String tagType;
        private String content;
        private BoundingBox boundingBox;
        private List<DocumentElement> children;
    }
}
