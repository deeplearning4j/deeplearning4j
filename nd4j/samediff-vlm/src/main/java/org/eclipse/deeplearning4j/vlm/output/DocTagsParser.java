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
 * <p>SmolDocling DocTags format example:</p>
 * <pre>
 * &lt;doctag&gt;&lt;page_header&gt;&lt;loc_127&gt;&lt;loc_27&gt;&lt;loc_419&gt;&lt;loc_34&gt;Header Text&lt;/page_header&gt;
 * &lt;paragraph&gt;&lt;loc_57&gt;&lt;loc_46&gt;&lt;loc_457&gt;&lt;loc_71&gt;Paragraph content...&lt;/paragraph&gt;
 * &lt;otsl&gt;&lt;loc_57&gt;&lt;loc_100&gt;&lt;loc_460&gt;&lt;loc_280&gt;&lt;ched&gt;&lt;fcel&gt;Col1&lt;fcel&gt;Col2&lt;nl&gt;
 * &lt;fcel&gt;A&lt;fcel&gt;B&lt;nl&gt;&lt;/otsl&gt;&lt;/doctag&gt;
 * </pre>
 *
 * <p>Supported element types: paragraph, page_header, page_footer, section_header_level_N,
 * caption, footnote, formula, picture, code, chart, otsl (tables), ordered_list,
 * unordered_list, list_item, reference, form, group, key_value_region.</p>
 *
 * <p>Table cells use OTSL (Optimized Table Structure Language) tokens:
 * fcel (full cell), ecel (empty cell), lcel (left-span), ucel (up-span),
 * xcel (cross-span), ched (column header), rhed (row header), nl (newline/row separator).</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class DocTagsParser {

    // Pattern to match SmolDocling elements: handles tag names with underscores, digits, and
    // nested content that may contain other tags (like location tokens or OTSL cell tokens).
    // Captures: (1) tag name, then within the body: optional 4 loc tokens, then remaining content.
    private static final Pattern ELEMENT_PATTERN = Pattern.compile(
            "<((?:section_header_level_\\d+|page_header|page_footer|paragraph|text|caption|footnote|" +
                    "formula|picture|code|chart|otsl|ordered_list|unordered_list|list_item|reference|" +
                    "form|group|key_value_region|title|table|checkbox_selected|checkbox_unselected|smiles))>" +
                    "(.*?)" +
                    "</\\1>",
            Pattern.DOTALL
    );

    // Pattern to extract 4 location tokens at the start of element body
    private static final Pattern LOC_PREFIX_PATTERN = Pattern.compile(
            "^<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>"
    );

    // Pattern to match location tokens (for stripping from content)
    private static final Pattern LOC_PATTERN = Pattern.compile("<loc_(\\d+)>");

    // OTSL tokens for table parsing
    private static final Pattern OTSL_ROW_PATTERN = Pattern.compile("<nl>");
    private static final Pattern OTSL_CELL_PATTERN = Pattern.compile(
            "<(fcel|ecel|lcel|ucel|xcel)>(.*?)(?=<(?:fcel|ecel|lcel|ucel|xcel|nl|ched|rhed|/))",
            Pattern.DOTALL
    );

    // Legacy HTML table patterns (for backward compatibility)
    private static final Pattern TR_PATTERN = Pattern.compile("<tr>(.*?)</tr>", Pattern.DOTALL);
    private static final Pattern TD_PATTERN = Pattern.compile("<td>(.*?)</td>", Pattern.DOTALL);

    /**
     * Parse DocTags output into a structured document.
     *
     * @param docTagsOutput the raw DocTags output from the model
     * @return parsed document structure
     */
    public DocumentStructure parse(String docTagsOutput) {
        List<DocumentElement> elements = new ArrayList<>();

        // Strip outer <doctag>...</doctag> wrapper if present
        String body = docTagsOutput;
        int doctStart = body.indexOf("<doctag>");
        int doctEnd = body.lastIndexOf("</doctag>");
        if (doctStart >= 0) {
            body = body.substring(doctStart + "<doctag>".length(),
                    doctEnd >= 0 ? doctEnd : body.length());
        }

        // Strip control tokens
        body = body.replace("<end_of_utterance>", "")
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "");

        Matcher matcher = ELEMENT_PATTERN.matcher(body);
        while (matcher.find()) {
            String tagType = matcher.group(1);
            String innerContent = matcher.group(2);

            // Extract bounding box from loc tokens at start of content
            BoundingBox bbox = null;
            String textContent = innerContent;
            Matcher locMatcher = LOC_PREFIX_PATTERN.matcher(innerContent);
            if (locMatcher.find()) {
                bbox = new BoundingBox(
                        Integer.parseInt(locMatcher.group(1)),
                        Integer.parseInt(locMatcher.group(2)),
                        Integer.parseInt(locMatcher.group(3)),
                        Integer.parseInt(locMatcher.group(4))
                );
                textContent = innerContent.substring(locMatcher.end());
            }

            // Strip any remaining loc tokens from content
            textContent = LOC_PATTERN.matcher(textContent).replaceAll("");

            DocumentElement element = DocumentElement.builder()
                    .tagType(tagType)
                    .content(textContent.trim())
                    .boundingBox(bbox)
                    .build();

            // Parse OTSL table content
            if ("otsl".equals(tagType) || "chart".equals(tagType)) {
                element.setChildren(parseOtslRows(textContent));
                element.setTagType("otsl");
            }
            // Legacy HTML table support
            else if ("table".equals(tagType)) {
                element.setChildren(parseTableRows(innerContent));
            }

            elements.add(element);
        }

        return DocumentStructure.builder()
                .elements(elements)
                .rawOutput(docTagsOutput)
                .build();
    }

    /**
     * Parse OTSL table rows from SmolDocling output.
     * OTSL uses &lt;nl&gt; as row separator and &lt;fcel&gt;/&lt;ecel&gt;/etc as cell markers.
     */
    private List<DocumentElement> parseOtslRows(String otslContent) {
        List<DocumentElement> rows = new ArrayList<>();

        // Split on <nl> to get rows
        String[] rowParts = OTSL_ROW_PATTERN.split(otslContent);
        for (String rowPart : rowParts) {
            if (rowPart.trim().isEmpty()) continue;

            List<DocumentElement> cells = new ArrayList<>();
            // Strip header markers for content extraction
            String cleanedRow = rowPart.replace("<ched>", "").replace("<rhed>", "");

            // Parse cells: <fcel>content, <ecel>, etc.
            Matcher cellMatcher = Pattern.compile(
                    "<(fcel|ecel|lcel|ucel|xcel)>([^<]*)",
                    Pattern.DOTALL
            ).matcher(cleanedRow);

            while (cellMatcher.find()) {
                String cellType = cellMatcher.group(1);
                String cellContent = cellMatcher.group(2).trim();
                cells.add(DocumentElement.builder()
                        .tagType(cellType)
                        .content(cellContent)
                        .build());
            }

            if (!cells.isEmpty()) {
                rows.add(DocumentElement.builder()
                        .tagType("row")
                        .children(cells)
                        .build());
            }
        }

        return rows;
    }

    /**
     * Parse legacy HTML table rows from table content.
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
     * Check if the raw DocTags output is structurally complete.
     *
     * @param rawOutput the raw model output
     * @return true if output starts with &lt;doctag&gt; and ends with &lt;/doctag&gt;
     */
    public boolean isComplete(String rawOutput) {
        if (rawOutput == null) return false;
        String trimmed = rawOutput.trim()
                .replace("<end_of_utterance>", "")
                .replace("<|im_end|>", "")
                .trim();
        return trimmed.contains("<doctag>") && trimmed.contains("</doctag>");
    }

    /**
     * Extract the plain text content from a raw DocTags string, stripping all tags.
     *
     * @param rawOutput the raw DocTags output
     * @return plain text content
     */
    public String extractPlainText(String rawOutput) {
        if (rawOutput == null) return "";
        // Remove all tags
        String text = rawOutput.replaceAll("<[^>]+>", " ");
        // Collapse whitespace
        text = text.replaceAll("\\s+", " ").trim();
        return text;
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
            String tagType = element.getTagType();

            // Handle section_header_level_N
            if (tagType.startsWith("section_header_level_")) {
                int level = 1;
                try {
                    level = Integer.parseInt(tagType.substring("section_header_level_".length()));
                } catch (NumberFormatException ignored) {}
                String prefix = "#".repeat(Math.min(level, 6));
                sb.append(prefix).append(" ").append(element.getContent()).append("\n\n");
                continue;
            }

            switch (tagType) {
                case "page_header":
                    sb.append("# ").append(element.getContent()).append("\n\n");
                    break;
                case "page_footer":
                    sb.append("---\n").append(element.getContent()).append("\n\n");
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
                case "otsl":
                case "table":
                    sb.append(tableToMarkdown(element)).append("\n\n");
                    break;
                case "code":
                    sb.append("```\n").append(element.getContent()).append("\n```\n\n");
                    break;
                case "formula":
                    sb.append("$$").append(element.getContent()).append("$$\n\n");
                    break;
                case "caption":
                    sb.append("*").append(element.getContent()).append("*\n\n");
                    break;
                case "footnote":
                    sb.append("[^]: ").append(element.getContent()).append("\n\n");
                    break;
                case "picture":
                    sb.append("[Image]\n\n");
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
     * Convert a table element (OTSL or legacy HTML) to markdown.
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
                    String cellType = cell.getTagType();
                    String content = cell.getContent();
                    // Empty/spanning cells
                    if ("ecel".equals(cellType) || "lcel".equals(cellType) ||
                            "ucel".equals(cellType) || "xcel".equals(cellType)) {
                        content = content.isEmpty() ? " " : content;
                    }
                    sb.append(" ").append(content).append(" |");
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
