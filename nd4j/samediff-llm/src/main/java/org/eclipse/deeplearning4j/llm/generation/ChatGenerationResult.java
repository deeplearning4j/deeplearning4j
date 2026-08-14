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
package org.eclipse.deeplearning4j.llm.generation;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;

import java.util.ArrayList;
import java.util.List;

/**
 * Structured result of one model-owned chat generation.
 *
 * <p>Raw protocol text is retained for replay and diagnostics while cleaned
 * content and parsed tool calls are exposed independently.</p>
 */
public final class ChatGenerationResult {
    private final String rawText;
    private final String content;
    private final String reasoningContent;
    private final List<ChatTemplate.OutputBlock> outputBlocks;
    private final List<ChatTemplate.ToolCall> toolCalls;
    private final List<String> parseErrors;

    public ChatGenerationResult(String rawText, List<ChatTemplate.Tool> tools) {
        ToolCallParser.ParseResult parsed = ToolCallParser.parse(rawText, tools);
        this.rawText = parsed.getRawText();
        this.content = parsed.getContent();
        this.reasoningContent = "";
        this.outputBlocks = List.of();
        this.toolCalls = parsed.getToolCalls();
        this.parseErrors = parsed.getErrors();
    }

    public ChatGenerationResult(String rawText, List<ChatTemplate.Tool> tools,
                                ChatTemplate.ToolCallFormat format) {
        ToolCallParser.ParseResult parsed = tools == null || tools.isEmpty()
                ? ToolCallParser.parse(rawText, List.of(), ToolCallParser.Protocol.CONTENT_ONLY)
                : ToolCallParser.parse(rawText, tools, format);
        this.rawText = parsed.getRawText();
        this.content = parsed.getContent();
        this.reasoningContent = "";
        this.outputBlocks = List.of();
        this.toolCalls = parsed.getToolCalls();
        this.parseErrors = parsed.getErrors();
    }

    public ChatGenerationResult(String rawText, String content,
                                List<ChatTemplate.ToolCall> toolCalls,
                                List<String> parseErrors) {
        this(rawText, content, "", toolCalls, parseErrors);
    }

    public ChatGenerationResult(String rawText, String content, String reasoningContent,
                                List<ChatTemplate.ToolCall> toolCalls,
                                List<String> parseErrors) {
        this(rawText, content, reasoningContent,
                reasoningContent == null || reasoningContent.isBlank()
                        ? List.of()
                        : List.of(new ChatTemplate.OutputBlock("think", reasoningContent)),
                toolCalls, parseErrors);
    }

    public ChatGenerationResult(String rawText, String content, String reasoningContent,
                                List<ChatTemplate.OutputBlock> outputBlocks,
                                List<ChatTemplate.ToolCall> toolCalls,
                                List<String> parseErrors) {
        this.rawText = rawText == null ? "" : rawText;
        this.content = content == null ? "" : content;
        this.reasoningContent = reasoningContent == null ? "" : reasoningContent;
        this.outputBlocks = outputBlocks == null ? List.of() : List.copyOf(outputBlocks);
        this.toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
        this.parseErrors = parseErrors == null ? List.of() : List.copyOf(parseErrors);
    }

    public ChatGenerationResult(String rawText, ChatTemplate.AssistantOutput normalized,
                                List<ChatTemplate.Tool> tools,
                                ChatTemplate.ToolCallFormat format,
                                ChatTemplate.ToolChoice toolChoice) {
        List<String> protocolErrors = normalized == null
                ? List.of("model output normalization failed") : normalized.getErrors();
        ToolCallParser.ParseResult parsed;
        if (!protocolErrors.isEmpty()) {
            parsed = new ToolCallParser.ParseResult(rawText, "", List.of(), protocolErrors);
        } else if (tools == null || tools.isEmpty()) {
            parsed = ToolCallParser.parse(normalized.getContent(), List.of(),
                    ToolCallParser.Protocol.CONTENT_ONLY, toolChoice);
        } else {
            parsed = ToolCallParser.parse(normalized.getContent(), tools, format, toolChoice);
        }
        List<String> errors = new ArrayList<>(protocolErrors);
        if (protocolErrors.isEmpty()) errors.addAll(parsed.getErrors());
        this.rawText = rawText == null ? "" : rawText;
        this.content = parsed.getContent();
        this.reasoningContent = normalized == null ? "" : normalized.getReasoningContent();
        this.outputBlocks = normalized == null ? List.of() : normalized.getOutputBlocks();
        this.toolCalls = parsed.getToolCalls();
        this.parseErrors = List.copyOf(errors);
    }

    public String getRawText() {
        return rawText;
    }

    public String getText() {
        return rawText;
    }

    public String getContent() {
        return content;
    }

    public String getReasoningContent() {
        return reasoningContent;
    }

    /** Every template-declared model output block in emitted closing order. */
    public List<ChatTemplate.OutputBlock> getOutputBlocks() {
        return outputBlocks;
    }

    public List<ChatTemplate.ToolCall> getToolCalls() {
        return toolCalls;
    }

    public List<String> getParseErrors() {
        return parseErrors;
    }

    public List<String> getErrors() {
        return parseErrors;
    }

    public boolean hasToolCalls() {
        return !toolCalls.isEmpty();
    }

    public boolean isParsedCleanly() {
        return parseErrors.isEmpty();
    }

    /**
     * Replay both raw protocol content and structured calls into a subsequent
     * template request. Keeping both forms is required by templates that use
     * native sentinels as well as templates that inspect tool_calls.
     */
    public ChatTemplate.Message asAssistantMessage() {
        return ChatTemplate.Message.assistantToolCalls(rawText, toolCalls);
    }
}
