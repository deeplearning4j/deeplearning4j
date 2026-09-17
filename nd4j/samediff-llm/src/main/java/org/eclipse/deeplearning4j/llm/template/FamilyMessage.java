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
 *  *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  License for the specific language governing permissions and limitations
 *  *  under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.template;

import java.util.ArrayList;
import java.util.List;

/**
 * One conversation message for architecture-family chat templates.
 *
 * <p>Unlike {@link org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.Message},
 * this type carries the template-level fields the official per-family templates
 * actually consume: {@code reasoningContent} (Qwen {@code message.reasoning_content},
 * GLM {@code m.reasoning_content}, DeepSeek {@code reasoning_content}) and the
 * DeepSeek {@code tools} / {@code toolCalls} attachments.</p>
 *
 * <p>Instances are immutable and safe to share across threads.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class FamilyMessage {

    /** Roles understood by the supported template families. */
    public static final String ROLE_SYSTEM = "system";
    public static final String ROLE_USER = "user";
    public static final String ROLE_ASSISTANT = "assistant";
    public static final String ROLE_TOOL = "tool";
    public static final String ROLE_LATEST_REMINDER = "latest_reminder";

    private final String role;
    private final String content;
    private final String reasoningContent;
    private final String toolsJson;
    private final List<ToolCall> toolCalls;

    private FamilyMessage(Builder builder) {
        if (builder.role == null || builder.role.isBlank()) {
            throw new IllegalArgumentException("Message role must not be blank");
        }
        this.role = builder.role;
        this.content = builder.content == null ? "" : builder.content;
        this.reasoningContent = builder.reasoningContent == null ? "" : builder.reasoningContent;
        this.toolsJson = builder.toolsJson == null ? "" : builder.toolsJson;
        this.toolCalls = builder.toolCalls == null
                ? List.of() : List.copyOf(builder.toolCalls);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static FamilyMessage system(String content) {
        return builder().role(ROLE_SYSTEM).content(content).build();
    }

    public static FamilyMessage user(String content) {
        return builder().role(ROLE_USER).content(content).build();
    }

    public static FamilyMessage assistant(String content) {
        return builder().role(ROLE_ASSISTANT).content(content).build();
    }

    /** Assistant message with separate reasoning content (Qwen/GLM/DeepSeek reasoning_content). */
    public static FamilyMessage assistantThinking(String content, String reasoningContent) {
        return builder().role(ROLE_ASSISTANT).content(content)
                .reasoningContent(reasoningContent).build();
    }

    /** DeepSeek-only role for date/locale reminders; rendered after the user turn. */
    public static FamilyMessage latestReminder(String content) {
        return builder().role(ROLE_LATEST_REMINDER).content(content).build();
    }

    /** DeepSeek tool message; merged into the preceding user turn as a tool_result block. */
    public static FamilyMessage toolResult(String toolCallId, String content) {
        return builder().role(ROLE_TOOL).toolCallId(toolCallId).content(content).build();
    }

    public String getRole() {
        return role;
    }

    /** Visible answer text (never includes reasoning). Empty string, never null. */
    public String getContent() {
        return content;
    }

    /** Reasoning text for assistant messages. Empty string, never null. */
    public String getReasoningContent() {
        return reasoningContent;
    }

    /** DeepSeek: JSON array of OpenAI-format tool definitions attached to a system message. */
    public String getToolsJson() {
        return toolsJson;
    }

    /** DeepSeek tool calls attached to an assistant message. */
    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

    public boolean isToolCall() {
        return role.equals(ROLE_TOOL);
    }

    @Override
    public String toString() {
        return role + ": " + content;
    }

    /** One assistant tool call, in OpenAI wire format. */
    public static final class ToolCall {
        private final String id;
        private final String name;
        private final String argumentsJson;

        public ToolCall(String id, String name, String argumentsJson) {
            this.id = id == null ? "" : id;
            this.name = name == null ? "" : name;
            this.argumentsJson = argumentsJson == null ? "" : argumentsJson;
        }

        public String getId() {
            return id;
        }

        public String getName() {
            return name;
        }

        public String getArgumentsJson() {
            return argumentsJson;
        }
    }

    /** Builder for {@link FamilyMessage}. */
    public static final class Builder {
        private String role;
        private String content;
        private String reasoningContent;
        private String toolsJson;
        private String toolCallId;
        private List<ToolCall> toolCalls;

        public Builder role(String value) {
            this.role = value;
            return this;
        }

        public Builder content(String value) {
            this.content = value;
            return this;
        }

        public Builder reasoningContent(String value) {
            this.reasoningContent = value;
            return this;
        }

        public Builder toolsJson(String value) {
            this.toolsJson = value;
            return this;
        }

        public Builder toolCallId(String value) {
            this.toolCallId = value;
            return this;
        }

        public Builder toolCalls(List<ToolCall> value) {
            this.toolCalls = value;
            return this;
        }

        public Builder addToolCall(ToolCall value) {
            if (this.toolCalls == null) {
                this.toolCalls = new ArrayList<>();
            }
            this.toolCalls.add(value);
            return this;
        }

        public FamilyMessage build() {
            if (ROLE_TOOL.equals(role) && toolCallId != null) {
                if (toolCalls == null) {
                    toolCalls = List.of();
                }
            }
            return new FamilyMessage(this);
        }
    }
}
