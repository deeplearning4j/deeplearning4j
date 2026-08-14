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

package org.eclipse.deeplearning4j.llm.tokenizer;

import lombok.Builder;
import lombok.Data;
import org.eclipse.deeplearning4j.llm.config.TokenizerConfig;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Chat template processor for instruction-following models.
 *
 * Handles the conversion of conversation messages into the format expected
 * by different LLM architectures. Supports common formats like ChatML,
 * Llama-2, Vicuna, and Alpaca.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * ChatTemplate template = ChatTemplate.chatML();
 *
 * List<Message> messages = List.of(
 *     new Message("system", "You are a helpful assistant."),
 *     new Message("user", "Hello!"),
 *     new Message("assistant", "Hi there!")
 * );
 *
 * String formatted = template.apply(messages, true);
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class ChatTemplate {

    /** Canonical sentinel/function tool-call delimiters used by model templates such as LFM. */
    public static final String NATIVE_TOOL_CALL_START = "<|tool_call_start|>";
    public static final String NATIVE_TOOL_CALL_END = "<|tool_call_end|>";
    /** Canonical XML function-call delimiters declared by templates such as Qwen 3.5. */
    public static final String XML_TOOL_CALL_START = "<tool_call>";
    public static final String XML_TOOL_CALL_END = "</tool_call>";
    public static final String XML_FUNCTION_START = "<function=";
    public static final String XML_PARAMETER_START = "<parameter=";
    /** Canonical reasoning delimiters, enabled only when they occur in the imported template. */
    public static final String THINKING_START = "<think>";
    public static final String THINKING_END = "</think>";

    private final String template;
    private final String bosToken;
    private final String eosToken;

    // Compiled pattern for simple variable substitution
    private static final Pattern VARIABLE_PATTERN = Pattern.compile("\\{\\{\\s*(\\w+)\\s*}}");
    private static final ObjectMapper JSON_MAPPER = new ObjectMapper()
            .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    /**
     * Create a chat template from a Jinja2-style template string.
     *
     * @param template the template string
     * @param bosToken the beginning-of-sequence token
     * @param eosToken the end-of-sequence token
     */
    public ChatTemplate(String template, String bosToken, String eosToken) {
        this.template = template;
        this.bosToken = bosToken != null ? bosToken : "";
        this.eosToken = eosToken != null ? eosToken : "";
    }

    /**
     * Create a chat template from tokenizer configuration.
     *
     * @param config the tokenizer configuration
     * @return the chat template
     */
    public static ChatTemplate fromConfig(TokenizerConfig config) {
        if (config == null || !config.hasChatTemplate()) {
            throw new IllegalArgumentException("Tokenizer configuration does not declare a chat template");
        }
        return new ChatTemplate(
                config.getChatTemplate(),
                config.getBosToken(),
                config.getEosToken()
        );
    }

    /**
     * Apply the template to a list of messages.
     *
     * @param messages the conversation messages
     * @param addGenerationPrompt whether to add prompt for assistant response
     * @return the formatted string
     */
    public String apply(List<Message> messages, boolean addGenerationPrompt) {
        // Simple template processing for common patterns
        // Note: Full Jinja2 support would require a dedicated library

        StringBuilder result = new StringBuilder();

        // Check for common template patterns
        if (template.contains("<end_of_utterance>")) {
            // Idefics3/SmolDocling multimodal format
            return applyIdefics3(messages, addGenerationPrompt);
        } else if (template.contains("<|im_start|>")) {
            // ChatML format
            return applyChatML(messages, addGenerationPrompt);
        } else if (template.contains("[INST]")) {
            // Llama-2/Mistral format
            return applyLlama2(messages, addGenerationPrompt);
        } else if (template.contains("### ")) {
            // Alpaca format
            return applyAlpaca(messages, addGenerationPrompt);
        } else if (template.contains("USER:") || template.contains("ASSISTANT:")) {
            // Vicuna format
            return applyVicuna(messages, addGenerationPrompt);
        }

        // Generic simple processing
        return applyGeneric(messages, addGenerationPrompt);
    }

    /**
     * Apply a complete structured request. Tool definitions are made visible to
     * the template as a deterministic system-side JSON block while preserving
     * the original ordered conversation and tool-call messages.
     */
    public String apply(Request request) {
        if (request == null) {
            throw new IllegalArgumentException("Chat request must not be null");
        }
        List<Message> messages = new ArrayList<>(request.getMessages());
        if (!request.getTools().isEmpty()) {
            ToolDefinitionFormat format = request.getToolDefinitionFormat() == null
                    ? ToolDefinitionFormat.STANDARD : request.getToolDefinitionFormat();
            boolean lfmNativeProtocol = usesLfmNativeToolProtocol(request);
            String definitions = renderToolDefinitions(
                    request.getTools(), format, lfmNativeProtocol);
            int system = -1;
            for (int i = 0; i < messages.size(); i++) {
                if ("system".equals(messages.get(i).getRole())) {
                    system = i;
                    break;
                }
            }
            if (system >= 0) {
                Message original = messages.get(system);
                String systemContent = original.getContent() == null
                        ? "" : original.getContent();
                messages.set(system, new Message("system", lfmNativeProtocol
                        ? appendSection(systemContent, definitions)
                        : definitions + systemContent));
            } else {
                messages.add(0, Message.system(definitions));
            }
        }
        return apply(messages, request.isAddGenerationPrompt());
    }

    private boolean usesLfmNativeToolProtocol(Request request) {
        return request.getToolCallFormat() == ToolCallFormat.NATIVE
                || toolCallFormat() == ToolCallFormat.NATIVE;
    }

    /**
     * Resolve the tool-call envelope from the imported model template. An explicit
     * request/config value may still override this, but callers never need to inspect
     * model names or protocol sentinels themselves.
     */
    public ToolCallFormat toolCallFormat() {
        if (template.contains(NATIVE_TOOL_CALL_START)
                && template.contains(NATIVE_TOOL_CALL_END)) {
            return ToolCallFormat.NATIVE;
        }
        if (template.contains(XML_TOOL_CALL_START)
                && template.contains(XML_TOOL_CALL_END)
                && template.contains(XML_FUNCTION_START)
                && template.contains(XML_PARAMETER_START)) {
            return ToolCallFormat.XML;
        }
        return ToolCallFormat.JSON;
    }

    /**
     * Parse model-owned terminal and reasoning delimiters without leaking those
     * details into callers. Only delimiters declared by this imported template are
     * active; ordinary answer text is otherwise left untouched.
     */
    public AssistantOutput parseAssistantOutput(String rawText) {
        String raw = rawText == null ? "" : rawText;
        String value = stripTerminalToken(raw, eosToken).trim();
        List<String> errors = new ArrayList<>();
        String reasoning = "";
        if (template.contains(THINKING_START) && template.contains(THINKING_END)
                && value.startsWith(THINKING_START)) {
            int end = value.indexOf(THINKING_END, THINKING_START.length());
            if (end < 0) {
                errors.add("incomplete model reasoning block");
                return new AssistantOutput(raw, "", "", errors);
            }
            reasoning = value.substring(THINKING_START.length(), end).trim();
            value = value.substring(end + THINKING_END.length()).trim();
        }
        return new AssistantOutput(raw, value, reasoning, errors);
    }

    private static String stripTerminalToken(String value, String terminalToken) {
        if (value == null || value.isEmpty() || terminalToken == null || terminalToken.isEmpty()) {
            return value == null ? "" : value;
        }
        String result = value;
        while (result.endsWith(terminalToken)) {
            result = result.substring(0, result.length() - terminalToken.length()).stripTrailing();
        }
        return result;
    }

    /** Model-normalized assistant output with reasoning kept separate from answer content. */
    public static final class AssistantOutput {
        private final String rawText;
        private final String content;
        private final String reasoningContent;
        private final List<String> errors;

        public AssistantOutput(String rawText, String content, String reasoningContent,
                               List<String> errors) {
            this.rawText = rawText == null ? "" : rawText;
            this.content = content == null ? "" : content;
            this.reasoningContent = reasoningContent == null ? "" : reasoningContent;
            this.errors = errors == null ? List.of() : List.copyOf(errors);
        }

        public String getRawText() { return rawText; }
        public String getContent() { return content; }
        public String getReasoningContent() { return reasoningContent; }
        public List<String> getErrors() { return errors; }
    }

    /** Serialize the complete request for the native model-owned MiniJinja renderer. */
    public static String requestContextJson(Request request) {
        if (request == null) {
            throw new IllegalArgumentException("Chat request must not be null");
        }
        Map<String, Object> context = new LinkedHashMap<>(request.getTemplateArguments());
        List<Object> messages = new ArrayList<>();
        for (Message message : request.getMessages()) {
            Map<String, Object> value = new LinkedHashMap<>();
            value.put("role", message.getRole());
            value.put("content", message.getContent());
            if (!message.getToolCalls().isEmpty()) {
                List<Object> calls = new ArrayList<>();
                for (ToolCall call : message.getToolCalls()) {
                    Map<String, Object> function = new LinkedHashMap<>();
                    function.put("name", call.getName());
                    function.put("arguments", call.getArguments());
                    Map<String, Object> encoded = new LinkedHashMap<>();
                    if (call.getId() != null) encoded.put("id", call.getId());
                    encoded.put("type", "function");
                    encoded.put("function", function);
                    calls.add(encoded);
                }
                value.put("tool_calls", calls);
            }
            if (message.getToolCallId() != null) value.put("tool_call_id", message.getToolCallId());
            if (message.getToolName() != null) value.put("name", message.getToolName());
            messages.add(value);
        }
        context.put("messages", messages);

        List<Object> tools = new ArrayList<>();
        ToolDefinitionFormat definitionFormat = request.getToolDefinitionFormat() == null
                ? ToolDefinitionFormat.STANDARD : request.getToolDefinitionFormat();
        for (Tool tool : request.getTools()) {
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", tool.getName());
            function.put("description", tool.getDescription());
            function.put("parameters", tool.getParameters());
            tools.add(definitionFormat == ToolDefinitionFormat.FLAT
                    ? function : Map.of("type", "function", "function", function));
        }
        context.put("tools", tools.isEmpty() ? null : tools);
        context.put("add_generation_prompt", request.isAddGenerationPrompt());
        context.put("tool_choice", request.getToolChoice().name().toLowerCase(java.util.Locale.ROOT));
        return toJson(context);
    }

    private static String renderToolDefinitions(
            List<Tool> tools,
            ToolDefinitionFormat format,
            boolean lfmNativeProtocol) {
        List<String> definitions = new ArrayList<>(tools.size());
        for (Tool tool : tools) {
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", tool.getName());
            function.put("description", tool.getDescription());
            function.put("parameters", tool.getParameters());
            Object definition = format == ToolDefinitionFormat.FLAT
                    ? function
                    : Map.of("type", "function", "function", function);
            definitions.add(toJson(definition));
        }
        if (lfmNativeProtocol) {
            return "List of tools: [" + String.join(", ", definitions) + "]";
        }
        return "Available tools:\n" + String.join("\n", definitions) + "\n";
    }

    private static String appendSection(String first, String second) {
        if (first == null || first.isBlank()) {
            return second;
        }
        if (second == null || second.isBlank()) {
            return first;
        }
        return first + "\n" + second;
    }

    private static String toJson(Object value) {
        try {
            return JSON_MAPPER.writeValueAsString(value);
        } catch (Exception e) {
            throw new IllegalArgumentException("Tool schema is not JSON serializable", e);
        }
    }

    /**
     * Apply ChatML format.
     */
    private String applyChatML(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();
        if (template.contains("bos_token") && bosToken != null && !bosToken.isEmpty()) {
            sb.append(bosToken);
        }

        for (Message msg : messages) {
            sb.append("<|im_start|>").append(msg.getRole()).append("\n");
            sb.append(msg.getContent());
            sb.append("<|im_end|>\n");
        }

        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n");
        }

        return sb.toString();
    }

    /**
     * Apply Idefics3/SmolDocling multimodal format.
     *
     * <p>Template pattern: {@code <|im_start|>Role:content<end_of_utterance>\nAssistant:}</p>
     * <p>Roles are capitalized. Colon has no trailing space when first content is image type.</p>
     * <p>Supports multimodal messages with interleaved image and text content parts.</p>
     *
     * @see ContentPart
     */
    private String applyIdefics3(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();
        sb.append(bosToken);

        for (Message msg : messages) {
            List<ContentPart> parts = msg.resolveContentParts();

            // Role with capitalized first letter (Jinja2 capitalize filter)
            sb.append(capitalize(msg.getRole()));

            // Colon placement: no space if first content part is image, space otherwise
            if (!parts.isEmpty() && "image".equals(parts.get(0).getType())) {
                sb.append(":");
            } else {
                sb.append(": ");
            }

            // Render content parts
            for (ContentPart part : parts) {
                if ("text".equals(part.getType())) {
                    if (part.getText() != null) {
                        sb.append(part.getText());
                    }
                } else if ("image".equals(part.getType())) {
                    // Use pre-expanded image tokens if provided, otherwise single <image> placeholder
                    sb.append(part.getText() != null ? part.getText() : "<image>");
                }
            }

            sb.append("<end_of_utterance>\n");
        }

        if (addGenerationPrompt) {
            sb.append("Assistant:");
        }

        return sb.toString();
    }

    /**
     * Capitalize the first letter of a string (matches Jinja2 capitalize filter behavior).
     */
    private static String capitalize(String s) {
        if (s == null || s.isEmpty()) return s;
        return Character.toUpperCase(s.charAt(0)) + s.substring(1);
    }

    /**
     * Apply Llama-2/Mistral format.
     */
    private String applyLlama2(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();
        String systemPrompt = null;

        // Extract system message
        List<Message> nonSystemMessages = new ArrayList<>();
        for (Message msg : messages) {
            if ("system".equals(msg.getRole())) {
                systemPrompt = msg.getContent();
            } else {
                nonSystemMessages.add(msg);
            }
        }

        sb.append(bosToken);

        for (int i = 0; i < nonSystemMessages.size(); i++) {
            Message msg = nonSystemMessages.get(i);

            if ("user".equals(msg.getRole())) {
                sb.append("[INST] ");
                if (i == 0 && systemPrompt != null) {
                    sb.append("<<SYS>>\n").append(systemPrompt).append("\n<</SYS>>\n\n");
                }
                sb.append(msg.getContent()).append(" [/INST]");
            } else if ("assistant".equals(msg.getRole())) {
                sb.append(" ").append(msg.getContent()).append(eosToken);
            }
        }

        return sb.toString();
    }

    /**
     * Apply Alpaca format.
     */
    private String applyAlpaca(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();

        for (Message msg : messages) {
            String role = msg.getRole();
            if ("system".equals(role)) {
                sb.append("### System:\n").append(msg.getContent()).append("\n\n");
            } else if ("user".equals(role)) {
                sb.append("### Instruction:\n").append(msg.getContent()).append("\n\n");
            } else if ("assistant".equals(role)) {
                sb.append("### Response:\n").append(msg.getContent()).append("\n\n");
            }
        }

        if (addGenerationPrompt) {
            sb.append("### Response:\n");
        }

        return sb.toString();
    }

    /**
     * Apply Vicuna format.
     */
    private String applyVicuna(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();

        for (Message msg : messages) {
            String role = msg.getRole();
            if ("system".equals(role)) {
                sb.append(msg.getContent()).append("\n\n");
            } else if ("user".equals(role)) {
                sb.append("USER: ").append(msg.getContent()).append("\n");
            } else if ("assistant".equals(role)) {
                sb.append("ASSISTANT: ").append(msg.getContent()).append(eosToken).append("\n");
            }
        }

        if (addGenerationPrompt) {
            sb.append("ASSISTANT:");
        }

        return sb.toString();
    }

    /**
     * Apply generic template processing.
     */
    private String applyGeneric(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();
        sb.append(bosToken);

        for (Message msg : messages) {
            sb.append(msg.getRole()).append(": ").append(msg.getContent()).append("\n");
        }

        if (addGenerationPrompt) {
            sb.append("assistant: ");
        }

        return sb.toString();
    }

    /**
     * Create a single formatted message.
     *
     * @param role the message role
     * @param content the message content
     * @return the formatted message
     */
    public String applySingle(String role, String content) {
        return apply(List.of(new Message(role, content)), false);
    }

    // ========== Built-in Templates ==========

    /**
     * Create a ChatML template.
     *
     * @return ChatML template
     */
    public static ChatTemplate chatML() {
        return new ChatTemplate(
                "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
                "",
                "<|im_end|>"
        );
    }

    /**
     * Create a Llama-2 template.
     *
     * @return Llama-2 template
     */
    public static ChatTemplate llama2() {
        return new ChatTemplate(
                "{% for message in messages %}{% if message.role == 'user' %}[INST] {{ message.content }} [/INST]{% elif message.role == 'assistant' %} {{ message.content }}{% endif %}{% endfor %}",
                "<s>",
                "</s>"
        );
    }

    /**
     * Create a Vicuna template.
     *
     * @return Vicuna template
     */
    public static ChatTemplate vicuna() {
        return new ChatTemplate(
                "{% for message in messages %}{% if message.role == 'user' %}USER: {{ message.content }}\n{% elif message.role == 'assistant' %}ASSISTANT: {{ message.content }}</s>\n{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT:{% endif %}",
                "",
                "</s>"
        );
    }

    /**
     * Create an Alpaca template.
     *
     * @return Alpaca template
     */
    public static ChatTemplate alpaca() {
        return new ChatTemplate(
                "{% for message in messages %}{% if message.role == 'system' %}### System:\n{{ message.content }}\n\n{% elif message.role == 'user' %}### Instruction:\n{{ message.content }}\n\n{% elif message.role == 'assistant' %}### Response:\n{{ message.content }}\n\n{% endif %}{% endfor %}{% if add_generation_prompt %}### Response:\n{% endif %}",
                "",
                ""
        );
    }

    /** Shape used when function tools are exposed to a model-owned template. */
    public enum ToolDefinitionFormat {
        STANDARD,
        FLAT
    }

    /** Output envelope selected by a model's native chat protocol. */
    public enum ToolCallFormat {
        NATIVE,
        XML,
        JSON
    }

    /** Whether tools are optional, required, or disabled for this turn. */
    public enum ToolChoice {
        AUTO,
        REQUIRED,
        NONE
    }

    /** A portable function tool definition. */
    public static final class Tool {
        private final String name;
        private final String description;
        private final Map<String, Object> parameters;

        public Tool(String name, String description, Map<String, Object> parameters) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Tool name must not be blank");
            }
            this.name = name;
            this.description = description == null ? "" : description;
            this.parameters = parameters == null ? Map.of() : Map.copyOf(parameters);
        }

        public static Tool function(String name, String description,
                                    Map<String, Object> parameters) {
            return new Tool(name, description, parameters);
        }

        public String getName() { return name; }
        public String getDescription() { return description; }
        public Map<String, Object> getParameters() { return parameters; }
        public Map<String, Object> getArgumentsSchema() { return parameters; }
    }

    /** A parsed model tool call. */
    public static final class ToolCall {
        private final String id;
        private final String name;
        private final Map<String, Object> arguments;

        public ToolCall(String id, String name, Map<String, Object> arguments) {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Tool-call name must not be blank");
            }
            this.id = id;
            this.name = name;
            this.arguments = arguments == null ? Map.of() : Map.copyOf(arguments);
        }

        public static ToolCall function(String id, String name,
                                        Map<String, Object> arguments) {
            return new ToolCall(id, name, arguments);
        }

        public String getId() { return id; }
        public String getName() { return name; }
        public String getToolName() { return name; }
        public Map<String, Object> getArguments() { return arguments; }
        public Map<String, Object> getArgs() { return arguments; }
    }

    /** Complete runtime input to the model-owned chat template. */
    public static final class Request {
        private final List<Message> messages;
        private final List<Tool> tools;
        private final boolean addGenerationPrompt;
        private final ToolDefinitionFormat toolDefinitionFormat;
        private final ToolCallFormat toolCallFormat;
        private final ToolChoice toolChoice;
        private final Map<String, Object> templateArguments;

        private Request(Builder builder) {
            this.messages = builder.messages == null ? List.of() : List.copyOf(builder.messages);
            this.tools = builder.tools == null ? List.of() : List.copyOf(builder.tools);
            this.addGenerationPrompt = builder.addGenerationPrompt;
            this.toolDefinitionFormat = builder.toolDefinitionFormat;
            this.toolCallFormat = builder.toolCallFormat;
            this.toolChoice = builder.toolChoice == null ? ToolChoice.AUTO : builder.toolChoice;
            this.templateArguments = builder.templateArguments == null
                    ? Map.of() : Map.copyOf(builder.templateArguments);
        }

        public static Builder builder() { return new Builder(); }
        public List<Message> getMessages() { return messages; }
        public List<Tool> getTools() { return tools; }
        public boolean isAddGenerationPrompt() { return addGenerationPrompt; }
        public ToolDefinitionFormat getToolDefinitionFormat() { return toolDefinitionFormat; }
        public ToolCallFormat getToolCallFormat() { return toolCallFormat; }
        public ToolChoice getToolChoice() { return toolChoice; }
        public Map<String, Object> getTemplateArguments() { return templateArguments; }

        public static final class Builder {
            private List<Message> messages = List.of();
            private List<Tool> tools = List.of();
            private boolean addGenerationPrompt = true;
            private ToolDefinitionFormat toolDefinitionFormat;
            private ToolCallFormat toolCallFormat;
            private ToolChoice toolChoice = ToolChoice.AUTO;
            private Map<String, Object> templateArguments = Map.of();

            public Builder messages(List<Message> value) {
                this.messages = value == null ? List.of() : value;
                return this;
            }
            public Builder tools(List<Tool> value) {
                this.tools = value == null ? List.of() : value;
                return this;
            }
            public Builder addGenerationPrompt(boolean value) {
                this.addGenerationPrompt = value;
                return this;
            }
            public Builder toolDefinitionFormat(ToolDefinitionFormat value) {
                this.toolDefinitionFormat = value;
                return this;
            }
            public Builder toolCallFormat(ToolCallFormat value) {
                this.toolCallFormat = value;
                return this;
            }
            public Builder toolChoice(ToolChoice value) {
                this.toolChoice = value == null ? ToolChoice.AUTO : value;
                return this;
            }
            public Builder templateArguments(Map<String, Object> value) {
                this.templateArguments = value == null ? Map.of() : value;
                return this;
            }
            public Request build() { return new Request(this); }
        }
    }

    /**
     * Represents a content part in a multimodal message.
     * Used by Idefics3/SmolDocling templates where message content is an array
     * of typed parts (text, image, etc.).
     */
    @Data
    public static class ContentPart {
        private final String type;
        private final String text;

        public ContentPart(String type, String text) {
            this.type = type;
            this.text = text;
        }

        public static ContentPart text(String text) {
            return new ContentPart("text", text);
        }

        public static ContentPart image() {
            return new ContentPart("image", null);
        }

        public static ContentPart image(String expandedImageTokens) {
            return new ContentPart("image", expandedImageTokens);
        }
    }

    /**
     * Represents a chat message with optional multimodal content.
     *
     * <p>For text-only messages, use the two-arg constructor or factory methods
     * ({@link #user(String)}, {@link #system(String)}, {@link #assistant(String)}).
     * For multimodal messages (e.g., image + text), use the list constructor or
     * {@link #userMultimodal(List)}.</p>
     */
    @Data
    public static class Message {
        private final String role;
        private final String content;
        private final List<ContentPart> contentParts;
        private final List<ToolCall> toolCalls;
        private final String toolCallId;
        private final String toolName;

        @Builder
        public Message(String role, String content, List<ContentPart> contentParts) {
            this(role, content, contentParts, List.of(), null, null);
        }

        public Message(String role, String content, List<ContentPart> contentParts,
                       List<ToolCall> toolCalls, String toolCallId, String toolName) {
            this.role = role;
            this.content = content;
            this.contentParts = contentParts;
            this.toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
            this.toolCallId = toolCallId;
            this.toolName = toolName;
        }

        public Message(String role, String content) {
            this(role, content, null);
        }

        public Message(String role, List<ContentPart> contentParts) {
            this(role, null, contentParts);
        }

        /**
         * Resolve content as a list of parts.
         * For text-only messages, wraps the string content in a single text part.
         */
        public List<ContentPart> resolveContentParts() {
            if (contentParts != null && !contentParts.isEmpty()) return contentParts;
            if (content != null) return List.of(ContentPart.text(content));
            return List.of();
        }

        /**
         * Check if this message contains multimodal content (images).
         */
        public boolean isMultimodal() {
            if (contentParts == null) return false;
            for (ContentPart p : contentParts) {
                if ("image".equals(p.getType())) return true;
            }
            return false;
        }

        /**
         * Create a system message.
         */
        public static Message system(String content) {
            return new Message("system", content);
        }

        /**
         * Create a user message.
         */
        public static Message user(String content) {
            return new Message("user", content);
        }

        /**
         * Create an assistant message.
         */
        public static Message assistant(String content) {
            return new Message("assistant", content);
        }

        public static Message assistantToolCalls(String rawContent, List<ToolCall> calls) {
            return new Message("assistant", rawContent, null,
                    calls == null ? List.of() : calls, null, null);
        }

        public static Message toolResult(String toolCallId, String toolName, String content) {
            return new Message("tool", content, null, List.of(), toolCallId, toolName);
        }

        public List<ToolCall> getToolCalls() {
            return toolCalls;
        }

        public String getToolCallId() {
            return toolCallId;
        }

        public String getToolName() {
            return toolName;
        }

        /**
         * Create a multimodal user message with image and text content parts.
         */
        public static Message userMultimodal(List<ContentPart> parts) {
            return new Message("user", parts);
        }
    }
}
