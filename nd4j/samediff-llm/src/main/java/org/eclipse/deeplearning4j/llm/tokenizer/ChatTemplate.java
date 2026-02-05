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

import java.util.ArrayList;
import java.util.HashMap;
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

    private final String template;
    private final String bosToken;
    private final String eosToken;

    // Compiled pattern for simple variable substitution
    private static final Pattern VARIABLE_PATTERN = Pattern.compile("\\{\\{\\s*(\\w+)\\s*}}");

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
            return chatML(); // Default fallback
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
        if (template.contains("<|im_start|>")) {
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
     * Apply ChatML format.
     */
    private String applyChatML(List<Message> messages, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();

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

    /**
     * Represents a chat message.
     */
    @Data
    @Builder
    public static class Message {
        private final String role;
        private final String content;

        public Message(String role, String content) {
            this.role = role;
            this.content = content;
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
    }
}
