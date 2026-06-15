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

package org.eclipse.deeplearning4j.pipeline;

import lombok.Builder;
import lombok.Data;

import java.util.*;
import java.util.regex.*;

/**
 * Handles chat template formatting for LLMs.
 * Provides a simplified implementation of Jinja2-style chat templates.
 */
@Data
@Builder
public class ChatTemplate {

    private String template;
    private String bosToken;
    private String eosToken;
    private boolean addGenerationPrompt;

    // Common chat template patterns
    public static final String LLAMA_STYLE = "{% for message in messages %}{% if message['role'] == 'system' %}<<SYS>>{{ message['content'] }}<</SYS>>{% elif message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}";
    public static final String CHATML_STYLE = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    public static final String ZEPHYR_STYLE = "{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}</s>\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}</s>\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}</s>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";

    @Data
    @Builder
    public static class Message {
        private String role;
        private String content;

        public static Message system(String content) {
            return Message.builder().role("system").content(content).build();
        }

        public static Message user(String content) {
            return Message.builder().role("user").content(content).build();
        }

        public static Message assistant(String content) {
            return Message.builder().role("assistant").content(content).build();
        }
    }

    /**
     * Apply the chat template to a list of messages.
     * This is a simplified implementation that handles common patterns.
     */
    public String apply(List<Message> messages) {
        return apply(messages, addGenerationPrompt);
    }

    /**
     * Apply the chat template to a list of messages.
     */
    public String apply(List<Message> messages, boolean addGenPrompt) {
        if (template == null || template.isEmpty()) {
            return formatSimple(messages, addGenPrompt);
        }

        // Simple Jinja2-like template processing
        StringBuilder result = new StringBuilder();

        // Handle BOS token
        if (bosToken != null && template.contains("bos_token")) {
            result.append(bosToken);
        }

        // Process messages with for loop
        for (Message message : messages) {
            String messageOutput = processMessageTemplate(template, message);
            result.append(messageOutput);
        }

        // Handle generation prompt
        if (addGenPrompt) {
            String genPrompt = extractGenerationPrompt(template);
            if (genPrompt != null) {
                result.append(genPrompt);
            }
        }

        return result.toString();
    }

    private String processMessageTemplate(String template, Message message) {
        // Extract the content between {% for message in messages %} and {% endfor %}
        Pattern forPattern = Pattern.compile("\\{%\\s*for\\s+message\\s+in\\s+messages\\s*%\\}(.+?)\\{%\\s*endfor\\s*%\\}", Pattern.DOTALL);
        Matcher forMatcher = forPattern.matcher(template);

        if (!forMatcher.find()) {
            return formatMessageSimple(message);
        }

        String loopBody = forMatcher.group(1);
        String role = message.getRole();
        String content = message.getContent();

        // Process conditionals for this message's role
        StringBuilder result = new StringBuilder();
        String[] parts = loopBody.split("\\{%\\s*(?:el)?if\\s+");

        for (String part : parts) {
            if (part.isEmpty()) continue;

            // Check if this part matches the current role
            if (part.contains("message['role'] == '" + role + "'") ||
                    part.contains("message[\"role\"] == \"" + role + "\"")) {
                // Extract content after the condition
                int endIdx = part.indexOf("%}");
                if (endIdx >= 0) {
                    String afterCondition = part.substring(endIdx + 2);
                    // Remove trailing endif/else
                    afterCondition = afterCondition.replaceAll("\\{%\\s*(endif|else|elif).*", "");
                    // Replace message content placeholders
                    afterCondition = afterCondition.replace("{{ message['content'] }}", content);
                    afterCondition = afterCondition.replace("{{ message[\"content\"] }}", content);
                    afterCondition = afterCondition.replace("{{ message.content }}", content);
                    result.append(afterCondition);
                    break;
                }
            }
        }

        return result.toString();
    }

    private String extractGenerationPrompt(String template) {
        // Look for add_generation_prompt block
        Pattern genPattern = Pattern.compile("\\{%\\s*if\\s+add_generation_prompt\\s*%\\}(.+?)\\{%\\s*endif\\s*%\\}", Pattern.DOTALL);
        Matcher matcher = genPattern.matcher(template);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return null;
    }

    /**
     * Simple fallback formatting without a template.
     */
    private String formatSimple(List<Message> messages, boolean addGenPrompt) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            sb.append(formatMessageSimple(msg));
        }
        if (addGenPrompt) {
            sb.append("Assistant: ");
        }
        return sb.toString();
    }

    private String formatMessageSimple(Message msg) {
        String role = msg.getRole();
        String content = msg.getContent();
        if ("system".equals(role)) {
            return "System: " + content + "\n\n";
        } else if ("user".equals(role)) {
            return "User: " + content + "\n\n";
        } else if ("assistant".equals(role)) {
            return "Assistant: " + content + "\n\n";
        }
        return content + "\n\n";
    }

    /**
     * Create a ChatTemplate from TokenizerConfig.
     */
    public static ChatTemplate fromTokenizerConfig(TokenizerConfig config) {
        return ChatTemplate.builder()
                .template(config.getChatTemplate())
                .bosToken(config.getBosToken())
                .eosToken(config.getEosToken())
                .addGenerationPrompt(true)
                .build();
    }

    /**
     * Create a ChatTemplate from TokenizerConfig and SpecialTokensMap.
     */
    public static ChatTemplate fromConfigs(TokenizerConfig tokenizerConfig, SpecialTokensMap specialTokens) {
        ChatTemplateBuilder builder = ChatTemplate.builder()
                .addGenerationPrompt(true);

        if (tokenizerConfig != null) {
            builder.template(tokenizerConfig.getChatTemplate());
            if (tokenizerConfig.getBosToken() != null) {
                builder.bosToken(tokenizerConfig.getBosToken());
            }
            if (tokenizerConfig.getEosToken() != null) {
                builder.eosToken(tokenizerConfig.getEosToken());
            }
        }

        if (specialTokens != null) {
            if (builder.build().getBosToken() == null && specialTokens.getBosTokenString() != null) {
                builder.bosToken(specialTokens.getBosTokenString());
            }
            if (builder.build().getEosToken() == null && specialTokens.getEosTokenString() != null) {
                builder.eosToken(specialTokens.getEosTokenString());
            }
        }

        return builder.build();
    }

    /**
     * Create a default ChatML-style template.
     */
    public static ChatTemplate chatMl() {
        return ChatTemplate.builder()
                .template(CHATML_STYLE)
                .bosToken("<|im_start|>")
                .eosToken("<|im_end|>")
                .addGenerationPrompt(true)
                .build();
    }

    /**
     * Create a default Llama-style template.
     */
    public static ChatTemplate llama() {
        return ChatTemplate.builder()
                .template(LLAMA_STYLE)
                .bosToken("<s>")
                .eosToken("</s>")
                .addGenerationPrompt(true)
                .build();
    }
}
