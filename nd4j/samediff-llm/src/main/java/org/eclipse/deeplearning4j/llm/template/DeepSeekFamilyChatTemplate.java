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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * DeepSeek-V4.1 (deepseek-ai/DeepSeek-V4.1-Flash) prompt encoder.
 *
 * <p>DeepSeek ships no Jinja chat template: the official reference is the Python
 * {@code encoding/encoding.py} in the model repo (see the source URL recorded in
 * {@code DeepSeekFamilyChatTemplateGoldenTest}). This class ports the exact behavior:</p>
 * <ul>
 *   <li>Special tokens are byte-exact, including the full-width vertical bar U+FF5C
 *       {@code ｜} and the lower-one-eighth-block U+2581 {@code ▁}:
 *       {@code <｜begin▁of▁sentence｜>}, {@code <｜end▁of▁sentence｜>},
 *       {@code <｜User｜>}, {@code <｜Assistant｜>}, {@code <｜System｜>}.</li>
 *   <li>BOS is emitted once at position 0.</li>
 *   <li>Thinking mode: assistant turns render
 *       {@code <think>reasoning</think>} before content and close with EOS. Chat mode
 *       ({@code enableThinking=false}) closes the block immediately:
 *       {@code <｜Assistant｜></think>}.</li>
 *   <li>{@code reasoning_effort} is a numeric budget 1-100 (aliases low/high/max map to
 *       50/75/100). The prefix
 *       {@code Reasoning Effort: N (range 1-100, the higher the value, the more thorough the reasoning)\n\n}
 *       renders only in thinking mode and only at conversation index 0. Default is the
 *       official default {@code high} (75). It always leads with the
 *       {@code <｜System｜>} token.</li>
 *   <li>History thinking is dropped before the last user (or mid-conversation system)
 *       turn; the final assistant turn keeps its block. With {@code dropThinking=false}
 *       all thinking is retained and generation opens {@code <｜Assistant｜><think>}.</li>
 *   <li>Tool results merge into the preceding user turn as
 *       {@code <tool_result>content</tool_result>}; consecutive tool results accumulate
 *       into the same user turn.</li>
 *   <li>V4.1 DSML tool-call tag names carry a leading space:
 *       {@code <｜DSML｜ calls>} blocks with {@code <｜DSML｜ invoke>} and
 *       {@code <｜DSML｜ parameter>} tags (note the space before {@code calls},
 *       {@code invoke}, and {@code parameter}).</li>
 * </ul>
 * The {@code <｜Assistant｜>} header attaches to the END of the preceding user or
 * mid-conversation system turn (exactly like the official transition tokens); every
 * assistant turn ends with EOS.
 *
 * <p>Pure string transformation: no I/O, no tokenizer, no state.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class DeepSeekFamilyChatTemplate implements FamilyChatTemplate {

    /** Byte-exact full-width special tokens (U+FF5C bars, U+2581 inner underscore). */
    public static final String BOS = "<｜begin▁of▁sentence｜>";
    public static final String EOS = "<｜end▁of▁sentence｜>";
    public static final String USER_TOKEN = "<｜User｜>";
    public static final String ASSISTANT_TOKEN = "<｜Assistant｜>";
    public static final String SYSTEM_TOKEN = "<｜System｜>";
    public static final String LATEST_REMINDER_TOKEN = "<｜latest▁reminder｜>";
    public static final String THINK_START = "<think>";
    public static final String THINK_END = "</think>";
    public static final String TOOL_RESULT_OPEN = "<tool_result>";
    public static final String TOOL_RESULT_CLOSE = "</tool_result>";
    public static final String IMAGE_PLACEHOLDER = "<｜deepseek_image｜>";

    /** Official alias table: low -> 50, high -> 75, max -> 100. */
    private static final Map<String, Integer> EFFORT_ALIASES = new LinkedHashMap<>();
    static {
        EFFORT_ALIASES.put("low", 50);
        EFFORT_ALIASES.put("high", 75);
        EFFORT_ALIASES.put("max", 100);
    }

    /** Official default is "high" (75). */
    public static final int DEFAULT_REASONING_EFFORT = 75;

    private static final String EFFORT_TEMPLATE =
            "Reasoning Effort: {0} (range 1-100, the higher the value, the more thorough the reasoning)\n\n";

    private static final DeepSeekFamilyChatTemplate INSTANCE = new DeepSeekFamilyChatTemplate();

    public static DeepSeekFamilyChatTemplate instance() {
        return INSTANCE;
    }

    @Override
    public Family family() {
        return Family.DEEPSEEK;
    }

    /** Map an official alias ("low"/"high"/"max") to its numeric budget. */
    public static int effortAlias(String alias) {
        Integer mapped = EFFORT_ALIASES.get(alias);
        if (mapped == null) {
            throw new IllegalArgumentException(
                    "Invalid reasoning effort for deepseek_v41: " + alias
                            + ", should be int within [1,100] or [low, high, max]");
        }
        return mapped;
    }

    /**
     * Render the exact official effort prefix for a budget, or the empty string when
     * the prefix is not applicable at this position.
     */
    public static String renderEffortPrefix(int effort, boolean thinkingMode, boolean firstMessage) {
        if (effort < 1 || effort > 100) {
            throw new IllegalArgumentException(
                    "Invalid reasoning effort for deepseek_v41: " + effort
                            + ", should be int within [1,100] or [low, high, max]");
        }
        if (firstMessage && thinkingMode) {
            return EFFORT_TEMPLATE.replace("{0}", Integer.toString(effort));
        }
        return "";
    }

    @Override
    public String apply(List<FamilyMessage> messages, FamilyChatOptions options) {
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException(
                    "DeepSeek chat template requires at least one message");
        }
        boolean thinkingMode = options.isEnableThinking();
        boolean dropThinking = options.isDeepseekDropThinking();

        // The official encoder merges tool messages into the preceding user turn
        // (merge_tool_messages) before rendering.
        List<FamilyMessage> merged = mergeToolMessages(messages);

        int lastUserIndex = lastUserIndex(merged);
        StringBuilder sb = new StringBuilder();
        if (options.isAddBosToken()) {
            sb.append(BOS);
        }

        for (int index = 0; index < merged.size(); index++) {
            FamilyMessage message = merged.get(index);
            String role = message.getRole();
            boolean first = index == 0;

            // System-token lead when the first message is a system message, or when the
            // effort prefix renders (thinking mode, index 0).
            String effortPrefix = renderEffortPrefix(
                    resolveEffort(options), thinkingMode, first);
            if (first && (FamilyMessage.ROLE_SYSTEM.equals(role) || !effortPrefix.isEmpty())) {
                sb.append(SYSTEM_TOKEN);
            }
            sb.append(effortPrefix);

            if (FamilyMessage.ROLE_SYSTEM.equals(role)) {
                if (!first) {
                    sb.append(SYSTEM_TOKEN);
                }
                sb.append(message.getContent());
            } else if (FamilyMessage.ROLE_USER.equals(role)) {
                sb.append(USER_TOKEN).append(message.getContent());
            } else if (FamilyMessage.ROLE_LATEST_REMINDER.equals(role)) {
                sb.append(LATEST_REMINDER_TOKEN).append(message.getContent());
            } else if (FamilyMessage.ROLE_ASSISTANT.equals(role)) {
                appendAssistant(sb, message, thinkingMode, dropThinking,
                        index > lastUserIndex);
            } else {
                throw new IllegalArgumentException(
                        "DeepSeek chat template does not support role: " + role);
            }

            // Official transition tokens (render_message tail): a user turn ends with
            // '<｜Assistant｜>' + think token exactly when the next message is an
            // assistant or latest_reminder turn, or when it is the final turn of a
            // generation request. Any other continuation returns early with no header.
            boolean hasNext = index + 1 < merged.size();
            String nextRole = hasNext ? merged.get(index + 1).getRole() : null;
            boolean appendHeader = !hasNext
                    || FamilyMessage.ROLE_ASSISTANT.equals(nextRole)
                    || FamilyMessage.ROLE_LATEST_REMINDER.equals(nextRole);
            boolean triggersHeader = FamilyMessage.ROLE_USER.equals(role)
                    || (FamilyMessage.ROLE_SYSTEM.equals(role) && !first);
            if (appendHeader && triggersHeader) {
                sb.append(ASSISTANT_TOKEN);
                if (thinkingMode && (!dropThinking || index >= lastUserIndex)) {
                    sb.append(THINK_START);
                } else {
                    sb.append(THINK_END);
                }
            }
        }
        return sb.toString();
    }

    private static void appendAssistant(StringBuilder sb, FamilyMessage message,
                                        boolean thinkingMode, boolean dropThinking,
                                        boolean afterLastUser) {
        String reasoning = message.getReasoningContent() == null
                ? "" : message.getReasoningContent();
        String content = message.getContent() == null ? "" : message.getContent();
        String toolCalls = renderToolCalls(message);

        // Official thinking_part is ALWAYS "{reasoning_content}</think>" when kept and
        // "" when dropped: the opening <think> comes solely from the preceding user or
        // mid-conversation system turn's <｜Assistant｜> header, never from this branch.
        if (thinkingMode && (!dropThinking || afterLastUser)) {
            sb.append(reasoning).append(THINK_END);
        }
        sb.append(content).append(toolCalls).append(EOS);
    }

    /** Official tool-call envelope: "\n\n<｜DSML｜ calls>..." (leading-space tag names). */
    private static String renderToolCalls(FamilyMessage message) {
        if (message.getToolCalls().isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder("\n\n<｜DSML｜ calls>\n");
        for (int i = 0; i < message.getToolCalls().size(); i++) {
            FamilyMessage.ToolCall call = message.getToolCalls().get(i);
            if (i > 0) {
                sb.append('\n');
            }
            sb.append("<｜DSML｜ invoke name=\"").append(call.getName()).append("\">\n");
            sb.append(encodeArgumentsToDsml(call.getArgumentsJson()));
            sb.append("\n</｜DSML｜ invoke>");
        }
        sb.append("\n</｜DSML｜ calls>");
        return sb.toString();
    }

    /** Shared mapper: nd4j shade Jackson, same as the rest of samediff-llm. */
    private static final ObjectMapper JSON_MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    private static String encodeArgumentsToDsml(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isEmpty()) {
            return "";
        }
        Map<?, ?> map;
        try {
            map = JSON_MAPPER.readValue(argumentsJson, Map.class);
        } catch (Exception e) {
            // Official fallback wraps a non-object argument in {"arguments": ...}.
            return "<｜DSML｜ parameter name=\"arguments\" string=\"true\">"
                    + argumentsJson + "</｜DSML｜ parameter>";
        }
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!first) {
                sb.append('\n');
            }
            first = false;
            boolean isString = entry.getValue() instanceof String;
            sb.append("<｜DSML｜ parameter name=\"").append(entry.getKey())
                    .append("\" string=\"").append(isString ? "true" : "false").append("\">");
            sb.append(isString ? (String) entry.getValue() : writeJson(entry.getValue()));
            sb.append("</｜DSML｜ parameter>");
        }
        return sb.toString();
    }

    private static String writeJson(Object value) {
        try {
            return JSON_MAPPER.writeValueAsString(value);
        } catch (Exception e) {
            throw new IllegalArgumentException("Tool argument is not JSON serializable", e);
        }
    }

    /**
     * Official {@code merge_tool_messages}: a tool message becomes a
     * {@code <tool_result>} block merged into the preceding user turn; consecutive
     * user turns (including tool-result-only ones) merge into one user turn. Rendered
     * blocks are joined with a blank line, matching the official
     * {@code "\n\n".join(parts)} content-block rendering.
     */
    private static List<FamilyMessage> mergeToolMessages(List<FamilyMessage> messages) {
        List<FamilyMessage> merged = new ArrayList<>();
        for (FamilyMessage message : messages) {
            String role = message.getRole();
            boolean previousIsUser = !merged.isEmpty()
                    && FamilyMessage.ROLE_USER.equals(merged.get(merged.size() - 1).getRole());
            if (FamilyMessage.ROLE_TOOL.equals(role)) {
                String block = TOOL_RESULT_OPEN + message.getContent() + TOOL_RESULT_CLOSE;
                if (previousIsUser) {
                    FamilyMessage previous = merged.remove(merged.size() - 1);
                    merged.add(FamilyMessage.builder()
                            .role(FamilyMessage.ROLE_USER)
                            .content(previous.getContent() + "\n\n" + block)
                            .build());
                } else {
                    merged.add(FamilyMessage.builder()
                            .role(FamilyMessage.ROLE_USER)
                            .content(block)
                            .build());
                }
            } else if (FamilyMessage.ROLE_USER.equals(role) && previousIsUser) {
                FamilyMessage previous = merged.remove(merged.size() - 1);
                merged.add(FamilyMessage.builder()
                        .role(FamilyMessage.ROLE_USER)
                        .content(previous.getContent() + "\n\n" + message.getContent())
                        .build());
            } else {
                merged.add(message);
            }
        }
        return merged;
    }

    /** Official definition: last user turn, or a mid-conversation system turn. */
    private static int lastUserIndex(List<FamilyMessage> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            String role = messages.get(i).getRole();
            if (FamilyMessage.ROLE_USER.equals(role)
                    || (FamilyMessage.ROLE_SYSTEM.equals(role) && i > 0)) {
                return i;
            }
        }
        return -1;
    }

    private static int resolveEffort(FamilyChatOptions options) {
        Integer effort = options.getDeepseekReasoningEffort();
        return effort == null ? DEFAULT_REASONING_EFFORT : effort;
    }

    /** Disallow external construction; use {@link #instance()}. */
    private DeepSeekFamilyChatTemplate() {
    }
}
