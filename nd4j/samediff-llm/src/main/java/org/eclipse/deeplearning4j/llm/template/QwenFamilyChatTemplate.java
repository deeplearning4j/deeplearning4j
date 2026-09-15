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

import java.util.List;

/**
 * Qwen3.5 / Qwen3.8 (incl. Flash-Next) ChatML chat template.
 *
 * <p>Mirrors the official HuggingFace {@code chat_template} of
 * {@code Qwen/Qwen3.8-Flash-Next} and {@code Qwen/Qwen3.5-27B} (see the source URLs in
 * {@code QwenFamilyChatTemplateGoldenTest}). Rendering rules implemented verbatim:</p>
 * <ul>
 *   <li>ChatML envelope: {@code <|im_start|>role\ncontent<|im_end|>\n}; system turn is
 *       emitted only when present (or when the 3.8 reasoning directives need one).</li>
 *   <li>3.8 {@code reasoning_effort} knob ({@code xhigh} default, {@code medium},
 *       {@code low}): {@code xhigh} and {@code low} prepend the official English
 *       directive to the system content (creating a synthetic system turn when the
 *       conversation has none); {@code medium} emits nothing. The exact directive
 *       strings are copied byte-for-byte from the official template.</li>
 *   <li>Thinking blocks: an assistant turn carries {@code <think>\n<reasoning>\n</think>\n\n}
 *       before its content when reasoning is available and preserved. 3.8 keeps thinking
 *       on the final turn (after the last user query) always; earlier history turns keep
 *       it only when the {@code preserve_thinking} knob is true.</li>
 *   <li>3.5 legacy behavior: inline {@code <think>...</think>} inside assistant content
 *       is split out automatically, and history turns before the last user query drop
 *       the think block (matching the official 3.5 template's {@code last_query_index}
 *       rule).</li>
 *   <li>Generation prefix: {@code <|im_start|>assistant\n<think>\n} with thinking on,
 *       or {@code <|im_start|>assistant\n<think>\n\n</think>\n\n} with thinking off.</li>
 * </ul>
 *
 * <p>Pure string transformation: no I/O, no tokenizer, no state.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class QwenFamilyChatTemplate implements FamilyChatTemplate {

    public static final String IM_START = "<|im_start|>";
    public static final String IM_END = "<|im_end|>";
    public static final String THINK_START = "<think>";
    public static final String THINK_END = "</think>";

    /** Official 3.8 xhigh directive, byte-exact from chat_template.jinja. */
    public static final String XHIGH_DIRECTIVE =
            "Reasoning effort is set to xhigh. Please think carefully through the task, "
                    + "validate key assumptions, consider plausible alternatives, and "
                    + "prioritize correctness, consistency, and clarity in the final answer.";

    /** Official 3.8 low directive, byte-exact from chat_template.jinja. */
    public static final String LOW_DIRECTIVE =
            "Reasoning effort is set to low. Keep your thinking brief and focused, "
                    + "moving directly to the conclusion without unnecessary elaboration.";

    private static final QwenFamilyChatTemplate V38 = new QwenFamilyChatTemplate(true);
    private static final QwenFamilyChatTemplate V35 = new QwenFamilyChatTemplate(false);

    private final boolean flashNext;

    private QwenFamilyChatTemplate(boolean flashNext) {
        this.flashNext = flashNext;
    }

    /** Qwen3.8-Flash-Next style template (reasoning_effort + preserve_thinking knobs). */
    public static QwenFamilyChatTemplate flashNext() {
        return V38;
    }

    /** Qwen3.5-27B style template (inline think extraction, always drop history thinking). */
    public static QwenFamilyChatTemplate qwen35() {
        return V35;
    }

    @Override
    public Family family() {
        return Family.QWEN;
    }

    @Override
    public String apply(List<FamilyMessage> messages, FamilyChatOptions options) {
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("Qwen chat template requires at least one message");
        }
        if (FamilyChatOptions.QwenReasoningEffort.XHIGH != options.getQwenReasoningEffort()
                && FamilyChatOptions.QwenReasoningEffort.MEDIUM != options.getQwenReasoningEffort()
                && FamilyChatOptions.QwenReasoningEffort.LOW != options.getQwenReasoningEffort()) {
            throw new IllegalArgumentException(
                    "Unexpected reasoning effort " + options.getQwenReasoningEffort()
                            + ". Supported types are xhigh (default), medium, and low.");
        }
        StringBuilder sb = new StringBuilder();
        int lastQueryIndex = lastQueryIndex(messages);

        String directive = effortDirective(options, flashNext);
        String systemContent = systemContent(messages);

        // The 3.8 template emits a system turn whenever directives exist, even if the
        // conversation itself has no system message. 3.5 only echoes an existing system.
        boolean hasSystemTurn = !systemContent.isEmpty() || !directive.isEmpty();
        if (hasSystemTurn) {
            sb.append(IM_START).append("system\n");
            if (!directive.isEmpty()) {
                sb.append(directive).append("\n\n");
            }
            sb.append(systemContent).append(IM_END).append('\n');
        }

        boolean firstNonSystem = true;
        for (int i = 0; i < messages.size(); i++) {
            FamilyMessage message = messages.get(i);
            String role = message.getRole();
            if (FamilyMessage.ROLE_SYSTEM.equals(role)) {
                continue;
            }
            boolean first = firstNonSystem;
            firstNonSystem = false;

            if (FamilyMessage.ROLE_USER.equals(role)) {
                sb.append(IM_START).append("user\n")
                        .append(message.getContent())
                        .append(IM_END).append('\n');
            } else if (FamilyMessage.ROLE_ASSISTANT.equals(role)) {
                appendAssistant(sb, message, i, lastQueryIndex, options, first);
            } else {
                throw new IllegalArgumentException(
                        "Qwen chat template does not support role: " + role);
            }
        }

        if (options.isAddGenerationPrompt()) {
            sb.append(IM_START).append("assistant\n");
            if (options.isEnableThinking()) {
                sb.append(THINK_START).append('\n');
            } else {
                // Canonical Qwen no-thinking block: <think>\n\n</think>\n\n — the
                // empty think body carries one blank line, matching the official
                // Jinja's default no-thinking rendering used since Qwen3.
                sb.append(THINK_START).append('\n').append('\n')
                        .append(THINK_END).append("\n\n");
            }
        }
        return sb.toString();
    }

    private void appendAssistant(StringBuilder sb, FamilyMessage message, int index,
                                 int lastQueryIndex, FamilyChatOptions options, boolean first) {
        String content = strip(message.getContent());
        String reasoning = strip(message.getReasoningContent());
        if (!flashNext && reasoning.isEmpty()) {
            // 3.5 template: extract an inline <think>...</think> block from content.
            int end = content.indexOf(THINK_END);
            if (end >= 0) {
                String inline = content.substring(0, end);
                int open = inline.lastIndexOf(THINK_START);
                reasoning = open >= 0 ? inline.substring(open + THINK_START.length()) : "";
                reasoning = stripNewlines(reasoning);
                content = stripNewlines(content.substring(end + THINK_END.length()));
            }
        }
        sb.append(IM_START).append("assistant\n");
        if (!flashNext) {
            // Official 3.5: thinking is retained only after the last user query.
            if (index > lastQueryIndex) {
                sb.append(THINK_START).append('\n').append(reasoning)
                        .append('\n').append(THINK_END).append("\n\n");
            }
            sb.append(content).append(IM_END).append('\n');
            return;
        }
        // Official 3.8: preserve_thinking keeps history thinking; the final turn
        // (after the last user query) always keeps it.
        boolean afterLastQuery = index > lastQueryIndex;
        boolean keepThinking = afterLastQuery || options.isPreserveThinking();
        if (keepThinking) {
            sb.append(THINK_START).append('\n').append(reasoning)
                    .append('\n').append(THINK_END).append("\n\n");
        }
        sb.append(content).append(IM_END).append('\n');
    }

    /**
     * Index of the last user message, exactly like the official template's
     * {@code ns.last_query_index} scan (last user turn that is not a tool_response).
     */
    private static int lastQueryIndex(List<FamilyMessage> messages) {
        int result = messages.size() - 1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            FamilyMessage message = messages.get(i);
            if (FamilyMessage.ROLE_USER.equals(message.getRole())) {
                String content = strip(message.getContent());
                if (!(content.startsWith("<tool_response>") && content.endsWith("</tool_response>"))) {
                    result = i;
                    break;
                }
            }
        }
        return result;
    }

    private static String systemContent(List<FamilyMessage> messages) {
        for (FamilyMessage message : messages) {
            if (FamilyMessage.ROLE_SYSTEM.equals(message.getRole())) {
                return strip(message.getContent());
            }
        }
        return "";
    }

    private static String effortDirective(FamilyChatOptions options, boolean flashNext) {
        // The reasoning_effort knob only exists in the 3.8-Flash-Next template; the
        // official 3.5-27B template has no directive logic at all.
        if (!flashNext || !options.isEnableThinking()) {
            return "";
        }
        switch (options.getQwenReasoningEffort()) {
            case XHIGH:
                return XHIGH_DIRECTIVE;
            case LOW:
                return LOW_DIRECTIVE;
            case MEDIUM:
            default:
                return "";
        }
    }

    /** Jinja {@code |trim}: strip surrounding whitespace. */
    private static String strip(String value) {
        return value == null ? "" : value.trim();
    }

    /** Jinja {@code rstrip('\n') / lstrip('\n')} combination used around think text. */
    private static String stripNewlines(String value) {
        String result = value;
        while (!result.isEmpty() && result.charAt(0) == '\n') {
            result = result.substring(1);
        }
        while (!result.isEmpty() && result.charAt(result.length() - 1) == '\n') {
            result = result.substring(0, result.length() - 1);
        }
        return result;
    }
}
