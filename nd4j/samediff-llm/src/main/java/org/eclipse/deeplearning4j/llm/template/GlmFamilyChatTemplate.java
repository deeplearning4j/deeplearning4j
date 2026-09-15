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
 * GLM-5.x (zai-org/GLM-5.3-Flash) chat template.
 *
 * <p>Mirrors the official {@code chat_template.jinja} (source URL is recorded in
 * {@code GlmFamilyChatTemplateGoldenTest}). Rendering rules implemented verbatim:</p>
 * <ul>
 *   <li>Prompt opens with the fixed literal prefix {@code [gMASK]<sop>}. The newline
 *       after it in the Jinja source is stripped by the leading {@code {%-} of the next
 *       block, so the prefix glues directly into the reasoning-effort system block.</li>
 *   <li>{@code reasoning_effort}: only {@code low}/{@code high} are honored; anything
 *       else (including absent) collapses to {@code max}. The system block
 *       {@code <|system|>Reasoning Effort: Max|Low|High} is therefore always rendered —
 *       the guard around it can never be false because the fallback is the literal
 *       {@code 'max'}.</li>
 *   <li>Every other Jinja block in this template uses whitespace control
 *       ({@code {%- ... -%}}), so role blocks glue together with no separators.</li>
 *   <li>Roles: {@code <|system|>content}, {@code <|user|>content},
 *       {@code <|assistant|>} + think block + answer for history turns — all glued
 *       with no separators, because every block in the role loop uses
 *       whitespace control.</li>
 *   <li>Thinking: {@code <think>reasoning</think>} when reasoning is available and
 *       kept; otherwise the empty pair {@code <think></think>}. History reasoning is
 *       cleared only when the {@code clear_thinking} knob is true AND the turn is
 *       before the last user query; the final turn always keeps its reasoning.</li>
 *   <li>Generation prefix: {@code <|assistant|><think>} — the official template emits
 *       the role tag and the think-open token glued together with no newline.</li>
 * </ul>
 *
 * <p>Pure string transformation: no I/O, no tokenizer, no state.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public final class GlmFamilyChatTemplate implements FamilyChatTemplate {

    public static final String PREFIX = "[gMASK]<sop>";
    public static final String SYSTEM = "<|system|>";
    public static final String USER = "<|user|>";
    public static final String ASSISTANT = "<|assistant|>";
    public static final String OBSERVATION = "<|observation|>";
    public static final String THINK_START = "<think>";
    public static final String THINK_END = "</think>";

    private static final GlmFamilyChatTemplate INSTANCE = new GlmFamilyChatTemplate();

    public static GlmFamilyChatTemplate instance() {
        return INSTANCE;
    }

    @Override
    public Family family() {
        return Family.GLM;
    }

    @Override
    public String apply(List<FamilyMessage> messages, FamilyChatOptions options) {
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("GLM chat template requires at least one message");
        }
        StringBuilder sb = new StringBuilder(PREFIX);
        sb.append(SYSTEM).append("Reasoning Effort: ")
                .append(options.getGlmReasoningEffort().capitalized());

        int lastUserIndex = -1;
        for (int i = 0; i < messages.size(); i++) {
            if (FamilyMessage.ROLE_USER.equals(messages.get(i).getRole())) {
                lastUserIndex = i;
            }
        }

        for (int i = 0; i < messages.size(); i++) {
            FamilyMessage message = messages.get(i);
            String role = message.getRole();
            if (FamilyMessage.ROLE_SYSTEM.equals(role)) {
                sb.append(SYSTEM).append(message.getContent());
            } else if (FamilyMessage.ROLE_USER.equals(role)) {
                sb.append(USER).append(message.getContent());
            } else if (FamilyMessage.ROLE_ASSISTANT.equals(role)) {
                // The official assistant branch is fully whitespace-controlled: the
                // role tag, think block, and answer glue together with no newline.
                sb.append(ASSISTANT);
                renderAssistantTurn(sb, message, i > lastUserIndex, options.isGlmClearThinking());
            } else if (FamilyMessage.ROLE_TOOL.equals(role)) {
                sb.append(OBSERVATION).append("<tool_response>")
                        .append(message.getContent()).append("</tool_response>");
            } else {
                throw new IllegalArgumentException(
                        "GLM chat template does not support role: " + role);
            }
        }

        if (options.isAddGenerationPrompt()) {
            sb.append(ASSISTANT).append(THINK_START);
        }
        return sb.toString();
    }

    private static void renderAssistantTurn(StringBuilder sb, FamilyMessage message,
                                            boolean afterLastUser, boolean clearThinking) {
        String content = message.getContent() == null ? "" : message.getContent();
        String reasoning = message.getReasoningContent() == null
                ? "" : message.getReasoningContent();
        if (reasoning.isEmpty() && content.contains(THINK_END)) {
            // Official template: an inline <think>...</think> block inside content is
            // split out before rendering.
            int end = content.indexOf(THINK_END);
            String inline = content.substring(0, end);
            int open = inline.lastIndexOf(THINK_START);
            reasoning = open >= 0 ? inline.substring(open + THINK_START.length()) : "";
            content = content.substring(end + THINK_END.length());
        }
        boolean keep = (afterLastUser || !clearThinking) && !reasoning.isEmpty();
        if (keep) {
            sb.append(THINK_START).append(reasoning).append(THINK_END);
        } else {
            sb.append(THINK_START).append(THINK_END);
        }
        String trimmed = content.strip();
        if (!trimmed.isEmpty()) {
            sb.append(trimmed);
        }
    }
}
