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

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Golden-string tests for {@link GlmFamilyChatTemplate}.
 *
 * <p>Goldens derive from the official upstream template:
 * https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/chat_template.jinja
 * Expected strings are hand-traced from that Jinja source, including the
 * {@code [gMASK]<sop>} literal header, the always-on
 * {@code <|system|>Reasoning Effort: Max|Low|High} block, whitespace-control gluing
 * (no newlines between role blocks), the assistant-turn
 * {@code <|assistant|>\n<think>...</think>} layout, and the
 * {@code <|assistant|><think>} generation prefix.</p>
 */
class GlmFamilyChatTemplateGoldenTest {

    private final GlmFamilyChatTemplate glm = GlmFamilyChatTemplate.instance();

    @Test
    void basicSystemUserGeneration() {
        // Source: GLM-5.3-Flash chat_template.jinja:
        //   '[gMASK]<sop>\n{%- set effective_reasoning_effort = ... -%}'
        //   The {%- whitespace-control prefix strips the newline after [gMASK]<sop>,
        //   so the header glues directly onto the next block.
        // Default (no knob) collapses to 'max' -> 'Max'.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("Hi there"));
        assertEquals(
                "[gMASK]<sop>"
                        + "<|system|>Reasoning Effort: Max"
                        + "<|system|>You are a helpful assistant."
                        + "<|user|>Hi there"
                        + "<|assistant|><think>",
                glm.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void basicSystemUserGenerationLowEffort() {
        // Source: reasoning_effort='low' -> '<|system|>Reasoning Effort: Low'.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("Hi there"));
        assertEquals(
                "[gMASK]<sop>"
                        + "<|system|>Reasoning Effort: Low"
                        + "<|system|>You are a helpful assistant."
                        + "<|user|>Hi there"
                        + "<|assistant|><think>",
                glm.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .glmReasoningEffort(FamilyChatOptions.GlmReasoningEffort.LOW)
                                .build()));
    }

    @Test
    void multiTurnHistory() {
        // Source: role loop: '<|user|>{{ visible_text(m.content) }}', assistant
        // history: '<|assistant|>' + think + content (whitespace control strips the
        // newlines, so '<|assistant|><think>R</think>Answer').
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistantThinking("2+2 equals 4.", "Trivial addition."),
                FamilyMessage.user("And 3+3?"));
        assertEquals(
                "[gMASK]<sop>"
                        + "<|system|>Reasoning Effort: Max"
                        + "<|user|>What is 2+2?"
                        + "<|assistant|><think>Trivial addition.</think>2+2 equals 4."
                        + "<|user|>And 3+3?"
                        + "<|assistant|><think>",
                glm.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void multiTurnHistoryClearThinking() {
        // Source: clear_thinking knob:
        //   {%- if (not clear_thinking or loop.index0 > ns.last_user_index)
        //       and reasoning_content is defined -%}
        //   {{ '<think>' + reasoning_content + '</think>'}}{%- else -%}
        //   {{ '<think></think>' }}{%- endif -%}
        // clear_thinking=true drops reasoning on turns before the last user query.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistantThinking("2+2 equals 4.", "Trivial addition."),
                FamilyMessage.user("And 3+3?"));
        assertEquals(
                "[gMASK]<sop>"
                        + "<|system|>Reasoning Effort: Max"
                        + "<|user|>What is 2+2?"
                        + "<|assistant|><think></think>2+2 equals 4."
                        + "<|user|>And 3+3?"
                        + "<|assistant|><think>",
                glm.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .glmClearThinking(true).build()));
    }

    @Test
    void completedAssistantTurnKeepsEmptyThinkPairWhenNoReasoning() {
        // Source: when no reasoning is available the template still emits
        // '<think></think>'; content.strip() is appended only when non-empty.
        // add_generation_prompt=false shows the plain conversation form.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("Hello"),
                FamilyMessage.assistant("Hi! How can I help?"));
        assertEquals(
                "[gMASK]<sop>"
                        + "<|system|>Reasoning Effort: Max"
                        + "<|user|>Hello"
                        + "<|assistant|><think></think>Hi! How can I help?",
                glm.apply(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .addGenerationPrompt(false).build()));
    }

    @Test
    void midConversationSystemTurn() {
        // Source: system branch inside the role loop:
        //   {%- elif m.role == 'system' -%}<|system|>{{ visible_text(m.content) }}
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("Hello"),
                FamilyMessage.system("New directive: answer in French."),
                FamilyMessage.user("Bonjour"));
        assertEquals(
                "[gMASK]<sop>"
                        + "<|system|>Reasoning Effort: Max"
                        + "<|user|>Hello"
                        + "<|system|>New directive: answer in French."
                        + "<|user|>Bonjour"
                        + "<|assistant|><think>",
                glm.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }
}
