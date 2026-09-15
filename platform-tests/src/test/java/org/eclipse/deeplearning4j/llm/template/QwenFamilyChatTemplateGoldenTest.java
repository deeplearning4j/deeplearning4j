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
 * Golden-string tests for {@link QwenFamilyChatTemplate}.
 *
 * <p>Every golden is derived directly from the official upstream templates:
 * <ul>
 *   <li>Qwen3.8-Flash-Next chat_template (tokenizer_config.json "chat_template"):
 *       https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/tokenizer_config.json</li>
 *   <li>Qwen3.5-27B chat_template (tokenizer_config.json "chat_template"):
 *       https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json</li>
 * </ul>
 * The expected strings below are hand-traced from those Jinja sources, including the
 * exact directive strings, whitespace-control behavior, and the
 * reasoning_effort / preserve_thinking knobs.</p>
 */
class QwenFamilyChatTemplateGoldenTest {

    private final QwenFamilyChatTemplate flashNext = QwenFamilyChatTemplate.flashNext();

    @Test
    void basicSystemUserGeneration38() {
        // 3.8, reasoning_effort defaults to xhigh: directive is prepended to the
        // system content, then ChatML user turn and the thinking generation prefix.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("Hi there"));

        // Source: Qwen3.8-Flash-Next tokenizer_config.json chat_template:
        //   reasoning_instructions = 'Reasoning effort is set to xhigh. Please think
        //   carefully through the task, validate key assumptions, consider plausible
        //   alternatives, and prioritize correctness, consistency, and clarity in the
        //   final answer.'
        //   {%- elif reasoning_instructions %}
        //       {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        //   ... system turn: '<|im_start|>system\n' + (reasoning_instructions +
        //       '\n\n' if reasoning_instructions else '') + content + '<|im_end|>\n'
        //   user turn: '<|im_start|>' + role + '\n' + content + '<|im_end|>' + '\n'
        //   add_generation_prompt: '<|im_start|>assistant\n<think>\n'
        assertEquals(
                "<|im_start|>system\n"
                        + "Reasoning effort is set to xhigh. Please think carefully through the task, "
                        + "validate key assumptions, consider plausible alternatives, and prioritize "
                        + "correctness, consistency, and clarity in the final answer."
                        + "\n\nYou are a helpful assistant.<|im_end|>\n"
                        + "<|im_start|>user\nHi there<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n",
                flashNext.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void basicSystemUserGenerationNoThinking() {
        // enable_thinking=false closes the think block immediately.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("Hi there"));
        assertEquals(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        + "<|im_start|>user\nHi there<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                flashNext.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .enableThinking(false).build()));
    }

    @Test
    void reasoningEffortLowEmitsLowDirective() {
        // Source: Qwen3.8-Flash-Next chat_template low branch:
        //   'Reasoning effort is set to low. Keep your thinking brief and focused,
        //   moving directly to the conclusion without unnecessary elaboration.'
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("Be terse."),
                FamilyMessage.user("Hi"));
        assertEquals(
                "<|im_start|>system\n"
                        + "Reasoning effort is set to low. Keep your thinking brief and focused, "
                        + "moving directly to the conclusion without unnecessary elaboration."
                        + "\n\nBe terse.<|im_end|>\n"
                        + "<|im_start|>user\nHi<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n",
                flashNext.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .qwenReasoningEffort(FamilyChatOptions.QwenReasoningEffort.LOW)
                                .build()));
    }

    @Test
    void reasoningEffortMediumEmitsNoDirective() {
        // Source: Qwen3.8-Flash-Next chat_template: 'medium' has no instruction text;
        // the system turn renders only when content exists.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("Be precise."),
                FamilyMessage.user("Hi"));
        assertEquals(
                "<|im_start|>system\nBe precise.<|im_end|>\n"
                        + "<|im_start|>user\nHi<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n",
                flashNext.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .qwenReasoningEffort(FamilyChatOptions.QwenReasoningEffort.MEDIUM)
                                .build()));
    }

    @Test
    void multiTurnHistoryWithPreservedThinking() {
        // preserve_thinking=true (default) keeps reasoning blocks on history turns.
        // reasoning_effort=MEDIUM renders no directive, isolating the preserve_thinking
        // assertion. Source: Qwen3.8-Flash-Next chat_template assistant branch:
        //   {%- if preserve_thinking is undefined or preserve_thinking is true
        //       or loop.index0 > ns.last_query_index %}
        //       '<|im_start|>assistant\n<think>\n' + reasoning_content +
        //       '\n</think>\n\n' + content
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistantThinking("2+2 equals 4.", "Trivial addition."),
                FamilyMessage.user("And 3+3?"));

        // Golden trace with preserve_thinking=true: the first assistant turn (before
        // the last user query) keeps its think block.
        assertEquals(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        + "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\nTrivial addition.\n</think>\n\n2+2 equals 4.<|im_end|>\n"
                        + "<|im_start|>user\nAnd 3+3?<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n",
                flashNext.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .qwenReasoningEffort(FamilyChatOptions.QwenReasoningEffort.MEDIUM)
                                .build()));
    }

    @Test
    void multiTurnHistoryDropsThinkingWhenPreserveThinkingFalse() {
        // preserve_thinking=false strips reasoning from turns BEFORE the last user
        // query; the final assistant turn would still keep it, but here history ends
        // at the last user query so all history thinking is dropped.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistantThinking("2+2 equals 4.", "Trivial addition."),
                FamilyMessage.user("And 3+3?"));
        assertEquals(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        + "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
                        + "<|im_start|>assistant\n2+2 equals 4.<|im_end|>\n"
                        + "<|im_start|>user\nAnd 3+3?<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n",
                flashNext.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .qwenReasoningEffort(FamilyChatOptions.QwenReasoningEffort.MEDIUM)
                                .preserveThinking(false).build()));
    }

    @Test
    void threePointFiveExtractsInlineThinkAndDropsHistoryThinking() {
        // Qwen3.5-27B: no reasoning_effort directive, no preserve_thinking knob.
        // Source: Qwen3.5-27B chat_template assistant branch:
        //   {%- if '</think>' in content %}
        //       reasoning_content = content.split('</think>')[0].rstrip('\n')
        //           .split('<think>')[-1].lstrip('\n')
        //       content = content.split('</think>')[-1].lstrip('\n')
        //   {%- if loop.index0 > ns.last_query_index %} keeps <think>...</think>;
        //   earlier turns render content only.
        QwenFamilyChatTemplate qwen35 = QwenFamilyChatTemplate.qwen35();
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistant("<think>\nTrivial addition.\n</think>\n\n2+2 equals 4."),
                FamilyMessage.user("And 3+3?"));
        assertEquals(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        + "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
                        + "<|im_start|>assistant\n2+2 equals 4.<|im_end|>\n"
                        + "<|im_start|>user\nAnd 3+3?<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n",
                qwen35.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void qwen35KeepsThinkingOnFinalAssistantTurn() {
        // A completed conversation (add_generation_prompt=false): the assistant turn
        // AFTER the last user query keeps its inline think block.
        QwenFamilyChatTemplate qwen35 = QwenFamilyChatTemplate.qwen35();
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistant("<think>\nTrivial addition.\n</think>\n\n2+2 equals 4."));
        assertEquals(
                "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\nTrivial addition.\n</think>\n\n2+2 equals 4.<|im_end|>\n",
                qwen35.apply(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .addGenerationPrompt(false).build()));
    }
}
