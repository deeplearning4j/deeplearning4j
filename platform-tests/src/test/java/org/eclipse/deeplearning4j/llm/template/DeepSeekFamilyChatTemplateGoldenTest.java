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
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Golden-string tests for {@link DeepSeekFamilyChatTemplate}.
 *
 * <p>DeepSeek-V4.1-Flash ships no Jinja chat template. All goldens are derived from the
 * official prompt-encoding reference and its documentation:
 * <ul>
 *   <li>Python encoder (special tokens, effort prefix, think placement, transition
 *       tokens): https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py</li>
 *   <li>Encoding README with worked examples:
 *       https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/README.md</li>
 * </ul>
 * Special tokens are byte-exact full-width sequences (U+FF5C fullwidth vertical line
 * {@code ｜}, U+2581 lower one eighth block {@code ▁}) copied from
 * {@code encoding.py}:
 * <pre>
 * bos_token = "&lt;｜begin▁of▁sentence｜&gt;"
 * eos_token = "&lt;｜end▁of▁sentence｜&gt;"
 * USER_SP_TOKEN = "&lt;｜User｜&gt;"
 * ASSISTANT_SP_TOKEN = "&lt;｜Assistant｜&gt;"
 * SYSTEM_SP_TOKEN = "&lt;｜System｜&gt;"
 * REASONING_EFFORT_TEMPLATE = "Reasoning Effort: {budget} (range 1-100, the higher
 *     the value, the more thorough the reasoning)\n\n"
 * </pre></p>
 */
class DeepSeekFamilyChatTemplateGoldenTest {

    private final DeepSeekFamilyChatTemplate deepseek = DeepSeekFamilyChatTemplate.instance();

    @Test
    void basicSystemUserGenerationMatchesOfficialQuickStart() {
        // Source: encoding/README.md "Quick start" example (exact prompt string shown
        // there for system+user, thinking mode, reasoning_effort=75):
        // '<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the
        //  value, the more thorough the reasoning)\n\nYou are a helpful assistant.
        //  <｜User｜>What is 2+2?<｜Assistant｜><think>'
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("What is 2+2?"));
        assertEquals(
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>"
                        + "Reasoning Effort: 75 (range 1-100, the higher the value, "
                        + "the more thorough the reasoning)\n\n"
                        + "You are a helpful assistant."
                        + "<｜User｜>What is 2+2?"
                        + "<｜Assistant｜><think>",
                deepseek.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void byteExactSpecialTokens() {
        // Source: encoding/encoding.py token block:
        //   bos_token = "<｜begin▁of▁sentence｜>"
        //   eos_token = "<｜end▁of▁sentence｜>"
        //   USER_SP_TOKEN = "<｜User｜>"
        //   ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
        //   SYSTEM_SP_TOKEN = "<｜System｜>"
        // Each token is asserted code-point by code-point so an ASCII '|' or '_' lookalike
        // substitution cannot pass silently.
        assertTokenBytes(DeepSeekFamilyChatTemplate.BOS,
                "<", '\uFF5C', "begin", '\u2581', "of", '\u2581', "sentence", '\uFF5C', ">");
        assertTokenBytes(DeepSeekFamilyChatTemplate.EOS,
                "<", '\uFF5C', "end", '\u2581', "of", '\u2581', "sentence", '\uFF5C', ">");
        assertTokenBytes(DeepSeekFamilyChatTemplate.USER_TOKEN,
                "<", '\uFF5C', "User", '\uFF5C', ">");
        assertTokenBytes(DeepSeekFamilyChatTemplate.ASSISTANT_TOKEN,
                "<", '\uFF5C', "Assistant", '\uFF5C', ">");
        assertTokenBytes(DeepSeekFamilyChatTemplate.SYSTEM_TOKEN,
                "<", '\uFF5C', "System", '\uFF5C', ">");
        assertTokenBytes(DeepSeekFamilyChatTemplate.LATEST_REMINDER_TOKEN,
                "<", '\uFF5C', "latest", '\u2581', "reminder", '\uFF5C', ">");
    }

    private static void assertTokenBytes(Object... expectedParts) {
        String token = String.valueOf(expectedParts[0]);
        StringBuilder expected = new StringBuilder();
        for (int i = 1; i < expectedParts.length; i++) {
            Object part = expectedParts[i];
            expected.append(part instanceof Character ? part.toString() : (String) part);
        }
        assertEquals(expected.toString(), token);
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (c == '\uFF5C' || c == '\u2581') {
                return; // found the full-width marker; an ASCII lookalike would have failed above
            }
        }
        throw new AssertionError("Token lacks full-width markers: " + token);
    }

    @Test
    void multiTurnThinkingHistoryDropsPreLastQueryReasoning() {
        // Source: encoding/README.md "Basic chat" + "Thinking mode":
        //   '<｜begin▁of▁sentence｜>{system_prompt}
        //    <｜User｜>{user_message}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>
        //    <｜User｜>{user_message_2}<｜Assistant｜></think>{response_2}<｜end▁of▁sentence｜>'
        // With drop_thinking=True (official default), assistant reasoning before the
        // last user message is stripped; the completed final turn keeps its block and
        // ends with EOS. Transition rule: Assistant header attaches to the END of the
        // preceding user turn unless the next message is an assistant turn.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistantThinking("2 + 2 = 4.", "Simple arithmetic."),
                FamilyMessage.user("Thanks! And 3+3?"));
        assertEquals(
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>"
                        + "Reasoning Effort: 75 (range 1-100, the higher the value, "
                        + "the more thorough the reasoning)\n\n"
                        + "You are a helpful assistant."
                        + "<｜User｜>What is 2+2?"
                        + "<｜Assistant｜></think>2 + 2 = 4.<｜end▁of▁sentence｜>"
                        + "<｜User｜>Thanks! And 3+3?"
                        + "<｜Assistant｜><think>",
                deepseek.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void multiTurnCompletedConversation() {
        // Completed two-round conversation (add_generation_prompt=false): both
        // assistant turns keep nothing before the last user turn except the final one.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("What is 2+2?"),
                FamilyMessage.assistantThinking("2 + 2 = 4.", "Simple arithmetic."),
                FamilyMessage.user("And 3+3?"),
                FamilyMessage.assistantThinking("3 + 3 = 6.", "Also simple."));
        assertEquals(
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>"
                        + "Reasoning Effort: 75 (range 1-100, the higher the value, "
                        + "the more thorough the reasoning)\n\n"
                        + "You are a helpful assistant."
                        + "<｜User｜>What is 2+2?"
                        + "<｜Assistant｜></think>2 + 2 = 4.<｜end▁of▁sentence｜>"
                        + "<｜User｜>And 3+3?"
                        + "<｜Assistant｜><think>Also simple.</think>3 + 3 = 6.<｜end▁of▁sentence｜>",
                deepseek.apply(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .addGenerationPrompt(false).build()));
    }

    @Test
    void chatModeEmitsClosedThinkBlock() {
        // Source: encoding/README.md "Basic chat":
        //   'In chat mode (thinking_mode="chat"), </think> is placed right after
        //    <｜Assistant｜> to immediately close the thinking block, so the model
        //    generates content directly.'
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("What is 2+2?"));
        assertEquals(
                "<｜begin▁of▁sentence｜>"
                        + "<｜User｜>What is 2+2?"
                        + "<｜Assistant｜></think>",
                deepseek.applyForGeneration(messages,
                        FamilyChatOptions.defaults().toBuilder()
                                .enableThinking(false).build()));
    }

    @Test
    void numericReasoningEffortBudgetAndAliases() {
        // Source: encoding/encoding.py:
        //   REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
        //   DEFAULT_REASONING_EFFORT = "high"
        //   REASONING_EFFORT_TEMPLATE = "Reasoning Effort: {budget} (range 1-100, "
        //       "the higher the value, the more thorough the reasoning)\n\n"
        assertEquals(50, DeepSeekFamilyChatTemplate.effortAlias("low"));
        assertEquals(75, DeepSeekFamilyChatTemplate.effortAlias("high"));
        assertEquals(100, DeepSeekFamilyChatTemplate.effortAlias("max"));

        // effort=100 ('max') renders at index 0 in thinking mode only.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.system("You are a helpful assistant."),
                FamilyMessage.user("Hi"));
        String effort100 =
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>"
                        + "Reasoning Effort: 100 (range 1-100, the higher the value, "
                        + "the more thorough the reasoning)\n\n"
                        + "You are a helpful assistant."
                        + "<｜User｜>Hi"
                        + "<｜Assistant｜><think>";
        assertEquals(effort100, deepseek.applyForGeneration(messages,
                FamilyChatOptions.defaults().toBuilder()
                        .deepseekReasoningEffort("max").build()));

        // Chat mode never renders the effort prefix.
        String chatMode =
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>"
                        + "You are a helpful assistant."
                        + "<｜User｜>Hi"
                        + "<｜Assistant｜></think>";
        assertEquals(chatMode, deepseek.applyForGeneration(messages,
                FamilyChatOptions.defaults().toBuilder()
                        .enableThinking(false).build()));
    }

    @Test
    void midConversationSystemMessageTriggersAssistantHeader() {
        // Source: encoding/README.md "V4.1 changes": 'Mid-conversation system messages
        // are supported via the <｜System｜> token. A mid-conversation system message
        // behaves like a user message for the purpose of appending the assistant
        // generation header.'
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("Hello"),
                FamilyMessage.system("Answer in French from now on."),
                FamilyMessage.user("Bonjour"));
        // Trace: in thinking mode the effort prefix always renders at index 0 and pulls
        // the <｜System｜> token with it, even when the first message is a user turn
        // (encoding.py: SYSTEM_SP_TOKEN if index == 0 and (reasoning_effort_prompt or
        // role == "system")). 'Hello' is followed by a system turn (not assistant/
        // latest_reminder), so render_message returns early with NO header; the
        // mid-conversation system turn renders via <｜System｜>; only the final user
        // turn gets the assistant generation header.
        assertEquals(
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, "
                        + "the more thorough the reasoning)\n\n"
                        + "<｜User｜>Hello"
                        + "<｜System｜>Answer in French from now on."
                        + "<｜User｜>Bonjour"
                        + "<｜Assistant｜><think>",
                deepseek.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }

    @Test
    void toolResultMergesIntoUserTurn() {
        // Source: encoding/README.md "Message format":
        //   '<｜User｜><tool_result>{result_json}</tool_result><｜Assistant｜><think>...'
        // A tool message becomes a <tool_result> block inside the preceding user turn.
        List<FamilyMessage> messages = Arrays.asList(
                FamilyMessage.user("What is the weather?"),
                FamilyMessage.toolResult("call_1", "{\"temp_c\": 21}"),
                FamilyMessage.user("Thanks"));
        // Trace: merge_tool_messages folds the tool result and the trailing user text
        // into one user message; its content blocks render joined with "\n\n"
        // (official tool_output_template + "\n\n".join(parts)). Thinking mode renders
        // the effort prefix at index 0 even for a user-first conversation.
        assertEquals(
                "<｜begin▁of▁sentence｜>"
                        + "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, "
                        + "the more thorough the reasoning)\n\n"
                        + "<｜User｜>What is the weather?\n\n"
                        + "<tool_result>{\"temp_c\": 21}</tool_result>\n\nThanks"
                        + "<｜Assistant｜><think>",
                deepseek.applyForGeneration(messages, FamilyChatOptions.defaults()));
    }
}
