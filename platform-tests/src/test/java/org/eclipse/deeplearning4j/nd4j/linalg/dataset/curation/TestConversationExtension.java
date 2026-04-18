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

package org.eclipse.deeplearning4j.nd4j.linalg.dataset.curation;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.dataset.curation.format.ConversationExtension;
import org.nd4j.linalg.dataset.curation.format.ConversationTurn;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ConversationExtension}: multi-turn conversation synthesis
 * from single-turn instruction/response data.
 */
public class TestConversationExtension {

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Build 10 single-turn pairs (user + assistant) as a flat list. */
    private static List<ConversationTurn> makeNTurns(int n) {
        List<ConversationTurn> turns = new ArrayList<>(n * 2);
        for (int i = 0; i < n; i++) {
            turns.add(ConversationTurn.user("Question " + i + "?"));
            turns.add(ConversationTurn.assistant("Answer " + i + "."));
        }
        return turns;
    }

    /** Count total user+assistant turns across all conversations. */
    private static int totalTurns(List<List<ConversationTurn>> conversations) {
        int count = 0;
        for (List<ConversationTurn> conv : conversations) {
            count += conv.size();
        }
        return count;
    }

    /** Count turns with the given role across all conversations. */
    private static int countRole(List<List<ConversationTurn>> conversations, String role) {
        int count = 0;
        for (List<ConversationTurn> conv : conversations) {
            for (ConversationTurn turn : conv) {
                if (role.equals(turn.getRole())) count++;
            }
        }
        return count;
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    @Test
    public void testBasicExtension() {
        // 10 single-turn pairs → some multi-turn conversations
        List<ConversationTurn> input = makeNTurns(10);

        ConversationExtension ext = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(3)
                .extensionProbability(0.8)
                .seed(42L)
                .build();

        List<List<ConversationTurn>> conversations = ext.extend(input);

        assertFalse(conversations.isEmpty(), "Should produce at least one conversation");

        // All 10 pairs must be consumed exactly once
        assertEquals(10, countRole(conversations, "user"),
                "All user turns must appear exactly once");
        assertEquals(10, countRole(conversations, "assistant"),
                "All assistant turns must appear exactly once");

        // Each conversation must have between minTurns and maxTurns pairs
        // (i.e., between 2 and 6 raw turns, since each pair = user + assistant)
        for (List<ConversationTurn> conv : conversations) {
            int pairs = conv.size() / 2; // user+assistant pairs
            assertTrue(pairs >= 1, "Each conversation must have at least minTurns=1 pair");
            assertTrue(pairs <= 3, "Each conversation must have at most maxTurns=3 pairs");
        }
    }

    @Test
    public void testExtensionProbabilityZero() {
        // With extensionProbability=0, every pair stays as its own 1-turn conversation
        List<ConversationTurn> input = makeNTurns(8);

        ConversationExtension ext = ConversationExtension.builder()
                .extensionProbability(0.0)
                .seed(99L)
                .build();

        List<List<ConversationTurn>> conversations = ext.extend(input);

        assertEquals(8, conversations.size(),
                "extensionProbability=0.0 must yield one conversation per input pair");

        for (List<ConversationTurn> conv : conversations) {
            assertEquals(2, conv.size(),
                    "Each conversation must contain exactly 1 user + 1 assistant turn");
            assertEquals("user", conv.get(0).getRole());
            assertEquals("assistant", conv.get(1).getRole());
        }
    }

    @Test
    public void testExtensionProbabilityOne() {
        // With extensionProbability=1.0 and maxTurns > 1, all pairs are merged
        List<ConversationTurn> input = makeNTurns(6);

        ConversationExtension ext = ConversationExtension.builder()
                .minTurns(2)
                .maxTurns(6)
                .extensionProbability(1.0)
                .seed(7L)
                .build();

        List<List<ConversationTurn>> conversations = ext.extend(input);

        // With minTurns=2 and 6 pairs, we expect fewer than 6 conversations
        assertTrue(conversations.size() < 6,
                "extensionProbability=1.0 must merge pairs into multi-turn conversations");

        // All pairs must still be consumed exactly once
        assertEquals(6, countRole(conversations, "user"),
                "All 6 user turns must appear");
        assertEquals(6, countRole(conversations, "assistant"),
                "All 6 assistant turns must appear");
    }

    @Test
    public void testMaxTotalTokens() {
        // Each pair has ~100 chars per turn → ~50 tokens/turn → 25 tokens/turn rough.
        // Use a very small token budget to force 1-pair conversations.
        String longInstruction = "This is a fairly long instruction with quite a few words in it";   // 62 chars
        String longResponse    = "This is a correspondingly long response to match the instruction"; // 64 chars
        // Total chars per pair: ~126 → ~32 tokens

        List<ConversationTurn> input = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            input.add(ConversationTurn.user(longInstruction));
            input.add(ConversationTurn.assistant(longResponse));
        }

        // maxTotalTokens=40 → fits 1 pair (32 tokens) but not 2 pairs (64 tokens)
        ConversationExtension ext = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(5)
                .extensionProbability(1.0)
                .maxTotalTokens(40)
                .seed(5L)
                .build();

        List<List<ConversationTurn>> conversations = ext.extend(input);

        for (List<ConversationTurn> conv : conversations) {
            // Only system turns are excluded from the pair count
            long pairs = conv.stream().filter(t -> "user".equals(t.getRole())).count();
            // With a tight budget, at most 1 pair should fit
            assertTrue(pairs <= 1, "Token budget should limit each conversation to 1 pair, got " + pairs);
        }
    }

    @Test
    public void testSeedReproducibility() {
        List<ConversationTurn> input = makeNTurns(12);

        ConversationExtension ext1 = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(4)
                .extensionProbability(0.7)
                .seed(123L)
                .build();

        ConversationExtension ext2 = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(4)
                .extensionProbability(0.7)
                .seed(123L)
                .build();

        List<List<ConversationTurn>> result1 = ext1.extend(input);
        List<List<ConversationTurn>> result2 = ext2.extend(input);

        assertEquals(result1.size(), result2.size(),
                "Same seed must produce same number of conversations");

        for (int i = 0; i < result1.size(); i++) {
            List<ConversationTurn> conv1 = result1.get(i);
            List<ConversationTurn> conv2 = result2.get(i);
            assertEquals(conv1.size(), conv2.size(),
                    "Conversation " + i + " must have same length with same seed");
            for (int j = 0; j < conv1.size(); j++) {
                assertEquals(conv1.get(j).getContent(), conv2.get(j).getContent(),
                        "Turn content must be identical with same seed");
            }
        }
    }

    @Test
    public void testExtendAndFormat() {
        List<ConversationTurn> input = makeNTurns(6);

        ConversationExtension ext = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(3)
                .extensionProbability(0.6)
                .seed(55L)
                .build();

        List<FormattedExample> formatted = ext.extendAndFormat(input, ChatTemplate.CHATML);

        assertFalse(formatted.isEmpty(), "Should produce formatted examples");

        for (FormattedExample example : formatted) {
            String text = example.getText();
            // CHATML format uses <|im_start|> markers
            assertTrue(text.contains("<|im_start|>user"),
                    "Formatted text must contain ChatML user marker");
            assertTrue(text.contains("<|im_start|>assistant"),
                    "Formatted text must contain ChatML assistant marker");
            // Loss mask must cover the full text
            assertEquals(text.length(), example.getLossMask().length,
                    "Loss mask length must equal text length");
            // At least some characters must be trainable (assistant content)
            int trainable = 0;
            for (int m : example.getLossMask()) trainable += m;
            assertTrue(trainable > 0, "Formatted example must have trainable characters");
        }
    }

    @Test
    public void testSystemMessage() {
        List<ConversationTurn> input = makeNTurns(6);
        String sysMsg = "You are a helpful AI assistant.";

        ConversationExtension ext = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(3)
                .extensionProbability(0.5)
                .addSystemMessage(true)
                .systemMessage(sysMsg)
                .seed(11L)
                .build();

        List<List<ConversationTurn>> conversations = ext.extend(input);

        assertFalse(conversations.isEmpty());

        for (List<ConversationTurn> conv : conversations) {
            // First turn must always be the system message
            assertFalse(conv.isEmpty(), "Conversation must not be empty");
            ConversationTurn first = conv.get(0);
            assertEquals("system", first.getRole(),
                    "First turn must be system when addSystemMessage=true");
            assertEquals(sysMsg, first.getContent(),
                    "System turn must carry the configured system message");
        }
    }

    @Test
    public void testFromPairsEntryList() {
        List<Map.Entry<String, String>> pairs = new ArrayList<>();
        pairs.add(new AbstractMap.SimpleImmutableEntry<>("What is 2+2?", "4"));
        pairs.add(new AbstractMap.SimpleImmutableEntry<>("Capital of France?", "Paris"));
        pairs.add(new AbstractMap.SimpleImmutableEntry<>("Explain gravity.", "Gravity is a force."));

        List<ConversationTurn> turns = ConversationExtension.fromPairs(pairs);

        assertEquals(6, turns.size(), "3 pairs must produce 6 turns");
        for (int i = 0; i < 6; i += 2) {
            assertEquals("user",      turns.get(i).getRole(),     "Even index must be user");
            assertEquals("assistant", turns.get(i + 1).getRole(), "Odd index must be assistant");
        }
        assertEquals("What is 2+2?", turns.get(0).getContent());
        assertEquals("4",            turns.get(1).getContent());
        assertEquals("Paris",        turns.get(3).getContent());
    }

    @Test
    public void testFromPairsParallelLists() {
        List<String> instructions = new ArrayList<>();
        List<String> responses    = new ArrayList<>();
        instructions.add("Tell me a joke.");
        responses.add("Why did the chicken cross the road? To get to the other side!");
        instructions.add("What is AI?");
        responses.add("Artificial Intelligence.");

        List<ConversationTurn> turns = ConversationExtension.fromPairs(instructions, responses);

        assertEquals(4, turns.size());
        assertEquals("user",      turns.get(0).getRole());
        assertEquals("Tell me a joke.", turns.get(0).getContent());
        assertEquals("assistant", turns.get(1).getRole());
        assertEquals("Why did the chicken cross the road? To get to the other side!",
                turns.get(1).getContent());
        assertEquals("user",      turns.get(2).getRole());
        assertEquals("assistant", turns.get(3).getRole());
    }

    @Test
    public void testFromPairsMismatchedSizesThrows() {
        List<String> instructions = new ArrayList<>();
        List<String> responses    = new ArrayList<>();
        instructions.add("A");
        // responses is empty → mismatch

        assertThrows(IllegalArgumentException.class,
                () -> ConversationExtension.fromPairs(instructions, responses),
                "Mismatched list sizes must throw IllegalArgumentException");
    }

    @Test
    public void testEmptyInput() {
        ConversationExtension ext = ConversationExtension.builder().build();

        List<List<ConversationTurn>> result = ext.extend(new ArrayList<>());
        assertTrue(result.isEmpty(), "Empty input must produce empty output");

        List<FormattedExample> formatted = ext.extendAndFormat(new ArrayList<>(), ChatTemplate.CHATML);
        assertTrue(formatted.isEmpty(), "Empty input must produce empty formatted output");
    }

    @Test
    public void testSingleInput() {
        // A single pair must always produce exactly one 1-turn conversation
        List<ConversationTurn> input = makeNTurns(1);

        ConversationExtension ext = ConversationExtension.builder()
                .minTurns(1)
                .maxTurns(5)
                .extensionProbability(0.9)
                .seed(0L)
                .build();

        List<List<ConversationTurn>> conversations = ext.extend(input);

        assertEquals(1, conversations.size(),
                "Single pair must yield exactly one conversation");
        // Must contain the one user + one assistant turn
        assertEquals(1, countRole(conversations, "user"));
        assertEquals(1, countRole(conversations, "assistant"));
    }

    @Test
    public void testBuilderDefaults() {
        ConversationExtension ext = ConversationExtension.builder().build();

        assertEquals(1,    ext.getMinTurns());
        assertEquals(5,    ext.getMaxTurns());
        assertEquals(0.5,  ext.getExtensionProbability(), 1e-9);
        assertEquals(4096, ext.getMaxTotalTokens());
        assertFalse(ext.isAddSystemMessage());
        assertNull(ext.getSystemMessage());
        assertEquals(-1L, ext.getSeed());
    }

    @Test
    public void testBuilderValidation() {
        // minTurns > maxTurns must fail
        assertThrows(IllegalArgumentException.class, () ->
                ConversationExtension.builder().minTurns(5).maxTurns(2).build());

        // extensionProbability out of range
        assertThrows(IllegalArgumentException.class, () ->
                ConversationExtension.builder().extensionProbability(-0.1).build());
        assertThrows(IllegalArgumentException.class, () ->
                ConversationExtension.builder().extensionProbability(1.1).build());

        // minTurns < 1
        assertThrows(IllegalArgumentException.class, () ->
                ConversationExtension.builder().minTurns(0).build());

        // maxTotalTokens < 1
        assertThrows(IllegalArgumentException.class, () ->
                ConversationExtension.builder().maxTotalTokens(0).build());
    }
}
