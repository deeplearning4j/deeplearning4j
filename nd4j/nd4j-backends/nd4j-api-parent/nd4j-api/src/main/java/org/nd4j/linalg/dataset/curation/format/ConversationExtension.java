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

package org.nd4j.linalg.dataset.curation.format;

import java.io.Serializable;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Synthesizes multi-turn conversations from single-turn instruction examples.
 *
 * Given a pool of single-turn (instruction, response) pairs represented as
 * sequences of two {@link ConversationTurn} objects (user turn followed by
 * assistant turn), this utility randomly merges them into multi-turn
 * conversations. This improves model chat capabilities without requiring
 * actual multi-turn training data.
 *
 * <p>The input convention is a flat list where each consecutive pair of turns
 * represents one single-turn example: {@code [user, assistant, user, assistant, ...]}.
 * See {@link #fromPairs(List)} for a convenient constructor from raw string pairs.
 *
 * <p>Example: Given single-turn pairs A, B, C the extension might produce:
 * <pre>
 *   Conversation 1: [userA, assistantA, userC, assistantC]  (2-turn)
 *   Conversation 2: [userB, assistantB]                     (1-turn)
 * </pre>
 *
 * <p>Inspired by Unsloth's {@code conversation_extension} augmentation strategy.
 */
public class ConversationExtension implements Serializable {
    private static final long serialVersionUID = 1L;

    /** Token-length estimate divisor: 1 token ~ 4 characters (rough approximation). */
    private static final int CHARS_PER_TOKEN = 4;

    private final int minTurns;
    private final int maxTurns;
    private final double extensionProbability;
    private final int maxTotalTokens;
    private final boolean addSystemMessage;
    private final String systemMessage;
    private final long seed;

    private ConversationExtension(Builder builder) {
        this.minTurns = builder.minTurns;
        this.maxTurns = builder.maxTurns;
        this.extensionProbability = builder.extensionProbability;
        this.maxTotalTokens = builder.maxTotalTokens;
        this.addSystemMessage = builder.addSystemMessage;
        this.systemMessage = builder.systemMessage;
        this.seed = builder.seed;
    }

    /**
     * Extend a flat list of single-turn pairs into multi-turn conversations.
     *
     * <p>Input is expected to be pairs of turns: user turn at even indices,
     * assistant turn at odd indices. Every two consecutive turns form one
     * single-turn exchange. If the input has an odd number of turns the last
     * unpaired turn is ignored.
     *
     * @param singleTurns flat list of alternating user/assistant turns
     * @return list of conversations, each being a {@code List<ConversationTurn>}
     */
    public List<List<ConversationTurn>> extend(List<ConversationTurn> singleTurns) {
        if (singleTurns == null || singleTurns.isEmpty()) {
            return Collections.emptyList();
        }

        // Build a list of single-turn pairs (each pair = one exchange)
        List<List<ConversationTurn>> pairs = buildPairs(singleTurns);
        if (pairs.isEmpty()) {
            return Collections.emptyList();
        }

        Random rng = seed >= 0 ? new Random(seed) : new Random();

        // Shuffle the pool so merges are random across the dataset
        List<List<ConversationTurn>> pool = new ArrayList<>(pairs);
        Collections.shuffle(pool, rng);

        List<List<ConversationTurn>> result = new ArrayList<>();

        int idx = 0;
        while (idx < pool.size()) {
            // Determine how many turns to draw for this conversation
            int numTurns;
            if (extensionProbability <= 0.0 || pool.size() - idx == 1) {
                numTurns = 1;
            } else if (extensionProbability >= 1.0) {
                // Always extend as much as possible up to maxTurns
                numTurns = minTurns + rng.nextInt(Math.max(1, maxTurns - minTurns + 1));
                numTurns = Math.min(numTurns, pool.size() - idx);
            } else {
                // Probabilistic: extend with extensionProbability
                boolean doExtend = rng.nextDouble() < extensionProbability;
                if (doExtend) {
                    int range = maxTurns - minTurns + 1;
                    numTurns = minTurns + rng.nextInt(Math.max(1, range));
                    numTurns = Math.min(numTurns, pool.size() - idx);
                } else {
                    numTurns = 1;
                }
            }

            // Consume numTurns pairs and merge into one conversation
            List<ConversationTurn> conversation = new ArrayList<>();

            // Optionally prepend a system message
            if (addSystemMessage && systemMessage != null && !systemMessage.isEmpty()) {
                conversation.add(ConversationTurn.system(systemMessage));
            }

            // Accumulate pairs respecting the token budget
            int tokenCount = estimateTokens(conversation);
            int pairsAdded = 0;
            for (int t = 0; t < numTurns && idx < pool.size(); t++) {
                List<ConversationTurn> pair = pool.get(idx);
                int pairTokens = estimateTokens(pair);
                if (tokenCount + pairTokens > maxTotalTokens && pairsAdded > 0) {
                    // Would exceed budget; stop adding turns to this conversation
                    break;
                }
                conversation.addAll(pair);
                tokenCount += pairTokens;
                pairsAdded++;
                idx++;
            }

            result.add(conversation);
        }

        return result;
    }

    /**
     * Extend single-turn pairs and format each resulting conversation using
     * the supplied {@link ChatTemplate}.
     *
     * @param singleTurns single-turn examples (alternating user/assistant turns)
     * @param template    chat template to apply during formatting
     * @return formatted multi-turn examples
     */
    public List<FormattedExample> extendAndFormat(List<ConversationTurn> singleTurns, ChatTemplate template) {
        List<List<ConversationTurn>> conversations = extend(singleTurns);
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(template)
                .build();

        List<FormattedExample> formatted = new ArrayList<>(conversations.size());
        for (List<ConversationTurn> conversation : conversations) {
            formatted.add(formatter.format(conversation));
        }
        return formatted;
    }

    /**
     * Create a flat list of alternating user/assistant {@link ConversationTurn} objects
     * from raw instruction/response string pairs.
     *
     * <p>Each map entry's key is the instruction (user turn) and value is the response
     * (assistant turn).
     *
     * @param pairs list of (instruction, response) entries
     * @return flat list of alternating user/assistant turns suitable for {@link #extend}
     */
    public static List<ConversationTurn> fromPairs(List<Map.Entry<String, String>> pairs) {
        if (pairs == null || pairs.isEmpty()) {
            return Collections.emptyList();
        }
        List<ConversationTurn> turns = new ArrayList<>(pairs.size() * 2);
        for (Map.Entry<String, String> entry : pairs) {
            turns.add(ConversationTurn.user(entry.getKey()));
            turns.add(ConversationTurn.assistant(entry.getValue()));
        }
        return turns;
    }

    /**
     * Convenience overload: create a flat list of turns from parallel lists of
     * instructions and responses.
     *
     * @param instructions list of user instructions
     * @param responses    corresponding assistant responses (same length)
     * @return flat list of alternating user/assistant turns
     * @throws IllegalArgumentException if the lists have different sizes
     */
    public static List<ConversationTurn> fromPairs(List<String> instructions, List<String> responses) {
        if (instructions == null || responses == null) {
            return Collections.emptyList();
        }
        if (instructions.size() != responses.size()) {
            throw new IllegalArgumentException(
                    "instructions and responses must have the same size, got "
                            + instructions.size() + " vs " + responses.size());
        }
        List<Map.Entry<String, String>> entries = new ArrayList<>(instructions.size());
        for (int i = 0; i < instructions.size(); i++) {
            entries.add(new AbstractMap.SimpleImmutableEntry<>(instructions.get(i), responses.get(i)));
        }
        return fromPairs(entries);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public int getMinTurns() { return minTurns; }
    public int getMaxTurns() { return maxTurns; }
    public double getExtensionProbability() { return extensionProbability; }
    public int getMaxTotalTokens() { return maxTotalTokens; }
    public boolean isAddSystemMessage() { return addSystemMessage; }
    public String getSystemMessage() { return systemMessage; }
    public long getSeed() { return seed; }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int minTurns = 1;
        private int maxTurns = 5;
        private double extensionProbability = 0.5;
        private int maxTotalTokens = 4096;
        private boolean addSystemMessage = false;
        private String systemMessage = null;
        private long seed = -1L;

        /** Minimum number of single-turn exchanges per merged conversation. Default: 1. */
        public Builder minTurns(int minTurns) {
            if (minTurns < 1) throw new IllegalArgumentException("minTurns must be >= 1");
            this.minTurns = minTurns;
            return this;
        }

        /** Maximum number of single-turn exchanges per merged conversation. Default: 5. */
        public Builder maxTurns(int maxTurns) {
            if (maxTurns < 1) throw new IllegalArgumentException("maxTurns must be >= 1");
            this.maxTurns = maxTurns;
            return this;
        }

        /**
         * Probability that a single-turn example is extended into a multi-turn
         * conversation. 0.0 means never extend; 1.0 means always extend. Default: 0.5.
         */
        public Builder extensionProbability(double extensionProbability) {
            if (extensionProbability < 0.0 || extensionProbability > 1.0) {
                throw new IllegalArgumentException("extensionProbability must be in [0, 1]");
            }
            this.extensionProbability = extensionProbability;
            return this;
        }

        /**
         * Maximum estimated total token count for a merged conversation.
         * Token count is approximated as character count / 4. Default: 4096.
         */
        public Builder maxTotalTokens(int maxTotalTokens) {
            if (maxTotalTokens < 1) throw new IllegalArgumentException("maxTotalTokens must be >= 1");
            this.maxTotalTokens = maxTotalTokens;
            return this;
        }

        /**
         * Whether to prepend a system message at the start of each conversation.
         * Requires {@link #systemMessage(String)} to also be set. Default: false.
         */
        public Builder addSystemMessage(boolean addSystemMessage) {
            this.addSystemMessage = addSystemMessage;
            return this;
        }

        /** System message text to prepend when {@link #addSystemMessage(boolean)} is true. */
        public Builder systemMessage(String systemMessage) {
            this.systemMessage = systemMessage;
            return this;
        }

        /**
         * Random seed for reproducibility. Use -1 (the default) for a non-deterministic
         * seed based on the current time.
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public ConversationExtension build() {
            if (minTurns > maxTurns) {
                throw new IllegalArgumentException(
                        "minTurns (" + minTurns + ") must be <= maxTurns (" + maxTurns + ")");
            }
            return new ConversationExtension(this);
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Split a flat alternating user/assistant turn list into individual exchange pairs.
     * Turns at even indices are user turns; turns at odd indices are assistant turns.
     * Unpaired trailing turns are discarded.
     */
    private static List<List<ConversationTurn>> buildPairs(List<ConversationTurn> turns) {
        List<List<ConversationTurn>> pairs = new ArrayList<>();
        for (int i = 0; i + 1 < turns.size(); i += 2) {
            List<ConversationTurn> pair = new ArrayList<>(2);
            pair.add(turns.get(i));
            pair.add(turns.get(i + 1));
            pairs.add(pair);
        }
        return pairs;
    }

    /**
     * Estimate token count for a list of turns using the 4-chars-per-token heuristic.
     */
    private static int estimateTokens(List<ConversationTurn> turns) {
        int chars = 0;
        for (ConversationTurn turn : turns) {
            chars += turn.getContent().length();
        }
        return (chars + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN;
    }
}
