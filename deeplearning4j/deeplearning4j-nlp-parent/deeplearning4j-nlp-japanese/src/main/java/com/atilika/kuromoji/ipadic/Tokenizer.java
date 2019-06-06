/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.atilika.kuromoji.ipadic;

import com.atilika.kuromoji.TokenizerBase;
import com.atilika.kuromoji.dict.*;
import com.atilika.kuromoji.ipadic.compile.DictionaryEntry;
import com.atilika.kuromoji.trie.DoubleArrayTrie;
import com.atilika.kuromoji.util.FileResourceResolver;
import com.atilika.kuromoji.viterbi.TokenFactory;
import com.atilika.kuromoji.viterbi.ViterbiNode;

import java.util.ArrayList;
import java.util.List;

/**
 * A tokenizer based on the IPADIC dictionary
 * <p>
 * See {@link Token} for details on the morphological features produced by this tokenizer
 * <p>
 * The following code example demonstrates how to use the Kuromoji tokenizer:
 * <pre>{@code
 * package com.atilika.kuromoji.example;
 * import com.atilika.kuromoji.ipadic.Token;
 * import com.atilika.kuromoji.ipadic.Tokenizer;
 * import java.util.List;
 *
 * public class KuromojiExample {
 *     public static void main(String[] args) {
 *         Tokenizer tokenizer = new Tokenizer() ;
 *         List<Token> tokens = tokenizer.tokenize("お寿司が食べたい。");
 *         for (Token token : tokens) {
 *             System.out.println(token.getSurface() + "\t" + token.getAllFeatures());
 *         }
 *     }
 * }
 * }
 * </pre>
 */
public class Tokenizer extends TokenizerBase {

    /**
     * Construct a default tokenizer
     */
    public Tokenizer() {
        this(new Builder());
    }

    /**
     * Construct a customized tokenizer
     * <p>
     * See {@see com.atilika.kuromoji.ipadic.Tokenizer#Builder}
     */
    private Tokenizer(Builder builder) {
        configure(builder);
    }

    /**
     * Tokenizes the provided text and returns a list of tokens with various feature information
     * <p>
     * This method is thread safe
     *
     * @param text  text to tokenize
     * @return list of Token, not null
     */
    @Override
    public List<Token> tokenize(String text) {
        return createTokenList(text);
    }

    /**
     * Builder class for creating a customized tokenizer instance
     */
    public static class Builder extends TokenizerBase.Builder {

        private static final int DEFAULT_KANJI_LENGTH_THRESHOLD = 2;
        private static final int DEFAULT_OTHER_LENGTH_THRESHOLD = 7;
        private static final int DEFAULT_KANJI_PENALTY = 3000;
        private static final int DEFAULT_OTHER_PENALTY = 1700;

        private int kanjiPenaltyLengthTreshold = DEFAULT_KANJI_LENGTH_THRESHOLD;
        private int kanjiPenalty = DEFAULT_KANJI_PENALTY;
        private int otherPenaltyLengthThreshold = DEFAULT_OTHER_LENGTH_THRESHOLD;
        private int otherPenalty = DEFAULT_OTHER_PENALTY;

        private boolean nakaguroSplit = false;

        /**
         * Creates a default builder
         */
        public Builder() {
            totalFeatures = DictionaryEntry.TOTAL_FEATURES;
            readingFeature = DictionaryEntry.READING_FEATURE;
            partOfSpeechFeature = DictionaryEntry.PART_OF_SPEECH_FEATURE;

            tokenFactory = new TokenFactory<Token>() {
                @Override
                public Token createToken(int wordId, String surface, ViterbiNode.Type type, int position,
                                Dictionary dictionary) {
                    return new Token(wordId, surface, type, position, dictionary);
                }
            };
        }

        /**
         * Sets the tokenization mode
         * <p>
         * The tokenization mode defines how Available modes are as follows:
         * <ul>
         * <li>{@link com.atilika.kuromoji.TokenizerBase.Mode#NORMAL} - The default mode
         * <li>{@link com.atilika.kuromoji.TokenizerBase.Mode#SEARCH} - Uses a heuristic to segment compound nouns (複合名詞) into their parts
         * <li>{@link com.atilika.kuromoji.TokenizerBase.Mode#EXTENDED} - Same as SEARCH, but emits unigram tokens for unknown terms
         * </ul>
         * See {@link #kanjiPenalty} and {@link #otherPenalty} for how to adjust costs used by SEARCH and EXTENDED mode
         *
         * @param mode  tokenization mode
         * @return this builder, not null
         */
        public Builder mode(Mode mode) {
            this.mode = mode;
            return this;
        }

        /**
         * Sets a custom kanji penalty
         * <p>
         * This is an expert feature used with {@link com.atilika.kuromoji.TokenizerBase.Mode#SEARCH} and {@link com.atilika.kuromoji.TokenizerBase.Mode#EXTENDED} modes that sets a length threshold and an additional costs used when running the Viterbi search.
         * The additional cost is applicable for kanji candidate tokens longer than the length threshold specified.
         * <p>
         * This is an expert feature and you usually would not need to change this.
         *
         * @param lengthThreshold  length threshold applicable for this penalty
         * @param penalty  cost added to Viterbi nodes for long kanji candidate tokens
         * @return this builder, not null
         */
        public Builder kanjiPenalty(int lengthThreshold, int penalty) {
            this.kanjiPenaltyLengthTreshold = lengthThreshold;
            this.kanjiPenalty = penalty;
            return this;
        }

        /**
         * Sets a custom non-kanji penalty
         * <p>
         * This is an expert feature used with {@link com.atilika.kuromoji.TokenizerBase.Mode#SEARCH} and {@link com.atilika.kuromoji.TokenizerBase.Mode#EXTENDED} modes that sets a length threshold and an additional costs used when running the Viterbi search.
         * The additional cost is applicable for non-kanji candidate tokens longer than the length threshold specified.
         * <p>
         * This is an expert feature and you usually would not need to change this.
         *
         * @param lengthThreshold  length threshold applicable for this penalty
         * @param penalty  cost added to Viterbi nodes for long non-kanji candidate tokens
         * @return this builder, not null
         */
        public Builder otherPenalty(int lengthThreshold, int penalty) {
            this.otherPenaltyLengthThreshold = lengthThreshold;
            this.otherPenalty = penalty;
            return this;
        }

        /**
         * Predictate that splits unknown words on the middle dot character (U+30FB KATAKANA MIDDLE DOT)
         * <p>
         * This feature is off by default.
         * This is an expert feature sometimes used with {@link com.atilika.kuromoji.TokenizerBase.Mode#SEARCH} and {@link com.atilika.kuromoji.TokenizerBase.Mode#EXTENDED} mode.
         *
         * @param split  predicate to indicate split on middle dot
         * @return this builder, not null
         */
        public Builder isSplitOnNakaguro(boolean split) {
            this.nakaguroSplit = split;
            return this;
        }

        /**
         * Creates the custom tokenizer instance
         *
         * @return tokenizer instance, not null
         */
        @Override
        public Tokenizer build() {
            return new Tokenizer(this);
        }

        @Override
        protected void loadDictionaries() {
            penalties = new ArrayList<>();
            penalties.add(kanjiPenaltyLengthTreshold);
            penalties.add(kanjiPenalty);
            penalties.add(otherPenaltyLengthThreshold);
            penalties.add(otherPenalty);

            //            resolver = new SimpleResourceResolver(this.getClass());
            resolver = new FileResourceResolver();

            try {
                doubleArrayTrie = DoubleArrayTrie.newInstance(resolver);
                connectionCosts = ConnectionCosts.newInstance(resolver);
                tokenInfoDictionary = TokenInfoDictionary.newInstance(resolver);
                characterDefinitions = CharacterDefinitions.newInstance(resolver);

                if (nakaguroSplit) {
                    characterDefinitions.setCategories('・', new String[] {"SYMBOL"});
                }

                unknownDictionary = UnknownDictionary.newInstance(resolver, characterDefinitions, totalFeatures);
                insertedDictionary = new InsertedDictionary(totalFeatures);
            } catch (Exception ouch) {
                throw new RuntimeException("Could not load dictionaries.", ouch);
            }
        }
    }
}
