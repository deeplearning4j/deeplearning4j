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
package com.atilika.kuromoji.viterbi;

import com.atilika.kuromoji.TokenizerBase.Mode;
import com.atilika.kuromoji.dict.CharacterDefinitions;
import com.atilika.kuromoji.dict.TokenInfoDictionary;
import com.atilika.kuromoji.dict.UnknownDictionary;
import com.atilika.kuromoji.dict.UserDictionary;
import com.atilika.kuromoji.trie.DoubleArrayTrie;

import java.util.ArrayList;
import java.util.List;

public class ViterbiBuilder {

    private final DoubleArrayTrie trie;
    private final TokenInfoDictionary dictionary;
    private final UnknownDictionary unknownDictionary;
    private final UserDictionary userDictionary;
    private final CharacterDefinitions characterDefinitions;
    private final boolean useUserDictionary;
    private boolean searchMode;

    /**
     * Constructor
     *
     * @param trie  trie with surface forms
     * @param dictionary  token info dictionary
     * @param unknownDictionary  unknown word dictionary
     * @param userDictionary  user dictionary
     * @param mode  tokenization {@link Mode mode}
     */
    public ViterbiBuilder(DoubleArrayTrie trie, TokenInfoDictionary dictionary, UnknownDictionary unknownDictionary,
                    UserDictionary userDictionary, Mode mode) {
        this.trie = trie;
        this.dictionary = dictionary;
        this.unknownDictionary = unknownDictionary;
        this.userDictionary = userDictionary;

        this.useUserDictionary = (userDictionary != null);

        if (mode == Mode.SEARCH || mode == Mode.EXTENDED) {
            searchMode = true;
        }
        this.characterDefinitions = unknownDictionary.getCharacterDefinition();
    }


    /**
     * Build lattice from input text
     *
     * @param text  source text for the lattice
     * @return built lattice, not null
     */
    public ViterbiLattice build(String text) {
        int textLength = text.length();
        ViterbiLattice lattice = new ViterbiLattice(textLength + 2);

        lattice.addBos();

        int unknownWordEndIndex = -1; // index of the last character of unknown word

        for (int startIndex = 0; startIndex < textLength; startIndex++) {
            // If no token ends where current token starts, skip this index
            if (lattice.tokenEndsWhereCurrentTokenStarts(startIndex)) {

                String suffix = text.substring(startIndex);
                boolean found = processIndex(lattice, startIndex, suffix);

                // In the case of normal mode, it doesn't process unknown word greedily.
                if (searchMode || unknownWordEndIndex <= startIndex) {

                    int[] categories = characterDefinitions.lookupCategories(suffix.charAt(0));

                    for (int i = 0; i < categories.length; i++) {
                        int category = categories[i];
                        unknownWordEndIndex = processUnknownWord(category, i, lattice, unknownWordEndIndex, startIndex,
                                        suffix, found);
                    }
                }
            }
        }

        if (useUserDictionary) {
            processUserDictionary(text, lattice);
        }

        lattice.addEos();

        return lattice;
    }

    private boolean processIndex(ViterbiLattice lattice, int startIndex, String suffix) {
        boolean found = false;
        for (int endIndex = 1; endIndex < suffix.length() + 1; endIndex++) {
            String prefix = suffix.substring(0, endIndex);
            int result = trie.lookup(prefix, 0, 0);

            if (result > 0) { // Found match in double array trie
                found = true; // Don't produce unknown word starting from this index
                for (int wordId : dictionary.lookupWordIds(result)) {
                    ViterbiNode node = new ViterbiNode(wordId, prefix, dictionary, startIndex, ViterbiNode.Type.KNOWN);
                    lattice.addNode(node, startIndex + 1, startIndex + 1 + endIndex);
                }
            } else if (result < 0) { // If result is less than zero, continue to next position
                break;
            }
        }
        return found;
    }

    private int processUnknownWord(int category, int i, ViterbiLattice lattice, int unknownWordEndIndex, int startIndex,
                    String suffix, boolean found) {
        int unknownWordLength = 0;
        int[] definition = characterDefinitions.lookupDefinition(category);

        if (definition[CharacterDefinitions.INVOKE] == 1 || found == false) {
            if (definition[CharacterDefinitions.GROUP] == 0) {
                unknownWordLength = 1;
            } else {
                unknownWordLength = 1;
                for (int j = 1; j < suffix.length(); j++) {
                    char c = suffix.charAt(j);

                    int[] categories = characterDefinitions.lookupCategories(c);

                    if (categories == null) {
                        break;
                    }

                    if (i < categories.length && category == categories[i]) {
                        unknownWordLength++;
                    } else {
                        break;
                    }
                }
            }
        }

        if (unknownWordLength > 0) {
            String unkWord = suffix.substring(0, unknownWordLength);
            int[] wordIds = unknownDictionary.lookupWordIds(category); // characters in input text are supposed to be the same

            for (int wordId : wordIds) {
                ViterbiNode node = new ViterbiNode(wordId, unkWord, unknownDictionary, startIndex,
                                ViterbiNode.Type.UNKNOWN);
                lattice.addNode(node, startIndex + 1, startIndex + 1 + unknownWordLength);
            }
            unknownWordEndIndex = startIndex + unknownWordLength;
        }

        return unknownWordEndIndex;
    }

    /**
     * Find token(s) in input text and set found token(s) in arrays as normal tokens
     *
     * @param text
     * @param lattice
     */
    private void processUserDictionary(final String text, ViterbiLattice lattice) {
        List<UserDictionary.UserDictionaryMatch> matches = userDictionary.findUserDictionaryMatches(text);

        for (UserDictionary.UserDictionaryMatch match : matches) {
            int wordId = match.getWordId();
            int index = match.getMatchStartIndex();
            int length = match.getMatchLength();

            String word = text.substring(index, index + length);

            ViterbiNode node = new ViterbiNode(wordId, word, userDictionary, index, ViterbiNode.Type.USER);
            int nodeStartIndex = index + 1;
            int nodeEndIndex = nodeStartIndex + length;

            lattice.addNode(node, nodeStartIndex, nodeEndIndex);

            if (isLatticeBrokenBefore(nodeStartIndex, lattice)) {
                repairBrokenLatticeBefore(lattice, index);
            }

            if (isLatticeBrokenAfter(nodeStartIndex + length, lattice)) {
                repairBrokenLatticeAfter(lattice, nodeEndIndex);
            }
        }
    }

    /**
     * Checks whether there exists any node in the lattice that connects to the newly inserted entry on the left side
     * (before the new entry).
     *
     * @param nodeIndex
     * @param lattice
     * @return whether the lattice has a node that ends at nodeIndex
     */
    private boolean isLatticeBrokenBefore(int nodeIndex, ViterbiLattice lattice) {
        ViterbiNode[][] nodeEndIndices = lattice.getEndIndexArr();

        return nodeEndIndices[nodeIndex] == null;
    }

    /**
     * Checks whether there exists any node in the lattice that connects to the newly inserted entry on the right side
     * (after the new entry).
     *
     * @param endIndex
     * @param lattice
     * @return whether the lattice has a node that starts at endIndex
     */
    private boolean isLatticeBrokenAfter(int endIndex, ViterbiLattice lattice) {
        ViterbiNode[][] nodeStartIndices = lattice.getStartIndexArr();

        return nodeStartIndices[endIndex] == null;
    }

    /**
     * Tries to repair the lattice by creating and adding an additional Viterbi node to the LEFT of the newly
     * inserted user dictionary entry by using the substring of the node in the lattice that overlaps the least
     *
     * @param lattice
     * @param index
     */
    private void repairBrokenLatticeBefore(ViterbiLattice lattice, int index) {
        ViterbiNode[][] nodeStartIndices = lattice.getStartIndexArr();

        for (int startIndex = index; startIndex > 0; startIndex--) {
            if (nodeStartIndices[startIndex] != null) {
                ViterbiNode glueBase = findGlueNodeCandidate(index, nodeStartIndices[startIndex], startIndex);
                if (glueBase != null) {
                    int length = index + 1 - startIndex;
                    String surface = glueBase.getSurface().substring(0, length);
                    ViterbiNode glueNode = createGlueNode(startIndex, glueBase, surface);
                    lattice.addNode(glueNode, startIndex, startIndex + glueNode.getSurface().length());
                    return;
                }
            }
        }
    }

    /**
     * Tries to repair the lattice by creating and adding an additional Viterbi node to the RIGHT of the newly
     * inserted user dictionary entry by using the substring of the node in the lattice that overlaps the least
     *  @param lattice
     * @param nodeEndIndex
     */
    private void repairBrokenLatticeAfter(ViterbiLattice lattice, int nodeEndIndex) {
        ViterbiNode[][] nodeEndIndices = lattice.getEndIndexArr();

        for (int endIndex = nodeEndIndex + 1; endIndex < nodeEndIndices.length; endIndex++) {
            if (nodeEndIndices[endIndex] != null) {
                ViterbiNode glueBase = findGlueNodeCandidate(nodeEndIndex, nodeEndIndices[endIndex], endIndex);
                if (glueBase != null) {
                    int delta = endIndex - nodeEndIndex;
                    String glueBaseSurface = glueBase.getSurface();
                    String surface = glueBaseSurface.substring(glueBaseSurface.length() - delta);
                    ViterbiNode glueNode = createGlueNode(nodeEndIndex, glueBase, surface);
                    lattice.addNode(glueNode, nodeEndIndex, nodeEndIndex + glueNode.getSurface().length());
                    return;
                }
            }
        }
    }

    /**
     * Tries to locate a candidate for a "glue" node that repairs the broken lattice by looking at all nodes at the
     * current index.
     *
     * @param index
     * @param latticeNodes
     * @param startIndex
     * @return new ViterbiNode that can be inserted to glue the graph if such a node exists, else null
     */
    private ViterbiNode findGlueNodeCandidate(int index, ViterbiNode[] latticeNodes, int startIndex) {
        List<ViterbiNode> candidates = new ArrayList<>();

        for (ViterbiNode viterbiNode : latticeNodes) {
            if (viterbiNode != null) {
                candidates.add(viterbiNode);
            }
        }
        if (!candidates.isEmpty()) {
            ViterbiNode glueBase = null;
            int length = index + 1 - startIndex;
            for (ViterbiNode candidate : candidates) {
                if (isAcceptableCandidate(length, glueBase, candidate)) {
                    glueBase = candidate;
                }
            }
            if (glueBase != null) {
                return glueBase;
            }
        }
        return null;
    }

    /**
     * Check whether a candidate for a glue node is acceptable.
     * The candidate should be as short as possible, but long enough to overlap with the inserted user entry
     *
     * @param targetLength
     * @param glueBase
     * @param candidate
     * @return whether candidate is acceptable
     */
    private boolean isAcceptableCandidate(int targetLength, ViterbiNode glueBase, ViterbiNode candidate) {
        return (glueBase == null || candidate.getSurface().length() < glueBase.getSurface().length())
                        && candidate.getSurface().length() >= targetLength;
    }

    /**
     * Create a glue node to be inserted based on ViterbiNode already in the lattice.
     * The new node takes the same parameters as the node it is based on, but the word is truncated to match the
     * hole in the lattice caused by the new user entry
     *
     * @param startIndex
     * @param glueBase
     * @param surface
     * @return new ViterbiNode to be inserted as glue into the lattice
     */
    private ViterbiNode createGlueNode(int startIndex, ViterbiNode glueBase, String surface) {
        return new ViterbiNode(glueBase.getWordId(), surface, glueBase.getLeftId(), glueBase.getRightId(),
                        glueBase.getWordCost(), startIndex, ViterbiNode.Type.INSERTED);
    }
}
