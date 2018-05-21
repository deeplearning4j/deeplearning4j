/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.corpora.treeparser;

import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;

import java.util.*;

public class HeadWordFinder {

    static final String[] head1 = {"ADJP JJ", "ADJP JJR", "ADJP JJS", "ADVP RB", "ADVP RBB", "LST LS", "NAC NNS",
                    "NAC NN", "NAC PRP", "NAC NNPS", "NAC NNP", "NX NNS", "NX NN", "NX PRP", "NX NNPS", "NX NNP",
                    "NP NNS", "NP NN", "NP PRP", "NP NNPS", "NP NNP", "NP POS", "NP $", "PP IN", "PP TO", "PP RP",
                    "PRT RP", "S VP", "S1 S", "SBAR IN", "SBAR WHNP", "SBARQ SQ", "SBARQ VP", "SINV VP", "SQ MD",
                    "SQ AUX", "VP VB", "VP VBZ", "VP VBP", "VP VBG", "VP VBN", "VP VBD", "VP AUX", "VP AUXG", "VP TO",
                    "VP MD", "WHADJP WRB", "WHADVP WRB", "WHNP WP", "WHNP WDT", "WHNP WP$", "WHPP IN", "WHPP TO"};

    static final String[] head2 = {"ADJP VBN", "ADJP RB", "NAC NP", "NAC CD", "NAC FW", "NAC ADJP", "NAC JJ", "NX NP",
                    "NX CD", "NX FW", "NX ADJP", "NX JJ", "NP CD", "NP ADJP", "NP JJ", "S SINV", "S SBARQ", "S X",
                    "PRT RB", "PRT IN", "SBAR WHADJP", "SBAR WHADVP", "SBAR WHPP", "SBARQ S", "SBARQ SINV", "SBARQ X",
                    "SINV SBAR", "SQ VP"};

    static final String[] term = {"AUX", "AUXG", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
                    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO",
                    "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", ".", ",", ":",
                    "-RRB-", "-LRB-", "``", "''", "EOS"};

    static final String[] punc = {"#", "$", ".", ",", ":", "-RRB-", "-LRB-", "``", "''"};

    static Set<String> headRules1;

    static Set<String> headRules2;

    static Set<String> terminals;

    static Set<String> punctuations;

    static Map<String, Integer> cache;

    static Boolean setsInitialized = false;

    static void buildSets() {
        synchronized (setsInitialized) {
            if (setsInitialized)
                return;
            HeadWordFinder.headRules1 = new HashSet<>(Arrays.asList(HeadWordFinder.head1));
            HeadWordFinder.headRules2 = new HashSet<>(Arrays.asList(HeadWordFinder.head2));
            HeadWordFinder.terminals = new HashSet<>(Arrays.asList(HeadWordFinder.term));
            HeadWordFinder.punctuations = new HashSet<>(Arrays.asList(HeadWordFinder.punc));
            HeadWordFinder.cache = new HashMap<>();
            setsInitialized = true;
        }
    }


    boolean includePPHead;

    public HeadWordFinder(boolean includePPHead) {
        this.includePPHead = includePPHead;
        HeadWordFinder.buildSets();
    }

    public HeadWordFinder() {
        this(false);
    }


    /**
     * Finds the bottom most head
     * @param parentNode the bottom most head
     * @return the bottom most head (no children) for the given parent
     */
    public Tree findHead(Tree parentNode) {
        Tree cursor = parentNode.getType().equals("TOP") ? parentNode.firstChild() : parentNode;

        while (cursor.children() != null && !cursor.children().isEmpty())
            cursor = findHead2(cursor);

        return cursor;
    }

    public Tree findHead2(Tree parentNode) {
        List<Tree> childNodes = parentNode.children();
        List<String> childTypes = new ArrayList<>(childNodes.size());

        String parentType = parentNode.getType();

        for (Tree childNode : childNodes)
            childTypes.add(childNode.getType());

        int headIndex = findHead3(parentType, childTypes);

        return childNodes.get(headIndex);
    }

    int findHead3(String lhs, List<String> rhss) {
        StringBuilder keyBuffer = new StringBuilder(lhs + " ->");
        for (String rhs : rhss)
            keyBuffer.append(" " + rhs);
        String key = keyBuffer.toString();

        synchronized (HeadWordFinder.cache) {
            if (cache.containsKey(key)) {
                return cache.get(key);
            }
        }

        int currentBestGuess = -1;
        int currentGuessUncertainty = 10;

        for (int current = 0; current < rhss.size(); current++) {
            String rhs = rhss.get(current);
            String rule = lhs + " " + rhs;

            if (currentGuessUncertainty >= 1 && headRules1.contains(rule)) {
                currentBestGuess = current;
                currentGuessUncertainty = 1;
            } else if (currentGuessUncertainty > 2 && lhs != null && lhs.equals(rhs)) {
                currentBestGuess = current;
                currentGuessUncertainty = 2;
            } else if (currentGuessUncertainty >= 3 && headRules2.contains(rule)) {
                currentBestGuess = current;
                currentGuessUncertainty = 3;
            } else if (currentGuessUncertainty >= 5 && !terminals.contains(rhs) && rhs != null && !rhs.equals("PP")) {
                currentBestGuess = current;
                currentGuessUncertainty = 5;
            } else if (currentGuessUncertainty >= 6 && !terminals.contains(rhs)) {
                currentBestGuess = current;
                currentGuessUncertainty = 6;
            } else if (currentGuessUncertainty >= 7) {
                currentBestGuess = current;
                currentGuessUncertainty = 7;
            }
        }

        synchronized (HeadWordFinder.cache) {
            cache.put(key, currentBestGuess);
        }

        return currentBestGuess;
    }


}
