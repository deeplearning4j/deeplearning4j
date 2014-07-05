package org.deeplearning4j.text.treeparser;

/**
 * Copyright (c) 2007-2008, Regents of the University of Colorado
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 * Neither the name of the University of Colorado at Boulder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


import org.deeplearning4j.rntn.Tree;

/**
 * <br>
 * Copyright (c) 2007-2008, Regents of the University of Colorado <br>
 * All rights reserved.
 *
 *
 * @author Philipp Wetzler
 * @author Adam Gibson
 *
 *
 *         The rules for finding the head word were adapted from the ASSERT system for semantic role
 *         labeling. For more information see:
 *
 *         http://cemantix.org/
 *
 *         and
 *
 *         Shallow Semantic Parsing using Support Vector Machines Sameer S. Pradhan, Wayne Ward,
 *         Kadri Hacioglu, James H. Martin, Daniel Jurafsky, in Proceedings of the Human Language
 *         Technology Conference/North American chapter of the Association for Computational
 *         Linguistics annual meeting (HLT/NAACL-2004), Boston, MA, May 2-7, 2004
 *
 *
 *
 *         Adapted from the HeadWordExtractor in Cleartk
 *
 */

public class HeadWordFinder  {

    static final String[] head1 = {
            "ADJP JJ",
            "ADJP JJR",
            "ADJP JJS",
            "ADVP RB",
            "ADVP RBB",
            "LST LS",
            "NAC NNS",
            "NAC NN",
            "NAC PRP",
            "NAC NNPS",
            "NAC NNP",
            "NX NNS",
            "NX NN",
            "NX PRP",
            "NX NNPS",
            "NX NNP",
            "NP NNS",
            "NP NN",
            "NP PRP",
            "NP NNPS",
            "NP NNP",
            "NP POS",
            "NP $",
            "PP IN",
            "PP TO",
            "PP RP",
            "PRT RP",
            "S VP",
            "S1 S",
            "SBAR IN",
            "SBAR WHNP",
            "SBARQ SQ",
            "SBARQ VP",
            "SINV VP",
            "SQ MD",
            "SQ AUX",
            "VP VB",
            "VP VBZ",
            "VP VBP",
            "VP VBG",
            "VP VBN",
            "VP VBD",
            "VP AUX",
            "VP AUXG",
            "VP TO",
            "VP MD",
            "WHADJP WRB",
            "WHADVP WRB",
            "WHNP WP",
            "WHNP WDT",
            "WHNP WP$",
            "WHPP IN",
            "WHPP TO" };

    static final String[] head2 = {
            "ADJP VBN",
            "ADJP RB",
            "NAC NP",
            "NAC CD",
            "NAC FW",
            "NAC ADJP",
            "NAC JJ",
            "NX NP",
            "NX CD",
            "NX FW",
            "NX ADJP",
            "NX JJ",
            "NP CD",
            "NP ADJP",
            "NP JJ",
            "S SINV",
            "S SBARQ",
            "S X",
            "PRT RB",
            "PRT IN",
            "SBAR WHADJP",
            "SBAR WHADVP",
            "SBAR WHPP",
            "SBARQ S",
            "SBARQ SINV",
            "SBARQ X",
            "SINV SBAR",
            "SQ VP" };

    static final String[] term = {
            "AUX",
            "AUXG",
            "CC",
            "CD",
            "DT",
            "EX",
            "FW",
            "IN",
            "JJ",
            "JJR",
            "JJS",
            "LS",
            "MD",
            "NN",
            "NNS",
            "NNP",
            "NNPS",
            "PDT",
            "POS",
            "PRP",
            "PRP$",
            "RB",
            "RBR",
            "RBS",
            "RP",
            "SYM",
            "TO",
            "UH",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "WDT",
            "WP",
            "WP$",
            "WRB",
            "#",
            "$",
            ".",
            ",",
            ":",
            "-RRB-",
            "-LRB-",
            "``",
            "''",
            "EOS" };

    static final String[] punc = { "#", "$", ".", ",", ":", "-RRB-", "-LRB-", "``", "''" };

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

    public HeadWordFinder( boolean includePPHead) {
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
        Tree cursor = parentNode;

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
        StringBuffer keyBuffer = new StringBuffer(lhs + " ->");
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
            } else if (currentGuessUncertainty >= 5 && !terminals.contains(rhs) && rhs != null
                    && !rhs.equals("PP")) {
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