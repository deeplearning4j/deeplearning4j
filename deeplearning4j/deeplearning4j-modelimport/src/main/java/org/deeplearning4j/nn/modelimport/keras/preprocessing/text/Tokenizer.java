/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.modelimport.keras.preprocessing.text;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import lombok.Data;

import java.util.*;

/**
 * Keras text tokenizer.
 *
 * @author Max Pumperla
 */
@Data
public class Tokenizer {

    private static final String DEFAULT_FILTER = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n";
    private static final String DEFAULT_SPLIT = " ";

    private Integer numWords;
    private String filters;
    private boolean lower;
    private String split;
    private boolean charLevel;
    private String outOfVocabularyToken;

    private Map<String, Integer> wordCounts = new HashMap<String, Integer>();
    private Map<String, Integer> wordDocs = new HashMap<String, Integer>();
    private Map<String, Integer> wordIndex = new HashMap<String, Integer>();
    private Map<Integer, String> indexWord = new HashMap<Integer, String>();
    private Map<Integer, Integer> indexDocs = new HashMap<Integer, Integer>();
    private int documentCount = 0;



    /**
     * Create a Keras Tokenizer instance
     *
     * @param numWords             The maximum vocabulary size, can be null
     * @param filters              Characters to filter
     * @param lower                whether to lowercase input or not
     * @param split                by which string to split words (usually single space)
     * @param charLevel            whether to operate on character- or word-level
     * @param outOfVocabularyToken replace items outside the vocabulary by this token
     */
    public Tokenizer(Integer numWords, String filters, boolean lower, String split, boolean charLevel,
                     String outOfVocabularyToken) {

        this.numWords = numWords;
        this.filters = filters;
        this.lower = lower;
        this.split = split;
        this.charLevel = charLevel;
        this.outOfVocabularyToken = outOfVocabularyToken;
    }

    public Tokenizer() {
        this(null, DEFAULT_FILTER, true, DEFAULT_SPLIT, false, null);
    }


    public String[] textToWordSequence(String text, String filters, boolean lower, String split) {
        return new String[]{"f", "o", "o"}; // TODO
    }

    public void fitOnTexts(String[] texts) {
        String[] sequence;
        for (String text : texts) {
            documentCount += 1;
            if (charLevel) {
                if (lower)
                    text = text.toLowerCase();
                sequence = text.split("");
            } else {
                sequence = textToWordSequence(text, filters, lower, split);
            }
            for (String word : sequence) {
                if (wordCounts.containsKey(word))
                    wordCounts.put(word, (int) wordCounts.get(word) + 1);
                else
                    wordCounts.put(word, 1);
            }
            Set<String> sequenceSet = new HashSet<>(Arrays.asList(sequence));
            for (String word: sequenceSet) {
                if (wordDocs.containsKey(word))
                    wordDocs.put(word, (int) wordDocs.get(word) + 1);
                else
                    wordDocs.put(word, 1);
            }
        }
        Ordering valueComparator = Ordering.natural().reverse().onResultOf(Functions.forMap(wordDocs));
        Map<String, Integer> sortedWordCounts = ImmutableSortedMap.copyOf(wordDocs, valueComparator);
        ArrayList<String> sortedVocabulary = new ArrayList<String>();
        if (outOfVocabularyToken != null)
            sortedVocabulary.add(outOfVocabularyToken);
        for (String word: sortedWordCounts.keySet()) {
            sortedVocabulary.add(word);
        }

        for (int i = 1; i <= sortedVocabulary.size() + 1; i++)
            wordIndex.put(sortedVocabulary.get(i), i);

        for(String key : wordIndex.keySet()){
            indexWord.put(wordIndex.get(key), key);
        }

        for (String key: wordDocs.keySet())
            indexDocs.put(wordIndex.get(key), wordDocs.get(key));
    }

    public void fitOnSequences(Integer[][] sequences) {
        documentCount += 1;
        for (Integer[] sequence: sequences) {
            Set<Integer> sequenceSet = new HashSet<>(Arrays.asList(sequence));
            for (Integer index: sequenceSet)
                indexDocs.put(index, indexDocs.get(index) + 1);
        }
    }


}
