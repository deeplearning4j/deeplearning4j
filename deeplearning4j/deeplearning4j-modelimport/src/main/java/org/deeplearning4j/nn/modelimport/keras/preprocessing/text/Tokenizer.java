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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    private Integer documentCount;



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


    public static String[] textToWordSequence(String text, String filters, boolean lower, String split) {
        if (lower)
            text = text.toLowerCase();

        for (String filter: filters.split("")) {
            text = text.replace(filter, split);
        }
        String[] sequences = text.split(split);
        List<String> seqList = Arrays.asList(sequences);
        seqList.removeAll(Arrays.asList("", null));

        return seqList.toArray(new String[seqList.size()]);
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

    public Integer[][] textsToSequences(String[] texts) {
        Integer oovTokenIndex  = wordIndex.get(outOfVocabularyToken);
        String[] wordSequence;
        ArrayList<Integer[]> sequences = new ArrayList<>();
        for (String text: texts) {
            if (charLevel) {
                if (lower) {
                    text = text.toLowerCase();
                }
                wordSequence = text.split("");
            } else {
                wordSequence = textToWordSequence(text, filters, lower, split);
            }
            ArrayList<Integer> indexVector = new ArrayList<>();
            for (String word: wordSequence) {
                if (wordIndex.containsKey(word)) {
                    int index = wordIndex.get(word);
                    if (numWords != null && index >= numWords) {
                        if (oovTokenIndex != null)
                            indexVector.add(oovTokenIndex);
                    } else {
                        indexVector.add(index);
                    }
                } else if (oovTokenIndex != null) {
                    indexVector.add(oovTokenIndex);
                }
            }
            Integer[] indices = indexVector.toArray(new Integer[indexVector.size()]);
            sequences.add(indices);
        }
        return sequences.toArray(new Integer[sequences.size()][]);
    }


    public String[] sequencesToTexts(Integer[][] sequences) {
        Integer oovTokenIndex  = wordIndex.get(outOfVocabularyToken);
        ArrayList<String> texts = new ArrayList<>();
        for (Integer[] sequence: sequences) {
            ArrayList<String> wordVector = new ArrayList<>();
            for (Integer index: sequence) {
                if (indexWord.containsKey(index)) {
                    String word = indexWord.get(index);
                    if (numWords != null && index >= numWords) {
                        if (oovTokenIndex != null) {
                            wordVector.add(indexWord.get(oovTokenIndex));
                        } else {
                            wordVector.add(word);
                        }
                    }
                } else if (oovTokenIndex != null) {
                    wordVector.add(indexWord.get(oovTokenIndex));
                }
            }
            StringBuilder builder = new StringBuilder();
            for (String word: wordVector) {
                builder.append(word + split);
            }
            String text = builder.toString();
            texts.add(text);
        }
        return texts.toArray(new String[texts.size()]);
    }


    public INDArray textsToMatrix(String[] texts, TokenizerMode mode) {
        Integer[][] sequences = textsToSequences(texts);
        return sequencesToMatrix(sequences, mode);
    }

    public INDArray sequencesToMatrix(Integer[][] sequences, TokenizerMode mode) {
        if (numWords == null) {
            if (!wordIndex.isEmpty()) {
                numWords = wordIndex.size();
            } else {
                throw new IllegalArgumentException("Either specify numWords argument" +
                        "or fit Tokenizer on data first, i.e. by using fitOnTexts");
            }
        }
        if (mode.equals(TokenizerMode.TFIDF) && documentCount == null) {
            throw new IllegalArgumentException("To use TFIDF mode you need to" +
                    "fit the Tokenizer instance with fitOnTexts first.");
        }
        INDArray x = Nd4j.zeros(sequences.length, numWords);
        for (int i=0; i< sequences.length; i++) {
            Integer[] sequence = sequences[i];
            if (sequence == null)
                continue;
            Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
            for (int j: sequence) {
                if (j >= numWords)
                    continue;
                counts.put(j, counts.get(j) + 1);
            }
            for (int j: counts.keySet()) {
                int count = counts.get(j);
                switch (mode) {
                    case COUNT:
                        x.put(i, j, count);
                        break;
                    case FREQ:
                        x.put(i, j, count / sequence.length);
                        break;
                    case BINARY:
                        x.put(i, j, 1);
                        break;
                    case TFIDF:
                        double tf = 1.0 + Math.log(count);
                        int index = indexDocs.containsKey(j) ? indexDocs.get(j) : 0;
                        double idf = Math.log(1 + documentCount / (1.0 + index));
                        x.put(i, j, tf * idf);
                        break;
                }
            }
        }
        return x;
    }

}
