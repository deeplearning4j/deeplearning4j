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

import lombok.Data;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils.parseJsonString;

/**
 * Java port of Keras' text tokenizer, see https://keras.io/preprocessing/text/ for more information.
 *
 * @author Max Pumperla
 */
@Data
public class KerasTokenizer {

    private static final String DEFAULT_FILTER = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n";
    private static final String DEFAULT_SPLIT = " ";

    private Integer numWords;
    private String filters;
    private boolean lower;
    private String split;
    private boolean charLevel;
    private String outOfVocabularyToken;

    private Map<String, Integer> wordCounts = new HashMap<>();
    private HashMap<String, Integer> wordDocs = new HashMap<>();
    private Map<String, Integer> wordIndex = new HashMap<>();
    private Map<Integer, String> indexWord = new HashMap<>();
    private Map<Integer, Integer> indexDocs = new HashMap<>();
    private Integer documentCount;



    /**
     * Create a Keras Tokenizer instance with full set of properties.
     *
     * @param numWords             The maximum vocabulary size, can be null
     * @param filters              Characters to filter
     * @param lower                whether to lowercase input or not
     * @param split                by which string to split words (usually single space)
     * @param charLevel            whether to operate on character- or word-level
     * @param outOfVocabularyToken replace items outside the vocabulary by this token
     */
    public KerasTokenizer(Integer numWords, String filters, boolean lower, String split, boolean charLevel,
                     String outOfVocabularyToken) {

        this.numWords = numWords;
        this.filters = filters;
        this.lower = lower;
        this.split = split;
        this.charLevel = charLevel;
        this.outOfVocabularyToken = outOfVocabularyToken;
    }


    /**
     * Tokenizer constructor with only numWords specified
     *
     * @param numWords             The maximum vocabulary size, can be null
     */
    public KerasTokenizer(Integer numWords) {
        this(numWords, DEFAULT_FILTER, true, DEFAULT_SPLIT, false, null);
    }

    /**
     * Default Keras tokenizer constructor
     */
    public KerasTokenizer() {
        this(null, DEFAULT_FILTER, true, DEFAULT_SPLIT, false, null);
    }


    /**
     * Import Keras Tokenizer from JSON file created with `tokenizer.to_json()` in Python.
     *
     * @param jsonFileName Full path of the JSON file to load
     * @return Keras Tokenizer instance loaded from JSON
     * @throws IOException I/O exception
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    public static KerasTokenizer fromJson(String jsonFileName) throws IOException, InvalidKerasConfigurationException {
        String json = new String(Files.readAllBytes(Paths.get(jsonFileName)));
        Map<String, Object> tokenizerBaseConfig = parseJsonString(json);
        Map<String, Object> tokenizerConfig;
        if (tokenizerBaseConfig.containsKey("config"))
            tokenizerConfig = (Map<String, Object>) tokenizerBaseConfig.get("config");
        else
            throw new InvalidKerasConfigurationException("No configuration found for Keras tokenizer");


        int numWords = (int) tokenizerConfig.get("num_words");
        String filters = (String) tokenizerConfig.get("filters");
        Boolean lower = (Boolean) tokenizerConfig.get("lower");
        String split = (String) tokenizerConfig.get("split");
        Boolean charLevel = (Boolean) tokenizerConfig.get("char_level");
        String oovToken = (String) tokenizerConfig.get("oov_token");
        int documentCount = (int) tokenizerConfig.get("document_count");

        @SuppressWarnings("unchecked")
        Map<String, Integer> wordCounts = (Map) parseJsonString((String) tokenizerConfig.get("word_counts"));
        @SuppressWarnings("unchecked")
        Map<String, Integer> wordDocs = (Map) parseJsonString((String) tokenizerConfig.get("word_docs"));
        @SuppressWarnings("unchecked")
        Map<String, Integer> wordIndex = (Map) parseJsonString((String) tokenizerConfig.get("word_index"));
        @SuppressWarnings("unchecked")
        Map<Integer, String> indexWord = (Map) parseJsonString((String) tokenizerConfig.get("index_word"));
        @SuppressWarnings("unchecked")
        Map<Integer, Integer> indexDocs = (Map) parseJsonString((String) tokenizerConfig.get("index_docs"));

        KerasTokenizer tokenizer = new KerasTokenizer(numWords, filters, lower, split, charLevel, oovToken);
        tokenizer.setDocumentCount(documentCount);
        tokenizer.setWordCounts(wordCounts);
        tokenizer.setWordDocs(new HashMap<>(wordDocs));
        tokenizer.setWordIndex(wordIndex);
        tokenizer.setIndexWord(indexWord);
        tokenizer.setIndexDocs(indexDocs);

        return tokenizer;
    }

    /**
     * Turns a String text into a sequence of tokens.
     *
     * @param text                 input text
     * @param filters              characters to filter
     * @param lower                whether to lowercase input or not
     * @param split                by which string to split words (usually single space)
     * @return Sequence of tokens as String array
     */
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

    /**
     * Fit this tokenizer on a corpus of texts.
     *
     * @param texts array of strings to fit tokenizer on.
     */
    public void fitOnTexts(String[] texts) {
        String[] sequence;
        for (String text : texts) {
            if (documentCount == null)
                documentCount = 1;
            else
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
                    wordCounts.put(word, wordCounts.get(word) + 1);
                else
                    wordCounts.put(word, 1);
            }
            Set<String> sequenceSet = new HashSet<>(Arrays.asList(sequence));
            for (String word: sequenceSet) {
                if (wordDocs.containsKey(word))
                    wordDocs.put(word, wordDocs.get(word) + 1);
                else
                    wordDocs.put(word, 1);
            }
        }
        Map<String, Integer> sortedWordCounts = reverseSortByValues(wordDocs);

        ArrayList<String> sortedVocabulary = new ArrayList<>();
        if (outOfVocabularyToken != null)
            sortedVocabulary.add(outOfVocabularyToken);
        for (String word: sortedWordCounts.keySet()) {
            sortedVocabulary.add(word);
        }

        for (int i = 0; i < sortedVocabulary.size(); i++)
            wordIndex.put(sortedVocabulary.get(i), i+1);

        for(String key : wordIndex.keySet()){
            indexWord.put(wordIndex.get(key), key);
        }

        for (String key: wordDocs.keySet())
            indexDocs.put(wordIndex.get(key), wordDocs.get(key));
    }

    /**
     * Sort HashMap by values in reverse order
     *
     * @param map input HashMap
     * @return sorted HashMap
     */
    private static HashMap reverseSortByValues(HashMap map) {
        List list = new LinkedList(map.entrySet());
        Collections.sort(list, new Comparator() {
            public int compare(Object o1, Object o2) {
                return ((Comparable) ((Map.Entry) (o2)).getValue())
                        .compareTo(((Map.Entry) (o1)).getValue());
            }
        });
        HashMap sortedHashMap = new LinkedHashMap();
        for (Iterator it = list.iterator(); it.hasNext();) {
            Map.Entry entry = (Map.Entry) it.next();
            sortedHashMap.put(entry.getKey(), entry.getValue());
        }
        return sortedHashMap;
    }

    /**
     * Fit this tokenizer on a corpus of word indices
     *
     * @param sequences array of indices derived from a text.
     */
    public void fitOnSequences(Integer[][] sequences) {
        documentCount += 1;
        for (Integer[] sequence: sequences) {
            Set<Integer> sequenceSet = new HashSet<>(Arrays.asList(sequence));
            for (Integer index: sequenceSet)
                indexDocs.put(index, indexDocs.get(index) + 1);
        }
    }

    /**
     * Transforms a bunch of texts into their index representations.
     *
     * @param texts input texts
     * @return array of indices of the texts
     */
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


    /**
     * Turns index sequences back into texts
     *
     * @param sequences index sequences
     * @return text reconstructed from sequences
     */
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


    /**
     * Turns an array of texts into an ND4J matrix of shape
     * (number of texts, number of words in vocabulary)
     *
     * @param texts input texts
     * @param mode TokenizerMode that controls how to vectorize data
     * @return resulting matrix representation
     */
    public INDArray textsToMatrix(String[] texts, TokenizerMode mode) {
        Integer[][] sequences = textsToSequences(texts);
        return sequencesToMatrix(sequences, mode);
    }

    /**
     * Turns an array of index sequences into an ND4J matrix of shape
     * (number of texts, number of words in vocabulary)
     *
     * @param sequences input sequences
     * @param mode TokenizerMode that controls how to vectorize data
     * @return resulting matrix representatio
     */
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
            HashMap<Integer, Integer> counts = new HashMap<>();
            for (int j: sequence) {
                if (j >= numWords)
                    continue;
                if (counts.containsKey(j))
                    counts.put(j, counts.get(j) + 1);
                else
                    counts.put(j, 1);
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
