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

package org.deeplearning4j.models.embeddings.wordvectors;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Word vectors. Handles operations based on the lookup table
 * and vocab.
 *
 * @author Adam Gibson
 */
public interface WordVectors extends Serializable {

    String getUNK();

    void setUNK(String newUNK);

    /**
     * Returns true if the model has this word in the vocab
     * @param word the word to test for
     * @return true if the model has the word in the vocab
     */
    boolean hasWord(String word);

    Collection<String> wordsNearest(INDArray words, int top);

    Collection<String> wordsNearestSum(INDArray words, int top);

    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    Collection<String> wordsNearestSum(String word, int n);


    /**
     * Words nearest based on positive and negative words
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top);

    /**
     * Accuracy based on questions which are a space separated list of strings
     * where the first word is the query word, the next 2 words are negative,
     * and the last word is the predicted word to be nearest
     * @param questions the questions to ask
     * @return the accuracy based on these questions
     */
    Map<String, Double> accuracy(List<String> questions);

    int indexOf(String word);

    /**
     * Find all words with a similar characters
     * in the vocab
     * @param word the word to compare
     * @param accuracy the accuracy: 0 to 1
     * @return the list of words that are similar in the vocab
     */
    List<String> similarWordsInVocabTo(String word, double accuracy);

    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    double[] getWordVector(String word);

    /**
     * Returns the word vector divided by the norm2 of the array
     * @param word the word to get the matrix for
     * @return the looked up matrix
     */
    INDArray getWordVectorMatrixNormalized(String word);

    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    INDArray getWordVectorMatrix(String word);


    /**
     * This method returns 2D array, where each row represents corresponding word/label
     *
     * @param labels
     * @return
     */
    INDArray getWordVectors(Collection<String> labels);

    /**
     * This method returns mean vector, built from words/labels passed in
     *
     * @param labels
     * @return
     */
    INDArray getWordVectorsMean(Collection<String> labels);

    /**
     * Words nearest based on positive and negative words
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top);


    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    Collection<String> wordsNearest(String word, int n);



    /**
     * Returns the similarity of 2 words
     * @param word the first word
     * @param word2 the second word
     * @return a normalized similarity (cosine similarity)
     */
    double similarity(String word, String word2);

    /**
     * Vocab for the vectors
     * @return
     */
    VocabCache vocab();

    /**
     * Lookup table for the vectors
     * @return
     */
    WeightLookupTable lookupTable();

    /**
     * Specifies ModelUtils to be used to access model
     * @param utils
     */
    void setModelUtils(ModelUtils utils);

}
