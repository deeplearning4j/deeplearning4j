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

package org.deeplearning4j.models.word2vec;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.AbstractStorage;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This is special limited Word2Vec implementation, suited for serving as lookup table in concurrent multi-gpu environment
 * This implementation DOES NOT load all vectors onto any of gpus, instead of that it holds vectors in, optionally, compressed state in host memory.
 * This implementation DOES NOT provide some of original Word2Vec methods, such as wordsNearest or wordsNearestSum.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class StaticWord2Vec implements WordVectors {
    private List<Map<Integer, INDArray>> cacheWrtDevice = new ArrayList<>();
    private AbstractStorage<Integer> storage;
    private long cachePerDevice = 0L;
    private VocabCache<VocabWord> vocabCache;
    private String unk = null;

    private StaticWord2Vec() {

    }

    @Override
    public String getUNK() {
        return unk;
    }

    @Override
    public void setUNK(String newUNK) {
        this.unk = newUNK;
    }

    /**
     * Init method validates configuration defined using
     */
    protected void init() {
        if (storage.size() != vocabCache.numWords())
            throw new RuntimeException("Number of words in Vocab isn't matching number of stored Vectors. vocab: ["
                            + vocabCache.numWords() + "]; storage: [" + storage.size() + "]");

        // initializing device cache
        for (int i = 0; i < Nd4j.getAffinityManager().getNumberOfDevices(); i++) {
            cacheWrtDevice.add(new ConcurrentHashMap<Integer, INDArray>());
        }
    }

    /**
     * Returns true if the model has this word in the vocab
     *
     * @param word the word to test for
     * @return true if the model has the word in the vocab
     */
    @Override
    public boolean hasWord(String word) {
        return vocabCache.containsWord(word);
    }

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    @Override
    public Collection<String> wordsNearestSum(INDArray words, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Get the top n words most similar to the given word
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param word the word to compare
     * @param n    the n to get
     * @return the top n words
     */
    @Override
    public Collection<String> wordsNearestSum(String word, int n) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Words nearest based on positive and negative words
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param positive the positive words
     * @param negative the negative words
     * @param top      the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Accuracy based on questions which are a space separated list of strings
     * where the first word is the query word, the next 2 words are negative,
     * and the last word is the predicted word to be nearest
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param questions the questions to ask
     * @return the accuracy based on these questions
     */
    @Override
    public Map<String, Double> accuracy(List<String> questions) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    @Override
    public int indexOf(String word) {
        return vocabCache.indexOf(word);
    }

    /**
     * Find all words with a similar characters
     * in the vocab
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param word     the word to compare
     * @param accuracy the accuracy: 0 to 1
     * @return the list of words that are similar in the vocab
     */
    @Override
    public List<String> similarWordsInVocabTo(String word, double accuracy) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Get the word vector for a given matrix
     *
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    @Override
    public double[] getWordVector(String word) {
        return getWordVectorMatrix(word).data().asDouble();
    }

    /**
     * Returns the word vector divided by the norm2 of the array
     *
     * @param word the word to get the matrix for
     * @return the looked up matrix
     */
    @Override
    public INDArray getWordVectorMatrixNormalized(String word) {
        return Transforms.unitVec(getWordVectorMatrix(word));
    }

    /**
     * Get the word vector for a given matrix
     *
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    @Override
    public INDArray getWordVectorMatrix(String word) {
        // TODO: add variable UNK here
        int idx = 0;
        if (hasWord(word))
            idx = vocabCache.indexOf(word);
        else if (getUNK() != null)
            idx = vocabCache.indexOf(getUNK());
        else
            return null;

        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        INDArray array = null;

        if (cachePerDevice > 0 && cacheWrtDevice.get(deviceId).containsKey(idx))
            return cacheWrtDevice.get(Nd4j.getAffinityManager().getDeviceForCurrentThread()).get(idx);

        array = storage.get(idx);

        if (cachePerDevice > 0) {
            // TODO: add cache here
            long arrayBytes = array.length() * array.data().getElementSize();
            if ((arrayBytes * cacheWrtDevice.get(deviceId).size()) + arrayBytes < cachePerDevice)
                cacheWrtDevice.get(deviceId).put(idx, array);
        }

        return array;
    }

    /**
     * This method returns 2D array, where each row represents corresponding word/label
     *
     * @param labels
     * @return
     */
    @Override
    public INDArray getWordVectors(Collection<String> labels) {
        List<INDArray> words = new ArrayList<>();
        for (String label : labels) {
            if (hasWord(label) || getUNK() != null)
                words.add(getWordVectorMatrix(label));
        }

        return Nd4j.vstack(words);
    }

    /**
     * This method returns mean vector, built from words/labels passed in
     *
     * @param labels
     * @return
     */
    @Override
    public INDArray getWordVectorsMean(Collection<String> labels) {
        INDArray matrix = getWordVectors(labels);

        // TODO: check this (1)
        return matrix.mean(1);
    }

    /**
     * Words nearest based on positive and negative words
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param positive the positive words
     * @param negative the negative words
     * @param top      the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Get the top n words most similar to the given word
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @param word the word to compare
     * @param n    the n to get
     * @return the top n words
     */
    @Override
    public Collection<String> wordsNearest(String word, int n) {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Returns the similarity of 2 words
     *
     * @param label1  the first word
     * @param label2 the second word
     * @return a normalized similarity (cosine similarity)
     */
    @Override
    public double similarity(String label1, String label2) {
        if (label1 == null || label2 == null) {
            log.debug("LABELS: " + label1 + ": " + (label1 == null ? "null" : "exists") + ";" + label2 + " vec2:"
                            + (label2 == null ? "null" : "exists"));
            return Double.NaN;
        }

        INDArray vec1 = getWordVectorMatrix(label1).dup();
        INDArray vec2 = getWordVectorMatrix(label2).dup();

        if (vec1 == null || vec2 == null) {
            log.debug(label1 + ": " + (vec1 == null ? "null" : "exists") + ";" + label2 + " vec2:"
                            + (vec2 == null ? "null" : "exists"));
            return Double.NaN;
        }

        if (label1.equals(label2))
            return 1.0;

        vec1 = Transforms.unitVec(vec1);
        vec2 = Transforms.unitVec(vec2);

        return Transforms.cosineSim(vec1, vec2);
    }

    /**
     * Vocab for the vectors
     *
     * @return
     */
    @Override
    public VocabCache vocab() {
        return vocabCache;
    }

    /**
     * Lookup table for the vectors
     * PLEASE NOTE: This method is not available in this implementation.
     *
     * @return
     */
    @Override
    public WeightLookupTable lookupTable() {
        throw new UnsupportedOperationException("Method isn't implemented. Please use usual Word2Vec implementation");
    }

    /**
     * Specifies ModelUtils to be used to access model
     * PLEASE NOTE: This method has no effect in this implementation.
     *
     * @param utils
     */
    @Override
    public void setModelUtils(ModelUtils utils) {
        // no-op
    }

    public static class Builder {

        private AbstractStorage<Integer> storage;
        private long cachePerDevice = 0L;
        private VocabCache<VocabWord> vocabCache;

        /**
         *
         * @param storage AbstractStorage implementation, key has to be Integer, index of vocabWords
         * @param vocabCache VocabCache implementation, which will be used to lookup word indexes
         */
        public Builder(AbstractStorage<Integer> storage, VocabCache<VocabWord> vocabCache) {
            this.storage = storage;
            this.vocabCache = vocabCache;
        }


        /**
         * This method lets you to define if decompressed values will be cached, to avoid excessive decompressions.
         * If bytes == 0 - no cache will be used.
         *
         * @param bytes
         * @return
         */
        public Builder setCachePerDevice(long bytes) {
            this.cachePerDevice = bytes;
            return this;
        }


        /**
         * This method returns Static Word2Vec implementation, which is suitable for tasks like neural nets feeding.
         *
         * @return
         */
        public StaticWord2Vec build() {
            StaticWord2Vec word2Vec = new StaticWord2Vec();
            word2Vec.cachePerDevice = this.cachePerDevice;
            word2Vec.storage = this.storage;
            word2Vec.vocabCache = this.vocabCache;

            word2Vec.init();

            return word2Vec;
        }
    }
}
