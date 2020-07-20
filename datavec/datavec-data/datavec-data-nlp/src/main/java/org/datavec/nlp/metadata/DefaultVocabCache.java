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

package org.datavec.nlp.metadata;

import org.nd4j.common.primitives.Counter;
import org.datavec.api.conf.Configuration;
import org.datavec.nlp.vectorizer.TextVectorizer;
import org.nd4j.common.util.MathUtils;
import org.nd4j.common.util.Index;

/**
 * Vocab cache used for storing information
 * about vocab
 *
 * @author Adam Gibson
 */
public class DefaultVocabCache implements VocabCache {

    private Counter<String> wordFrequencies = new Counter<>();
    private Counter<String> docFrequencies = new Counter<>();
    private int minWordFrequency;
    private Index vocabWords = new Index();
    private double numDocs = 0;

    /**
     * Instantiate with a given min word frequency
     * @param minWordFrequency
     */
    public DefaultVocabCache(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
    }

    /*
     * Constructor for use with initialize()
     */
    public DefaultVocabCache() {
    }

    @Override
    public void incrementNumDocs(double by) {
        numDocs += by;
    }

    @Override
    public double numDocs() {
        return numDocs;
    }

    @Override
    public String wordAt(int i) {
        return vocabWords.get(i).toString();
    }

    @Override
    public int wordIndex(String word) {
        return vocabWords.indexOf(word);
    }

    @Override
    public void initialize(Configuration conf) {
        minWordFrequency = conf.getInt(TextVectorizer.MIN_WORD_FREQUENCY, 5);
    }

    @Override
    public double wordFrequency(String word) {
        return wordFrequencies.getCount(word);
    }

    @Override
    public int minWordFrequency() {
        return minWordFrequency;
    }

    @Override
    public Index vocabWords() {
        return vocabWords;
    }

    @Override
    public void incrementDocCount(String word) {
        incrementDocCount(word, 1.0);
    }

    @Override
    public void incrementDocCount(String word, double by) {
        docFrequencies.incrementCount(word, by);

    }

    @Override
    public void incrementCount(String word) {
        incrementCount(word, 1.0);
    }

    @Override
    public void incrementCount(String word, double by) {
        wordFrequencies.incrementCount(word, by);
        if (wordFrequencies.getCount(word) >= minWordFrequency && vocabWords.indexOf(word) < 0)
            vocabWords.add(word);
    }

    @Override
    public double idf(String word) {
        return docFrequencies.getCount(word);
    }

    @Override
    public double tfidf(String word, double frequency, boolean smoothIdf) {
        double tf = tf((int) frequency);
        double docFreq = docFrequencies.getCount(word);

        double idf = idf(numDocs, docFreq, smoothIdf);
        double tfidf = MathUtils.tfidf(tf, idf);
        return tfidf;
    }

    public double idf(double totalDocs, double numTimesWordAppearedInADocument, boolean smooth) {
        if(smooth){
            return Math.log((1 + totalDocs) / (1 + numTimesWordAppearedInADocument)) + 1.0;
        } else {
            return Math.log(totalDocs / numTimesWordAppearedInADocument) + 1.0;
        }
    }

    public static double tf(int count) {
        return count;
    }

    public int getMinWordFrequency() {
        return minWordFrequency;
    }

    public void setMinWordFrequency(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
    }
}
