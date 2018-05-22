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

package org.deeplearning4j.bagofwords.vectorizer;

import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.List;

@Deprecated
public abstract class Builder {
    protected VocabCache cache = new InMemoryLookupCache();
    protected TokenizerFactory tokenizerFactory;
    protected List<String> stopWords = StopWords.getStopWords();
    protected int minWordFrequency = 5;
    protected DocumentIterator docIter;
    protected SentenceIterator sentenceIterator;
    protected List<String> labels;
    protected InvertedIndex index;
    protected int batchSize = 1000;
    protected double sample = 0.0;
    protected boolean stem = false;
    protected boolean cleanup = false;



    public Builder cleanup(boolean cleanup) {
        this.cleanup = cleanup;
        return this;
    }

    public Builder stem(boolean stem) {
        this.stem = stem;
        return this;
    }

    public Builder sample(double sample) {
        this.sample = sample;
        return this;
    }


    public Builder batchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }


    public Builder index(InvertedIndex index) {
        this.index = index;
        return this;
    }

    public Builder labels(List<String> labels) {
        this.labels = labels;
        return this;
    }


    public Builder cache(VocabCache cache) {
        this.cache = cache;
        return this;
    }

    public Builder tokenize(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
        return this;
    }

    public Builder stopWords(List<String> stopWords) {
        this.stopWords = stopWords;
        return this;
    }



    public Builder minWords(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
        return this;
    }

    public Builder iterate(DocumentIterator docIter) {
        this.docIter = docIter;
        return this;
    }

    public Builder iterate(SentenceIterator sentenceIterator) {
        this.sentenceIterator = sentenceIterator;
        return this;
    }



    public abstract TextVectorizer build();
}
