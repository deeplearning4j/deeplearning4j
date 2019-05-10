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

package org.deeplearning4j.bagofwords.vectorizer;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.ArrayList;
import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseTextVectorizer implements TextVectorizer {
    @Setter
    protected transient TokenizerFactory tokenizerFactory;
    protected transient LabelAwareIterator iterator;
    protected int minWordFrequency;
    @Getter
    protected VocabCache<VocabWord> vocabCache;
    protected LabelsSource labelsSource;
    protected Collection<String> stopWords = new ArrayList<>();
    @Getter
    protected transient InvertedIndex<VocabWord> index;
    protected boolean isParallel = true;

    protected LabelsSource getLabelsSource() {
        return labelsSource;
    }

    public void buildVocab() {
        if (vocabCache == null)
            vocabCache = new AbstractCache.Builder<VocabWord>().build();


        SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(this.iterator)
                        .tokenizerFactory(tokenizerFactory).build();

        AbstractSequenceIterator<VocabWord> iterator = new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(iterator, minWordFrequency).setTargetVocabCache(vocabCache).setStopWords(stopWords)
                        .allowParallelTokenization(isParallel).build();

        constructor.buildJointVocabulary(false, true);
    }

    @Override
    public void fit() {
        buildVocab();
    }

    /**
     * Returns the number of words encountered so far
     *
     * @return the number of words encountered so far
     */
    @Override
    public long numWordsEncountered() {
        return vocabCache.totalWordOccurrences();
    }
}
