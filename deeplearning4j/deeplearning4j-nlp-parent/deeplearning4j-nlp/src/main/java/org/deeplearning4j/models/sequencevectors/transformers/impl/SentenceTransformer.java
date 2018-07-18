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

package org.deeplearning4j.models.sequencevectors.transformers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.sequencevectors.transformers.impl.iterables.BasicTransformerIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.iterables.ParallelTransformerIterator;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.BasicLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This simple class is responsible for conversion lines of text to Sequences of SequenceElements to fit them into SequenceVectors model
 *
 * @author raver119@gmail.com
 */
public class SentenceTransformer implements SequenceTransformer<VocabWord, String>, Iterable<Sequence<VocabWord>> {
    /*
            So, we must accept any SentenceIterator implementations, and build vocab out of it, and use it for further transforms between text and Sequences
     */
    protected TokenizerFactory tokenizerFactory;
    protected LabelAwareIterator iterator;
    protected boolean readOnly = false;
    protected AtomicInteger sentenceCounter = new AtomicInteger(0);
    protected boolean allowMultithreading = false;
    protected BasicTransformerIterator currentIterator;

    protected static final Logger log = LoggerFactory.getLogger(SentenceTransformer.class);

    private SentenceTransformer(@NonNull LabelAwareIterator iterator) {
        this.iterator = iterator;
    }

    @Override
    public Sequence<VocabWord> transformToSequence(String object) {
        Sequence<VocabWord> sequence = new Sequence<>();

        Tokenizer tokenizer = tokenizerFactory.create(object);
        List<String> list = tokenizer.getTokens();

        for (String token : list) {
            if (token == null || token.isEmpty() || token.trim().isEmpty())
                continue;

            VocabWord word = new VocabWord(1.0, token);
            sequence.addElement(word);
        }

        sequence.setSequenceId(sentenceCounter.getAndIncrement());
        return sequence;
    }

    @Override
    public Iterator<Sequence<VocabWord>> iterator() {
        if (currentIterator != null)
            reset();

        if (!allowMultithreading)
            currentIterator = new BasicTransformerIterator(iterator, this);
        else
            currentIterator = new ParallelTransformerIterator(iterator, this, true);

        return currentIterator;
    }

    @Override
    public void reset() {
        if (currentIterator != null)
            currentIterator.reset();
    }


    public static class Builder {
        protected TokenizerFactory tokenizerFactory;
        protected LabelAwareIterator iterator;
        protected VocabCache<VocabWord> vocabCache;
        protected boolean readOnly = false;
        protected boolean allowMultithreading = false;

        public Builder() {

        }

        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder iterator(@NonNull LabelAwareIterator iterator) {
            this.iterator = iterator;
            return this;
        }

        public Builder iterator(@NonNull SentenceIterator iterator) {
            this.iterator = new BasicLabelAwareIterator.Builder(iterator).build();
            return this;
        }

        public Builder iterator(@NonNull DocumentIterator iterator) {
            this.iterator = new BasicLabelAwareIterator.Builder(iterator).build();
            return this;
        }

        public Builder readOnly(boolean readOnly) {
            this.readOnly = true;
            return this;
        }

        /**
         * This method enables/disables parallel processing over sentences
         *
         * @param reallyAllow
         * @return
         */
        public Builder allowMultithreading(boolean reallyAllow) {
            this.allowMultithreading = reallyAllow;
            return this;
        }

        public SentenceTransformer build() {
            SentenceTransformer transformer = new SentenceTransformer(this.iterator);
            transformer.tokenizerFactory = this.tokenizerFactory;
            transformer.readOnly = this.readOnly;
            transformer.allowMultithreading = this.allowMultithreading;

            return transformer;
        }
    }
}
