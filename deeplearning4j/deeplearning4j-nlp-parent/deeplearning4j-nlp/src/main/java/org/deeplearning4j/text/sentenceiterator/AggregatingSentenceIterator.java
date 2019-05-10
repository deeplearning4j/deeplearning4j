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

package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is simple wrapper suited for aggregation of few SentenceIterators into single flow.
 *
 * @author raver119@gmail.com
 */
public class AggregatingSentenceIterator implements SentenceIterator {
    private List<SentenceIterator> backendIterators;
    private SentencePreProcessor preProcessor;
    private AtomicInteger position = new AtomicInteger(0);

    private AggregatingSentenceIterator(@NonNull List<SentenceIterator> list) {
        this.backendIterators = list;
    }

    @Override
    public String nextSentence() {
        if (!backendIterators.get(position.get()).hasNext() && position.get() < backendIterators.size()) {
            position.incrementAndGet();
        }

        return (preProcessor == null) ? backendIterators.get(position.get()).nextSentence()
                        : preProcessor.preProcess(backendIterators.get(position.get()).nextSentence());
    }

    @Override
    public boolean hasNext() {
        for (SentenceIterator iterator : backendIterators) {
            if (iterator.hasNext()) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void reset() {
        for (SentenceIterator iterator : backendIterators) {
            iterator.reset();
        }
        this.position.set(0);
    }

    @Override
    public void finish() {
        for (SentenceIterator iterator : backendIterators) {
            iterator.finish();
        }
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    public static class Builder {
        private List<SentenceIterator> backendIterators = new ArrayList<>();
        private SentencePreProcessor preProcessor;

        public Builder() {

        }

        public Builder addSentenceIterator(@NonNull SentenceIterator iterator) {
            this.backendIterators.add(iterator);
            return this;
        }

        public Builder addSentenceIterators(@NonNull Collection<SentenceIterator> iterator) {
            this.backendIterators.addAll(iterator);
            return this;
        }

        public Builder addSentencePreProcessor(@NonNull SentencePreProcessor preProcessor) {
            this.preProcessor = preProcessor;
            return this;
        }

        public AggregatingSentenceIterator build() {
            AggregatingSentenceIterator sentenceIterator = new AggregatingSentenceIterator(this.backendIterators);
            if (this.preProcessor != null) {
                sentenceIterator.setPreProcessor(this.preProcessor);
            }
            return sentenceIterator;
        }
    }
}
