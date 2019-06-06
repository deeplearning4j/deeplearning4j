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

/**
 * Simple synchronized wrapper for SentenceIterator interface implementations
 *
 * @author raver119@gmail.com
 */
public class SynchronizedSentenceIterator implements SentenceIterator {
    private SentenceIterator underlyingIterator;

    public SynchronizedSentenceIterator(@NonNull SentenceIterator iterator) {
        this.underlyingIterator = iterator;
    }

    @Override
    public synchronized String nextSentence() {
        return this.underlyingIterator.nextSentence();
    }

    @Override
    public synchronized boolean hasNext() {
        return underlyingIterator.hasNext();
    }

    @Override
    public synchronized void reset() {
        this.underlyingIterator.reset();
    }

    @Override
    public synchronized void finish() {
        this.underlyingIterator.finish();
    }

    @Override
    public synchronized SentencePreProcessor getPreProcessor() {
        return this.underlyingIterator.getPreProcessor();
    }

    @Override
    public synchronized void setPreProcessor(SentencePreProcessor preProcessor) {
        this.underlyingIterator.setPreProcessor(preProcessor);
    }
}
