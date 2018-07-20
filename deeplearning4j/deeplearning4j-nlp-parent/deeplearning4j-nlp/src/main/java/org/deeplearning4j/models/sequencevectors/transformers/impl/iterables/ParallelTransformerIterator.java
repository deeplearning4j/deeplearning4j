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

package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.AsyncLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * TransformerIterator implementation that's does transformation/tokenization/normalization/whatever in parallel threads.
 * Suitable for cases when tokenization takes too much time for single thread.
 *
 * TL/DR: we read data from sentence iterator, and apply tokenization in parallel threads.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelTransformerIterator extends BasicTransformerIterator {

    protected BlockingQueue<Sequence<VocabWord>> buffer = new LinkedBlockingQueue<>(1024);
    protected BlockingQueue<LabelledDocument> stringBuffer;
    protected TokenizerThread[] threads;
    protected boolean underlyingHas = true;
    protected AtomicInteger processing = new AtomicInteger(0);

    protected static final AtomicInteger count = new AtomicInteger(0);

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer) {
        this(iterator, transformer, true);
    }

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer,
                    boolean allowMultithreading) {
        super(new AsyncLabelAwareIterator(iterator, 512), transformer);
        this.allowMultithreading = allowMultithreading;
        this.stringBuffer = new LinkedBlockingQueue<>(512);

        threads = new TokenizerThread[allowMultithreading ? Math.max(Runtime.getRuntime().availableProcessors(), 2) : 1];

        try {
            int cnt = 0;
            while (cnt < 256) {
                boolean before = underlyingHas;

                if (before)
                    underlyingHas = this.iterator.hasNextDocument();

                if (underlyingHas)
                    stringBuffer.put(this.iterator.nextDocument());
                else
                    cnt += 257;

                cnt++;
            }
        } catch (Exception e) {
            //
        }

        for (int x = 0; x < threads.length; x++) {
            threads[x] = new TokenizerThread(x, transformer, stringBuffer, buffer, processing);
            threads[x].setDaemon(true);
            threads[x].setName("ParallelTransformer thread " + x);
            threads[x].start();
        }
    }

    @Override
    public void reset() {
        this.iterator.shutdown();

        for (int x = 0; x < threads.length; x++) {
            if (threads[x] != null) {
                threads[x].shutdown();
                try {
                    threads[x].interrupt();
                } catch (Exception e) {
                    //
                }
            }
        }
    }

    @Override
    public boolean hasNext() {
        boolean before = underlyingHas;

        if (before)
            underlyingHas = iterator.hasNextDocument();
        else
            underlyingHas = false;

        return (underlyingHas || !buffer.isEmpty() || !stringBuffer.isEmpty() || processing.get() > 0);
    }

    @Override
    public Sequence<VocabWord> next() {
        try {
            if (underlyingHas)
                stringBuffer.put(iterator.nextDocument());

            return buffer.take();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    private static class TokenizerThread extends Thread implements Runnable {
        protected BlockingQueue<Sequence<VocabWord>> sequencesBuffer;
        protected BlockingQueue<LabelledDocument> stringsBuffer;
        protected SentenceTransformer sentenceTransformer;
        protected AtomicBoolean shouldWork = new AtomicBoolean(true);
        protected AtomicInteger processing;

        public TokenizerThread(int threadIdx, SentenceTransformer transformer,
                        BlockingQueue<LabelledDocument> stringsBuffer,
                        BlockingQueue<Sequence<VocabWord>> sequencesBuffer, AtomicInteger processing) {
            this.stringsBuffer = stringsBuffer;
            this.sequencesBuffer = sequencesBuffer;
            this.sentenceTransformer = transformer;
            this.processing = processing;

            this.setDaemon(true);
            this.setName("Tokenization thread " + threadIdx);
        }

        @Override
        public void run() {
            try {
                while (shouldWork.get()) {
                    LabelledDocument document = stringsBuffer.take();

                    if (document == null || document.getContent() == null)
                        continue;

                    processing.incrementAndGet();

                    Sequence<VocabWord> sequence = sentenceTransformer.transformToSequence(document.getContent());

                    if (document.getLabels() != null)
                        for (String label : document.getLabels()) {
                            if (label != null && !label.isEmpty())
                                sequence.addSequenceLabel(new VocabWord(1.0, label));
                        }

                    if (sequence != null)
                        sequencesBuffer.put(sequence);

                    processing.decrementAndGet();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                // do nothing
                shutdown();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public void shutdown() {
            shouldWork.set(false);
        }
    }
}
