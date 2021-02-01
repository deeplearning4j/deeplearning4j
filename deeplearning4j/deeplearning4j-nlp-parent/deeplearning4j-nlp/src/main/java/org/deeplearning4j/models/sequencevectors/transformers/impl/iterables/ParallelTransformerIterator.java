/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.AsyncLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
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

    protected static final int capacity = 1024;
    protected BlockingQueue<Future<Sequence<VocabWord>>> buffer = new LinkedBlockingQueue<>(capacity);
    //protected BlockingQueue<LabelledDocument> stringBuffer;
    //protected TokenizerThread[] threads;
    protected AtomicBoolean underlyingHas = new AtomicBoolean(true);
    protected AtomicInteger processing = new AtomicInteger(0);

    private ExecutorService executorService;

    protected static final AtomicInteger count = new AtomicInteger(0);

    private static final int PREFETCH_SIZE = 100;

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer) {
        this(iterator, transformer, true);
    }

    private void prefetchIterator() {
        /*for (int i = 0; i < PREFETCH_SIZE; ++i) {
            //boolean before = underlyingHas;

                if (underlyingHas.get())
                    underlyingHas.set(iterator.hasNextDocument());
                else
                    underlyingHas.set(false);

            if (underlyingHas.get()) {
                CallableTransformer callableTransformer = new CallableTransformer(iterator.nextDocument(), sentenceTransformer);
                Future<Sequence<VocabWord>> futureSequence = executorService.submit(callableTransformer);
                try {
                    buffer.put(futureSequence);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }*/
    }

    public ParallelTransformerIterator(@NonNull LabelAwareIterator iterator, @NonNull SentenceTransformer transformer,
                                       boolean allowMultithreading) {
        super(new AsyncLabelAwareIterator(iterator, 512), transformer);
        //super(iterator, transformer);
        this.allowMultithreading = allowMultithreading;
        //this.stringBuffer = new LinkedBlockingQueue<>(512);

        //threads = new TokenizerThread[1];
        //threads = new TokenizerThread[allowMultithreading ? Math.max(Runtime.getRuntime().availableProcessors(), 2) : 1];
        executorService = Executors.newFixedThreadPool(allowMultithreading ? Math.max(Runtime.getRuntime().availableProcessors(), 2) : 1);

        prefetchIterator();
    }

    @Override
    public void reset() {
        this.executorService.shutdownNow();
        this.iterator.reset();
        underlyingHas.set(true);
        prefetchIterator();
        this.buffer.clear();
        this.executorService = Executors.newFixedThreadPool(allowMultithreading ? Math.max(Runtime.getRuntime().availableProcessors(), 2) : 1);
    }

    public void shutdown() {
        executorService.shutdown();
    }

    private static class CallableTransformer implements Callable<Sequence<VocabWord>> {

        private LabelledDocument document;
        private SentenceTransformer transformer;

        public CallableTransformer(LabelledDocument document, SentenceTransformer transformer) {
            this.transformer = transformer;
            this.document = document;
        }

        @Override
        public Sequence<VocabWord> call() {
            Sequence<VocabWord> sequence = new Sequence<>();

            if (document != null && document.getContent() != null) {
                sequence = transformer.transformToSequence(document.getContent());
                if (document.getLabels() != null) {
                    for (String label : document.getLabels()) {
                        if (label != null && !label.isEmpty())
                            sequence.addSequenceLabel(new VocabWord(1.0, label));
                    }
                }
            }
            return sequence;
        }
    }

    @Override
    public boolean hasNext() {
        //boolean before = underlyingHas;

        //if (underlyingHas.get()) {
            if (buffer.size() < capacity && iterator.hasNextDocument()) {
                CallableTransformer transformer = new CallableTransformer(iterator.nextDocument(), sentenceTransformer);
                Future<Sequence<VocabWord>> futureSequence = executorService.submit(transformer);
                try {
                    buffer.put(futureSequence);
                } catch (InterruptedException e) {
                    log.error("",e);
                }
            }
          /*  else
                underlyingHas.set(false);

        }
        else {
           underlyingHas.set(false);
        }*/

        return (/*underlyingHas.get() ||*/ !buffer.isEmpty() || /*!stringBuffer.isEmpty() ||*/ processing.get() > 0);
    }

    @Override
    public Sequence<VocabWord> next() {
        try {
            /*if (underlyingHas)
                stringBuffer.put(iterator.nextDocument());*/
            processing.incrementAndGet();
            Future<Sequence<VocabWord>> future = buffer.take();
            Sequence<VocabWord>  sequence = future.get();
            processing.decrementAndGet();
            return sequence;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
