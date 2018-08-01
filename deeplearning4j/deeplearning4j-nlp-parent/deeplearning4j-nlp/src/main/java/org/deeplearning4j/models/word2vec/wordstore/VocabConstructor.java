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

package org.deeplearning4j.models.word2vec.wordstore;

import lombok.Data;
import lombok.NonNull;
import lombok.val;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.util.ThreadUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.threadly.concurrent.PriorityScheduler;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * This class can be used to build joint vocabulary from special sources, that should be treated separately.
 * I.e. words from one source should have minWordFrequency set to 1, while the rest of corpus should have minWordFrequency set to 5.
 * So, here's the way to deal with it.
 *
 * It also can be used to simply build vocabulary out of arbitrary number of Sequences derived from arbitrary number of SequenceIterators
 *
 * @author raver119@gmail.com
 */
public class VocabConstructor<T extends SequenceElement> {
    private List<VocabSource<T>> sources = new ArrayList<>();
    private VocabCache<T> cache;
    private Collection<String> stopWords;
    private boolean useAdaGrad = false;
    private boolean fetchLabels = false;
    private int limit;
    private AtomicLong seqCount = new AtomicLong(0);
    private InvertedIndex<T> index;
    private boolean enableScavenger = false;
    private T unk;
    private boolean allowParallelBuilder = true;

    protected static final Logger log = LoggerFactory.getLogger(VocabConstructor.class);

    private VocabConstructor() {

    }

    /**
     * Placeholder for future implementation
     * @return
     */
    protected WeightLookupTable<T> buildExtendedLookupTable() {
        return null;
    }

    /**
     * Placeholder for future implementation
     * @return
     */
    protected VocabCache<T> buildExtendedVocabulary() {
        return null;
    }

    /**
     * This method transfers existing WordVectors model into current one
     *
     * @param wordVectors
     * @return
     */
    @SuppressWarnings("unchecked") // method is safe, since all calls inside are using generic SequenceElement methods
    public VocabCache<T> buildMergedVocabulary(@NonNull WordVectors wordVectors, boolean fetchLabels) {
        return buildMergedVocabulary((VocabCache<T>) wordVectors.vocab(), fetchLabels);
    }


    /**
     * This method returns total number of sequences passed through VocabConstructor
     *
     * @return
     */
    public long getNumberOfSequences() {
        return seqCount.get();
    }

    /**
     * This method transfers existing vocabulary into current one
     *
     * Please note: this method expects source vocabulary has Huffman tree indexes applied
     *
     * @param vocabCache
     * @return
     */
    public VocabCache<T> buildMergedVocabulary(@NonNull VocabCache<T> vocabCache, boolean fetchLabels) {
        if (cache == null)
            cache = new AbstractCache.Builder<T>().build();
        for (int t = 0; t < vocabCache.numWords(); t++) {
            String label = vocabCache.wordAtIndex(t);
            if (label == null)
                continue;
            T element = vocabCache.wordFor(label);

            // skip this element if it's a label, and user don't want labels to be merged
            if (!fetchLabels && element.isLabel())
                continue;

            //element.setIndex(t);
            cache.addToken(element);
            cache.addWordToIndex(element.getIndex(), element.getLabel());

            // backward compatibility code
            cache.putVocabWord(element.getLabel());
        }

        if (cache.numWords() == 0)
            throw new IllegalStateException("Source VocabCache has no indexes available, transfer is impossible");

        /*
            Now, when we have transferred vocab, we should roll over iterator, and  gather labels, if any
         */
        log.info("Vocab size before labels: " + cache.numWords());

        if (fetchLabels) {
            for (VocabSource<T> source : sources) {
                SequenceIterator<T> iterator = source.getIterator();
                iterator.reset();

                while (iterator.hasMoreSequences()) {
                    Sequence<T> sequence = iterator.nextSequence();
                    seqCount.incrementAndGet();

                    if (sequence.getSequenceLabels() != null)
                        for (T label : sequence.getSequenceLabels()) {
                            if (!cache.containsWord(label.getLabel())) {
                                label.markAsLabel(true);
                                label.setSpecial(true);

                                label.setIndex(cache.numWords());

                                cache.addToken(label);
                                cache.addWordToIndex(label.getIndex(), label.getLabel());

                                // backward compatibility code
                                cache.putVocabWord(label.getLabel());

                                //  log.info("Adding label ["+label.getLabel()+"]: " + cache.wordFor(label.getLabel()));
                            } // else log.info("Label ["+label.getLabel()+"] already exists: " + cache.wordFor(label.getLabel()));
                        }
                }
            }
        }

        log.info("Vocab size after labels: " + cache.numWords());

        return cache;
    }

    public VocabCache<T> transferVocabulary(@NonNull VocabCache<T> vocabCache, boolean buildHuffman) {
        val result = cache != null ? cache : new AbstractCache.Builder<T>().build();

        for (val v: vocabCache.tokens()) {
            result.addToken(v);

            // optionally transferring indices
            if (v.getIndex() >= 0)
                result.addWordToIndex(v.getIndex(), v.getLabel());
            else
                result.addWordToIndex(result.numWords(), v.getLabel());
        }

        if (buildHuffman) {
            val huffman = new Huffman(result.vocabWords());
            huffman.build();
            huffman.applyIndexes(result);
        }

        return result;
    }

    /**
     * This method scans all sources passed through builder, and returns all words as vocab.
     * If TargetVocabCache was set during instance creation, it'll be filled too.
     *
     *
     * @return
     */
    public VocabCache<T> buildJointVocabulary(boolean resetCounters, boolean buildHuffmanTree) {
        long lastTime = System.currentTimeMillis();
        long lastSequences = 0;
        long lastElements = 0;
        long startTime = lastTime;
        long startWords = 0;
        AtomicLong parsedCount = new AtomicLong(0);
        if (resetCounters && buildHuffmanTree)
            throw new IllegalStateException("You can't reset counters and build Huffman tree at the same time!");

        if (cache == null)
            cache = new AbstractCache.Builder<T>().build();
        log.debug("Target vocab size before building: [" + cache.numWords() + "]");
        final AtomicLong loopCounter = new AtomicLong(0);

        AbstractCache<T> topHolder = new AbstractCache.Builder<T>().minElementFrequency(0).build();

        int cnt = 0;
        int numProc = Runtime.getRuntime().availableProcessors();
        int numThreads = Math.max(numProc / 2, 2);
        PriorityScheduler executorService = new PriorityScheduler(numThreads);
        final AtomicLong execCounter = new AtomicLong(0);
        final AtomicLong finCounter = new AtomicLong(0);

        for (VocabSource<T> source : sources) {
            SequenceIterator<T> iterator = source.getIterator();
            iterator.reset();

            log.debug("Trying source iterator: [" + cnt + "]");
            log.debug("Target vocab size before building: [" + cache.numWords() + "]");
            cnt++;

            AbstractCache<T> tempHolder = new AbstractCache.Builder<T>().build();


            List<Long> timesHasNext = new ArrayList<>();
            List<Long> timesNext = new ArrayList<>();
            int sequences = 0;
            long time3 = 0;
            while (iterator.hasMoreSequences()) {
                Sequence<T> document = iterator.nextSequence();

                seqCount.incrementAndGet();
                parsedCount.addAndGet(document.size());
                tempHolder.incrementTotalDocCount();
                execCounter.incrementAndGet();
                VocabRunnable runnable = new VocabRunnable(tempHolder, document, finCounter, loopCounter);

                executorService.execute(runnable);

                // if we're not in parallel mode - wait till this runnable finishes
                if (!allowParallelBuilder) {
                    try {
                        runnable.awaitDone();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException(e);
                    }
                }

                // as we see in profiler, this lock isn't really happen too often
                // we don't want too much left in tail

                while (execCounter.get() - finCounter.get() > numProc) {
                    ThreadUtils.uncheckedSleep(1);
                }



                sequences++;
                if (seqCount.get() % 100000 == 0) {
                    long currentTime = System.currentTimeMillis();
                    long currentSequences = seqCount.get();
                    long currentElements = parsedCount.get();

                    double seconds = (currentTime - lastTime) / (double) 1000;

                    //                    Collections.sort(timesHasNext);
                    //                    Collections.sort(timesNext);

                    double seqPerSec = (currentSequences - lastSequences) / seconds;
                    double elPerSec = (currentElements - lastElements) / seconds;
                    //                    log.info("Document time: {} us; hasNext time: {} us", timesNext.get(timesNext.size() / 2), timesHasNext.get(timesHasNext.size() / 2));
                    log.info("Sequences checked: [{}]; Current vocabulary size: [{}]; Sequences/sec: {}; Words/sec: {};",
                                    seqCount.get(), tempHolder.numWords(), String.format("%.2f", seqPerSec),
                                    String.format("%.2f", elPerSec));
                    lastTime = currentTime;
                    lastElements = currentElements;
                    lastSequences = currentSequences;

                    //                    timesHasNext.clear();
                    //                    timesNext.clear();
                }

                /**
                 * Firing scavenger loop
                 */
                if (enableScavenger && loopCounter.get() >= 2000000 && tempHolder.numWords() > 10000000) {
                    log.info("Starting scavenger...");
                    while (execCounter.get() != finCounter.get()) {
                        ThreadUtils.uncheckedSleep(1);
                    }

                    filterVocab(tempHolder, Math.max(1, source.getMinWordFrequency() / 2));
                    loopCounter.set(0);
                }

                //                timesNext.add((time2 - time1) / 1000L);
                //                timesHasNext.add((time1 - time3) / 1000L);

                //                time3 = System.nanoTime();
            }

            // block untill all threads are finished
            log.debug("Waiting till all processes stop...");
            while (execCounter.get() != finCounter.get()) {
                ThreadUtils.uncheckedSleep(1);
            }


            // apply minWordFrequency set for this source
            log.debug("Vocab size before truncation: [" + tempHolder.numWords() + "],  NumWords: ["
                            + tempHolder.totalWordOccurrences() + "], sequences parsed: [" + seqCount.get()
                            + "], counter: [" + parsedCount.get() + "]");
            if (source.getMinWordFrequency() > 0) {
                filterVocab(tempHolder, source.getMinWordFrequency());
            }

            log.debug("Vocab size after truncation: [" + tempHolder.numWords() + "],  NumWords: ["
                            + tempHolder.totalWordOccurrences() + "], sequences parsed: [" + seqCount.get()
                            + "], counter: [" + parsedCount.get() + "]");
            // at this moment we're ready to transfer
            topHolder.importVocabulary(tempHolder);
        }

        // at this moment, we have vocabulary full of words, and we have to reset counters before transfer everything back to VocabCache

        //topHolder.resetWordCounters();



        System.gc();

        cache.importVocabulary(topHolder);

        // adding UNK word
        if (unk != null) {
            log.info("Adding UNK element to vocab...");
            unk.setSpecial(true);
            cache.addToken(unk);
        }

        if (resetCounters) {
            for (T element : cache.vocabWords()) {
                element.setElementFrequency(0);
            }
            cache.updateWordsOccurrences();
        }

        if (buildHuffmanTree) {
            if (limit > 0) {
                // we want to sort labels before truncating them, so we'll keep most important words
                val words = new ArrayList<T>(cache.vocabWords());
                Collections.sort(words);

                // now rolling through them
                for (val element : words) {
                    if (element.getIndex() > limit && !element.isSpecial() && !element.isLabel())
                        cache.removeElement(element.getLabel());
                }
            }

            // and now we're building Huffman tree
            val huffman = new Huffman(cache.vocabWords());
            huffman.build();
            huffman.applyIndexes(cache);
        }

        executorService.shutdown();

        System.gc();

        long endSequences = seqCount.get();
        long endTime = System.currentTimeMillis();
        double seconds = (endTime - startTime) / (double) 1000;
        double seqPerSec = endSequences / seconds;
        log.info("Sequences checked: [{}], Current vocabulary size: [{}]; Sequences/sec: [{}];", seqCount.get(),
                        cache.numWords(), String.format("%.2f", seqPerSec));
        return cache;
    }

    protected void filterVocab(AbstractCache<T> cache, int minWordFrequency) {
        int numWords = cache.numWords();
        LinkedBlockingQueue<String> labelsToRemove = new LinkedBlockingQueue<>();
        for (T element : cache.vocabWords()) {
            if (element.getElementFrequency() < minWordFrequency && !element.isSpecial() && !element.isLabel())
                labelsToRemove.add(element.getLabel());
        }

        for (String label : labelsToRemove) {
            cache.removeElement(label);
        }

        log.debug("Scavenger: Words before: {}; Words after: {};", numWords, cache.numWords());
    }

    public static class Builder<T extends SequenceElement> {
        private List<VocabSource<T>> sources = new ArrayList<>();
        private VocabCache<T> cache;
        private Collection<String> stopWords = new ArrayList<>();
        private boolean useAdaGrad = false;
        private boolean fetchLabels = false;
        private InvertedIndex<T> index;
        private int limit;
        private boolean enableScavenger = false;
        private T unk;
        private boolean allowParallelBuilder = true;

        public Builder() {

        }

        /**
         * This method sets the limit to resulting vocabulary size.
         *
         * PLEASE NOTE:  This method is applicable only if huffman tree is built.
         *
         * @param limit
         * @return
         */
        public Builder<T> setEntriesLimit(int limit) {
            this.limit = limit;
            return this;
        }


        public Builder<T> allowParallelTokenization(boolean reallyAllow) {
            this.allowParallelBuilder = reallyAllow;
            return this;
        }

        /**
         * Defines, if adaptive gradients should be created during vocabulary mastering
         *
         * @param useAdaGrad
         * @return
         */
        protected Builder<T> useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        /**
         * After temporary internal vocabulary is built, it will be transferred to target VocabCache you pass here
         *
         * @param cache target VocabCache
         * @return
         */
        public Builder<T> setTargetVocabCache(@NonNull VocabCache<T> cache) {
            this.cache = cache;
            return this;
        }

        /**
         * Adds SequenceIterator for vocabulary construction.
         * Please note, you can add as many sources, as you wish.
         *
         * @param iterator SequenceIterator to build vocabulary from
         * @param minElementFrequency elements with frequency below this value will be removed from vocabulary
         * @return
         */
        public Builder<T> addSource(@NonNull SequenceIterator<T> iterator, int minElementFrequency) {
            sources.add(new VocabSource<T>(iterator, minElementFrequency));
            return this;
        }

        /*
        public Builder<T> addSource(LabelAwareIterator iterator, int minWordFrequency) {
            sources.add(new VocabSource(iterator, minWordFrequency));
            return this;
        }
        
        public Builder<T> addSource(SentenceIterator iterator, int minWordFrequency) {
            sources.add(new VocabSource(new SentenceIteratorConverter(iterator), minWordFrequency));
            return this;
        }
        */
        /*
        public Builder setTokenizerFactory(@NonNull TokenizerFactory factory) {
            this.tokenizerFactory = factory;
            return this;
        }
        */
        public Builder<T> setStopWords(@NonNull Collection<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        /**
         * Sets, if labels should be fetched, during vocab building
         *
         * @param reallyFetch
         * @return
         */
        public Builder<T> fetchLabels(boolean reallyFetch) {
            this.fetchLabels = reallyFetch;
            return this;
        }

        public Builder<T> setIndex(InvertedIndex<T> index) {
            this.index = index;
            return this;
        }

        public Builder<T> enableScavenger(boolean reallyEnable) {
            this.enableScavenger = reallyEnable;
            return this;
        }

        public Builder<T> setUnk(T unk) {
            this.unk = unk;
            return this;
        }

        public VocabConstructor<T> build() {
            VocabConstructor<T> constructor = new VocabConstructor<>();
            constructor.sources = this.sources;
            constructor.cache = this.cache;
            constructor.stopWords = this.stopWords;
            constructor.useAdaGrad = this.useAdaGrad;
            constructor.fetchLabels = this.fetchLabels;
            constructor.limit = this.limit;
            constructor.index = this.index;
            constructor.enableScavenger = this.enableScavenger;
            constructor.unk = this.unk;
            constructor.allowParallelBuilder = this.allowParallelBuilder;

            return constructor;
        }
    }

    @Data
    private static class VocabSource<T extends SequenceElement> {
        @NonNull
        private SequenceIterator<T> iterator;
        @NonNull
        private int minWordFrequency;
    }


    protected class VocabRunnable implements Runnable {
        private final AtomicLong finalCounter;
        private final Sequence<T> document;
        private final AbstractCache<T> targetVocab;
        private final AtomicLong loopCounter;
        private boolean done;

        public VocabRunnable(@NonNull AbstractCache<T> targetVocab, @NonNull Sequence<T> sequence,
                        @NonNull AtomicLong finalCounter, @NonNull AtomicLong loopCounter) {
            this.finalCounter = finalCounter;
            this.document = sequence;
            this.targetVocab = targetVocab;
            this.loopCounter = loopCounter;
        }

        public void awaitDone() throws InterruptedException {
            synchronized (this) {
                while (! done) {
                    this.wait();
                }
            }
        }

	@Override
        public void run() {
            try {
                Map<String, AtomicLong> seqMap = new HashMap<>();
                //  log.info("Sequence length: ["+ document.getElements().size()+"]");

                if (fetchLabels && document.getSequenceLabels() != null) {
                    for (T labelWord : document.getSequenceLabels()) {
                        if (!targetVocab.hasToken(labelWord.getLabel())) {
                            labelWord.setSpecial(true);
                            labelWord.markAsLabel(true);
                            labelWord.setElementFrequency(1);

                            targetVocab.addToken(labelWord);
                        }
                    }
                }

                List<String> tokens = document.asLabels();
                for (String token : tokens) {
                    if (stopWords != null && stopWords.contains(token))
                        continue;
                    if (token == null || token.isEmpty())
                        continue;

                    if (!targetVocab.containsWord(token)) {
                        T element = document.getElementByLabel(token);
                        element.setElementFrequency(1);
                        element.setSequencesCount(1);
                        targetVocab.addToken(element);
                        //                    elementsCounter.incrementAndGet();
                        loopCounter.incrementAndGet();

                        // if there's no such element in tempHolder, it's safe to set seqCount to 1
                        seqMap.put(token, new AtomicLong(0));
                    } else {
                        targetVocab.incrementWordCount(token);

                        // if element exists in tempHolder, we should update it seqCount, but only once per sequence
                        if (!seqMap.containsKey(token)) {
                            seqMap.put(token, new AtomicLong(1));
                            T element = targetVocab.wordFor(token);
                            element.incrementSequencesCount();
                        }

                        if (index != null) {
                            if (document.getSequenceLabel() != null) {
                                index.addWordsToDoc(index.numDocuments(), document.getElements(), document.getSequenceLabel());
                            } else {
                                index.addWordsToDoc(index.numDocuments(), document.getElements());
                            }
                        }
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                finalCounter.incrementAndGet();
                synchronized (this) {
                    this.notifyAll();
                }
            }
        }
    }
}
