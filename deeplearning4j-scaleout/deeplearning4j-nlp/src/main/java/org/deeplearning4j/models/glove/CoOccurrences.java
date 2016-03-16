/*
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

package org.deeplearning4j.models.glove;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.routing.RoundRobinPool;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.glove.actor.CoOccurrenceActor;
import org.deeplearning4j.models.glove.actor.SentenceWork;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * Co occurrence counts
 *
 * @author Adam Gibson
 */
@Deprecated
public class CoOccurrences implements Serializable {
    private transient  TokenizerFactory tokenizerFactory;
    private transient SentenceIterator sentenceIterator;
    private int windowSize = 15;
    protected transient VocabCache cache;
    protected InvertedIndex index;
    protected transient ActorSystem trainingSystem;
    protected boolean symmetric = true;
    private Counter<Integer> sentenceOccurrences = Util.parallelCounter();
    private CounterMap<String,String> coOCurreneCounts = Util.parallelCounterMap();
    private static final Logger log = LoggerFactory.getLogger(CoOccurrences.class);
    private List<Pair<String,String>> coOccurrences;


    private CoOccurrences() {}

    public CoOccurrences(TokenizerFactory tokenizerFactory, SentenceIterator sentenceIterator, int windowSize, VocabCache cache, CounterMap<String, String> coOCurreneCounts,boolean symmetric) {
        this.tokenizerFactory = tokenizerFactory;
        this.sentenceIterator = sentenceIterator;
        this.windowSize = windowSize;
        this.cache = cache;
        this.coOCurreneCounts = coOCurreneCounts;
        this.symmetric = symmetric;
    }

    /**
     *
     */
    public void fit() {
        if(trainingSystem == null)
            trainingSystem = ActorSystem.create();

        final AtomicInteger processed = new AtomicInteger(0);

        final ActorRef actor = trainingSystem.actorOf(
                new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(
                        Props.create(
                                CoOccurrenceActor.class,
                                processed,
                                tokenizerFactory,
                                windowSize,
                                cache,
                                coOCurreneCounts,symmetric,sentenceOccurrences)));


        sentenceIterator.reset();

        final AtomicInteger queued = new AtomicInteger(0);
        int id = 0;
        while(sentenceIterator.hasNext()) {
            actor.tell(new SentenceWork(id,sentenceIterator.nextSentence()),actor);
            id++;
            queued.incrementAndGet();
        }


        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        while(processed.get() < queued.get()) {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        trainingSystem.shutdown();
        trainingSystem = null;


        log.info("Done processing co occurrences: ended with " + numCoOccurrences());


    }



    public class CoOccurrenceBatchIterator implements Iterator<List<Pair<VocabWord,VocabWord>>> {
        private Iterator<Pair<VocabWord,VocabWord>> iter = coOccurrenceIteratorVocab();
        private int batchSize = 100;

        public CoOccurrenceBatchIterator(int batchSize) {
            this.batchSize = batchSize;
        }

        public CoOccurrenceBatchIterator() {
            this(100);
        }


        @Override
        public boolean hasNext() {
            return iter.hasNext();
        }

        @Override
        public List<Pair<VocabWord, VocabWord>> next() {
            List<Pair<VocabWord,VocabWord>> list = new ArrayList<>(batchSize);
            for(int i = 0; i < batchSize; i++) {
                if(!iter.hasNext())
                    break;
                Pair<VocabWord,VocabWord> next = iter.next();
                list.add(next);
            }

            return list;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }


    public class CoOccurrenceIterator implements Iterator<Pair<VocabWord,VocabWord>> {
        private Iterator<Pair<String,String>> iter = coOccurrenceIterator();

        @Override
        public boolean hasNext() {
            return iter.hasNext();
        }

        @Override
        public Pair<VocabWord, VocabWord> next() {
            Pair<String,String> next = iter.next();
            // TODO: fix this
            Pair<VocabWord,VocabWord> ret = new Pair<>((VocabWord)cache.wordFor(next.getFirst()),(VocabWord) cache.wordFor(next.getSecond()));
            return ret;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    public Iterator<List<Pair<VocabWord,VocabWord>>> coOccurrenceIteratorVocabBatch(int batchSize) {
        return new CoOccurrenceBatchIterator(batchSize);
    }
    public Iterator<Pair<VocabWord,VocabWord>> coOccurrenceIteratorVocab() {
        return new CoOccurrenceIterator();
    }

    /**
     * Load from an input stream with the following format:
     * w1 w2 score
     * @param from the input stream to read from
     * @return the co occurrences based on the input stream
     */
    public static CoOccurrences load(InputStream from) {
        CoOccurrences ret = new CoOccurrences();
        ret.coOccurrences = new ArrayList<>();
        CounterMap<String,String> counter = new CounterMap<>();
        Reader inputStream = new InputStreamReader(from);
        LineIterator iter = IOUtils.lineIterator(inputStream);
        String line;
        while((iter.hasNext())) {
            line = iter.nextLine();
            String[] split = line.split(" ");
            if(split.length < 3)
                continue;
            //no empty keys
            if(split[0].isEmpty() || split[1].isEmpty())
                continue;

            ret.coOccurrences.add(new Pair<>(split[0],split[1]));

            counter.incrementCount(split[0], split[1], Double.parseDouble(split[2]));

        }

        ret.coOCurreneCounts = counter;
        return ret;

    }

    public Counter<Integer> getSentenceOccurrences() {
        return sentenceOccurrences;
    }

    public void setSentenceOccurrences(Counter<Integer> sentenceOccurrences) {
        this.sentenceOccurrences = sentenceOccurrences;
    }

    /**
     * Return a list of all of the co occurrences
     * @return a list of all of the co occurrences
     */
    public List<Pair<String,String>> coOccurrenceList() {
        if(coOccurrences != null)
            return coOccurrences;
        Iterator<Pair<String,String>> pairIter = coOccurrenceIterator();
        final List<Pair<String,String>> pairList = new ArrayList<>();

        while(pairIter.hasNext())
            pairList.add(pairIter.next());
        return pairList;

    }

    /**
     * Return a randomized list of the co occurrences
     * @return
     */
    public List<Pair<String,String>> randomizedList() {
        List<Pair<String,String>> coOccurrences = coOccurrenceList();
        Collections.shuffle(coOccurrences);
        return coOccurrences;
    }

    /**
     * The number of co occurrences
     * @return
     */
    public int numCoOccurrences() {
        return coOCurreneCounts.totalSize();
    }


    public double count(String w1,String w2) {
        return coOCurreneCounts.getCount(w1, w2);
    }


    /**
     * Get an iterator over all possible non zero
     * co occurrences
     * @return the iterator
     */
    public Iterator<Pair<String,String>> coOccurrenceIterator() {
        return coOCurreneCounts.getPairIterator();
    }

    public CounterMap<String, String> getCoOCurreneCounts() {
        return coOCurreneCounts;
    }

    public void setCoOCurreneCounts(CounterMap<String, String> coOCurreneCounts) {
        this.coOCurreneCounts = coOCurreneCounts;
    }

    public static class Builder {
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private SentenceIterator sentenceIterator;
        private int windowSize = 15;
        private VocabCache cache;
        private CounterMap<String, String> coOCurreneCounts = Util.parallelCounterMap();
        private boolean symmetric = true;


        public Builder symmetric(boolean symmetric) {
            this.symmetric = symmetric;
            return this;
        }

        public Builder tokenizer(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder iterate(SentenceIterator sentenceIterator) {
            this.sentenceIterator = sentenceIterator;
            return this;
        }

        public Builder windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder cache(VocabCache cache) {
            this.cache = cache;
            return this;
        }

        public Builder coOCurreneCounts(CounterMap<String, String> coOCurreneCounts) {
            this.coOCurreneCounts = coOCurreneCounts;
            return this;
        }

        public CoOccurrences build() {
            if(cache == null)
                throw new IllegalArgumentException("Vocab cache must not be null!");

            if(sentenceIterator == null)
                throw new IllegalArgumentException("Sentence iterator must not be null");

            return new CoOccurrences(tokenizerFactory, sentenceIterator, windowSize, cache, coOCurreneCounts,symmetric);
        }
    }




}
