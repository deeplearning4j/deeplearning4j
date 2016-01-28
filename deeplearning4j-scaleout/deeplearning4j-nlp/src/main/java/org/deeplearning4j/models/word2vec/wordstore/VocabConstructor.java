package org.deeplearning4j.models.word2vec.wordstore;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
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
    private List<String> stopWords;
    private boolean useAdaGrad = false;
    private boolean fetchLabels = false;
    private int limit;

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
     * This method transfers existing vocabulary into current one
     *
     * Please note: this method expects source vocabulary has Huffman tree indexes applied
     *
     * @param vocabCache
     * @return
     */
    public VocabCache<T> buildMergedVocabulary(@NonNull VocabCache<T> vocabCache, boolean fetchLabels) {
        if (cache == null) cache = new AbstractCache.Builder<T>().build();
        for (int t = 0; t < vocabCache.numWords(); t++) {
            String label = vocabCache.wordAtIndex(t);
            if (label == null) continue;
            T element = vocabCache.wordFor(label);

            // skip this element if it's a label, and user don't want labels to be merged
            if (!fetchLabels && element.isLabel()) continue;

            cache.addToken(element);
            cache.addWordToIndex(cache.numWords() - 1, element.getLabel());

            // backward compatibility code
            cache.putVocabWord(element.getLabel());
        }

        if (cache.numWords() == 0) throw new IllegalStateException("Source VocabCache has no indexes available, transfer is impossible");

        /*
            Now, when we have transferred vocab, we should roll over iterator, and  gather labels, if any
         */
        if (fetchLabels) {
            for(VocabSource<T> source: sources) {
                SequenceIterator<T> iterator = source.getIterator();
                iterator.reset();

                while (iterator.hasMoreSequences()) {
                    Sequence<T> sequence = iterator.nextSequence();

                    for (T label: sequence.getSequenceLabels()) {
                        if (!cache.containsWord(label.getLabel())) {
                            cache.addToken(label);
                            cache.addWordToIndex(cache.numWords() - 1, label.getLabel());

                            // backward compatibility code
                            cache.putVocabWord(label.getLabel());
                        }
                    }
                }
            }
        }

        return cache;
    }


    /**
     * This method scans all sources passed through builder, and returns all words as vocab.
     * If TargetVocabCache was set during instance creation, it'll be filled too.
     *
     *
     * @return
     */
    public VocabCache<T> buildJointVocabulary(boolean resetCounters, boolean buildHuffmanTree) {
        if (resetCounters && buildHuffmanTree) throw new IllegalStateException("You can't reset counters and build Huffman tree at the same time!");

        if (cache == null) cache = new AbstractCache.Builder<T>().build();
        log.debug("Target vocab size before building: [" + cache.numWords() + "]");
        final AtomicLong sequenceCounter = new AtomicLong(0);
        final AtomicLong elementsCounter = new AtomicLong(0);

        AbstractCache<T> topHolder = new AbstractCache.Builder<T>()
                .minElementFrequency(0)
                .build();

        int cnt = 0;
        for(VocabSource<T> source: sources) {
            SequenceIterator<T> iterator = source.getIterator();
            iterator.reset();

            log.debug("Trying source iterator: ["+ cnt+"]");
            log.debug("Target vocab size before building: [" + cache.numWords() + "]");
            cnt++;

            AbstractCache<T> tempHolder = new AbstractCache.Builder<T>().build();

            int sequences = 0;
            long counter = 0;
            while (iterator.hasMoreSequences()) {
                Sequence<T> document = iterator.nextSequence();
                sequenceCounter.incrementAndGet();
              //  log.info("Sequence length: ["+ document.getElements().size()+"]");
             //   Tokenizer tokenizer = tokenizerFactory.create(document.getContent());




                if (fetchLabels) {
                    T labelWord = document.getSequenceLabel();
                    labelWord.setSpecial(true);
                    labelWord.markAsLabel(true);
                    labelWord.setElementFrequency(1);

                    tempHolder.addToken(labelWord);
                }

                List<String> tokens = document.asLabels();
                for (String token: tokens) {
                    if (stopWords !=null && stopWords.contains(token)) continue;
                    if (token == null || token.isEmpty()) continue;

                    if (!tempHolder.containsWord(token)) {
                        T element = document.getElementByLabel(token);
                        element.setElementFrequency(1);
                        tempHolder.addToken(element);
                        elementsCounter.incrementAndGet();
                        counter++;
                        // TODO: this line should be uncommented only after AdaGrad is fixed, so the size of AdaGrad array is known
                        /*
                        if (useAdaGrad) {
                            VocabularyWord word = tempHolder.getVocabularyWordByString(token);

                            word.setHistoricalGradient(new double[layerSize]);
                        }
                        */
                    } else {
                        counter++;
                        tempHolder.incrementWordCount(token);
                    }
                }

                sequences++;
                if (sequenceCounter.get() % 100000 == 0) log.info("Sequences checked: [" + sequenceCounter.get() +"], Current vocabulary size: [" + elementsCounter.get() +"]");
            }
            // apply minWordFrequency set for this source
            log.debug("Vocab size before truncation: [" + tempHolder.numWords() + "],  NumWords: [" + tempHolder.totalWordOccurrences()+ "], sequences parsed: [" + sequences+ "], counter: ["+counter+"]");
            if (source.getMinWordFrequency() > 0) {
                LinkedBlockingQueue<String> labelsToRemove = new LinkedBlockingQueue<>();
                for (T element : tempHolder.vocabWords()) {
                    if (element.getElementFrequency() < source.getMinWordFrequency() && !element.isSpecial() && !element.isLabel())
                        labelsToRemove.add(element.getLabel());
                }

                for (String label: labelsToRemove) {
                    tempHolder.removeElement(label);
                }
            }

            log.debug("Vocab size after truncation: [" + tempHolder.numWords() + "],  NumWords: [" + tempHolder.totalWordOccurrences()+ "], sequences parsed: [" + sequences+ "], counter: ["+counter+"]");

            // at this moment we're ready to transfer
            topHolder.importVocabulary(tempHolder);
        }

        // at this moment, we have vocabulary full of words, and we have to reset counters before transfer everything back to VocabCache

            //topHolder.resetWordCounters();



        cache.importVocabulary(topHolder);

        if (resetCounters) {
            for (T element: cache.vocabWords()) {
                element.setElementFrequency(0);
            }
            cache.updateWordsOccurencies();
        }

        if (buildHuffmanTree) {
            Huffman huffman = new Huffman(cache.vocabWords());
            huffman.build();
            huffman.applyIndexes(cache);
            //topHolder.updateHuffmanCodes();

            if (limit > 0) {
                LinkedBlockingQueue<String> labelsToRemove = new LinkedBlockingQueue<>();
                for (T element : cache.vocabWords()) {
                    if (element.getIndex() > limit && !element.isSpecial() && !element.isLabel())
                        labelsToRemove.add(element.getLabel());
                }

                for (String label: labelsToRemove) {
                    cache.removeElement(label);
                }
            }
        }

        log.info("Sequences checked: [" + sequenceCounter.get() +"], Current vocabulary size: [" + cache.numWords() +"]");
        return cache;
    }

    public static class Builder<T extends SequenceElement> {
        private List<VocabSource<T>> sources = new ArrayList<>();
        private VocabCache<T> cache;
        private List<String> stopWords = new ArrayList<>();
        private boolean useAdaGrad = false;
        private boolean fetchLabels = false;
        private int limit;

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
        public Builder<T> setStopWords(@NonNull List<String> stopWords) {
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

        public VocabConstructor<T> build() {
            VocabConstructor<T> constructor = new VocabConstructor<T>();
            constructor.sources = this.sources;
            constructor.cache = this.cache;
            constructor.stopWords = this.stopWords;
            constructor.useAdaGrad = this.useAdaGrad;
            constructor.fetchLabels = this.fetchLabels;
            constructor.limit = this.limit;

            return constructor;
        }
    }

    @Data
    private static class VocabSource<T extends SequenceElement> {
        @NonNull private SequenceIterator<T> iterator;
        @NonNull private int minWordFrequency;
    }
}
