package org.deeplearning4j.models.word2vec.wordstore;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * This class can be used to build joint vocabulary from special sources, that should be treated separately.
 * I.e. words from one source should have minWordFrequency set to 1, while the rest of corpus should have minWordFrequency set to 5.
 * So, here's the way to deal with it.
 *
 * @author raver119@gmail.com
 */
public class VocabConstructor {
    private List<VocabSource> sources = new ArrayList<>();
    private TokenizerFactory tokenizerFactory;
    private VocabCache cache;
    private List<String> stopWords;
    private boolean useAdaGrad = false;

    protected static final Logger log = LoggerFactory.getLogger(VocabConstructor.class);

    private VocabConstructor() {

    }

    /**
     * This method scans all sources passed through builder, and returns all words as vocab.
     * If TargetVocabCache was set during instance creation, it'll be filled too.
     *
     *
     * @return
     */
    public VocabCache buildJointVocabulary(boolean resetCounters, boolean buildHuffmanTree) {
        if (resetCounters && buildHuffmanTree) throw new IllegalStateException("You can't reset counters and build Huffman tree at the same time!");

        if (cache == null) cache = new InMemoryLookupCache(false);
        
        VocabularyHolder topHolder = new VocabularyHolder.Builder()
                .externalCache(cache)
                .minWordFrequency(0)
                .build();

        for(VocabSource source: sources) {
            SentenceIterator iterator = source.getIterator();
            iterator.reset();

            VocabularyHolder tempHolder = new VocabularyHolder.Builder()
                    .minWordFrequency(source.getMinWordFrequency())
                    .build();

            while (iterator.hasNext()) {
                String line = iterator.nextSentence();
                Tokenizer tokenizer = tokenizerFactory.create(line);

                List<String> tokens = tokenizer.getTokens();
                for (String token: tokens) {
                    if (stopWords !=null && stopWords.contains(token)) continue;
                    if (token == null || token.isEmpty()) continue;

                    if (!tempHolder.containsWord(token)) {
                        tempHolder.addWord(token);

                        // TODO: this line should be uncommented only after AdaGrad is fixed, so the size of AdaGrad array is known
                        /*
                        if (useAdaGrad) {
                            VocabularyWord word = tempHolder.getVocabularyWordByString(token);

                            word.setHistoricalGradient(new double[layerSize]);
                        }
                        */
                    } else {
                        tempHolder.incrementWordCounter(token);
                    }
                }
            }
            // apply minWordFrequency set for this source
            log.info("Vocab size before truncation: " + tempHolder.numWords());
            tempHolder.truncateVocabulary();

            log.info("Vocab size after truncation: " + tempHolder.numWords());

            // at this moment we're ready to transfer
            topHolder.consumeVocabulary(tempHolder);
        }

        // at this moment, we have vocabulary full of words, and we have to reset counters before transfer everything back to VocabCache
        if (resetCounters)
            topHolder.resetWordCounters();

        if (buildHuffmanTree)
            topHolder.updateHuffmanCodes();

        topHolder.transferBackToVocabCache(cache);
        return cache;
    }

    public static class Builder {
        private List<VocabSource> sources = new ArrayList<>();
        private TokenizerFactory tokenizerFactory;
        private VocabCache cache;
        private List<String> stopWords = new ArrayList<>();
        private boolean useAdaGrad = false;

        public Builder() {

        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder setTargetVocabCache(@NonNull VocabCache cache) {
            this.cache = cache;
            return this;
        }

        public Builder addSource(SentenceIterator iterator, int minWordFrequency) {
            sources.add(new VocabSource(iterator, minWordFrequency));
            return this;
        }

        public Builder setTokenizerFactory(@NonNull TokenizerFactory factory) {
            this.tokenizerFactory = factory;
            return this;
        }

        public Builder setStopWords(@NonNull List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public VocabConstructor build() {
            VocabConstructor constructor = new VocabConstructor();
            constructor.sources = this.sources;
            constructor.tokenizerFactory = this.tokenizerFactory;
            constructor.cache = this.cache;
            constructor.stopWords = this.stopWords;
            constructor.useAdaGrad = this.useAdaGrad;

            return constructor;
        }
    }

    @Data
    private static class VocabSource {
        @NonNull private SentenceIterator iterator;
        @NonNull private int minWordFrequency;
    }
}
