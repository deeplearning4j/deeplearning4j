package org.deeplearning4j.models.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Glove by socher et. al
 *
 * @author Adam Gibson
 */
public class Glove implements Serializable {

    private VocabCache vocabCache;
    private SentenceIterator sentenceIterator;
    private TextVectorizer textVectorizer;
    private TokenizerFactory tokenizerFactory;
    private GloveWeightLookupTable weightLookupTable;
    private int layerSize = 100;
    private double learningRate = 0.1;
    private double xMax = 0.75;
    private int windowSize = 15;
    private CoOccurrences coOccurrences;
    private List<String> stopWords = StopWords.getStopWords();
    private boolean stem = false;
    protected Queue<List<List<VocabWord>>> jobQueue = new LinkedBlockingDeque<>(10000);
    private int batchSize = 1000;
    private int minWordFrequency = 5;

    public Glove(VocabCache vocabCache, SentenceIterator sentenceIterator, TextVectorizer textVectorizer, TokenizerFactory tokenizerFactory, GloveWeightLookupTable weightLookupTable, int layerSize, double learningRate, double xMax, int windowSize, CoOccurrences coOccurrences, List<String> stopWords, boolean stem,int batchSize,int minWordFrequency) {
        this.vocabCache = vocabCache;
        this.sentenceIterator = sentenceIterator;
        this.textVectorizer = textVectorizer;
        this.tokenizerFactory = tokenizerFactory;
        this.weightLookupTable = weightLookupTable;
        this.layerSize = layerSize;
        this.learningRate = learningRate;
        this.xMax = xMax;
        this.windowSize = windowSize;
        this.coOccurrences = coOccurrences;
        this.stopWords = stopWords;
        this.stem = stem;
        this.batchSize = batchSize;
        this.minWordFrequency = minWordFrequency;
    }

    public void fit() {
        if(vocabCache == null)
            vocabCache = new InMemoryLookupCache();

        if(textVectorizer == null) {
            textVectorizer = new TfidfVectorizer.Builder().tokenize(tokenizerFactory)
                    .cache(vocabCache).iterate(sentenceIterator).minWords(minWordFrequency)
                    .stopWords(stopWords).stem(stem).build();

            textVectorizer.fit();
        }


        sentenceIterator.reset();

        if(coOccurrences == null) {
            coOccurrences = new CoOccurrences.Builder()
                    .cache(vocabCache).iterate(sentenceIterator)
                    .tokenizer(tokenizerFactory).windowSize(windowSize)
                    .build();

            coOccurrences.fit();

        }

        if(weightLookupTable == null)
            weightLookupTable = new GloveWeightLookupTable.Builder().xMax(xMax)
                    .cache(vocabCache).lr(learningRate).vectorLength(layerSize)
                    .build();


        if(weightLookupTable.getSyn0() == null)
            weightLookupTable.resetWeights();

        final List<List<VocabWord>> miniBatches = new CopyOnWriteArrayList<>();

        for(String s : coOccurrences.getCoOCurreneCounts().keySet()) {
            for(String s1 : coOccurrences.getCoOCurreneCounts().getCounter(s).keySet()) {
                VocabWord vocabWord = vocabCache.wordFor(s);
                VocabWord vocabWord1 = vocabCache.wordFor(s1);
                miniBatches.add(Arrays.asList(vocabWord, vocabWord1));
                if(miniBatches.size() >= batchSize) {
                    jobQueue.add(new ArrayList<>(miniBatches));
                    miniBatches.clear();
                }
            }
        }




    if(!miniBatches.isEmpty())
            jobQueue.add(miniBatches);


    final AtomicInteger processed = new AtomicInteger(coOccurrences.getCoOCurreneCounts().size());


    try {
        Thread.sleep(10000);
    } catch (InterruptedException e) {
        e.printStackTrace();
    }

    Parallelization.runInParallel(Runtime.getRuntime().availableProcessors(),new Runnable() {
        @Override
        public void run() {
            while(processed.get() > 0) {
                List<List<VocabWord>> batch = jobQueue.poll();
                if(batch == null)
                    continue;

                for(List<VocabWord> list : batch) {
                    VocabWord w1 = list.get(0);
                    VocabWord w2 = list.get(1);
                    double weight = coOccurrences.getCoOCurreneCounts().getCount(w1.getWord(),w2.getWord());
                    weightLookupTable.iterateSample(w1,w2,learningRate,weight);
                    processed.decrementAndGet();

                }

            }
        }
    });




}



public static class Builder {
    private VocabCache vocabCache;
    private SentenceIterator sentenceIterator;
    private TextVectorizer textVectorizer;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    private GloveWeightLookupTable weightLookupTable;
    private int layerSize = 300;
    private double learningRate = 0.5;
    private double xMax = 0.75;
    private int windowSize = 5;
    private CoOccurrences coOccurrences;
    private List<String> stopWords = StopWords.getStopWords();
    private boolean stem = false;
    private int batchSize = 100;
    private int minWordFrequency = 5;

    public Builder minWordFrequency(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
        return this;
    }

    public Builder cache(VocabCache vocabCache) {
        this.vocabCache = vocabCache;
        return this;
    }

    public Builder iterate(SentenceIterator sentenceIterator) {
        this.sentenceIterator = sentenceIterator;
        return this;
    }

    public Builder vectorizer(TextVectorizer textVectorizer) {
        this.textVectorizer = textVectorizer;
        return this;
    }

    public Builder tokenizer(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
        return this;
    }

    public Builder weights(GloveWeightLookupTable weightLookupTable) {
        this.weightLookupTable = weightLookupTable;
        return this;
    }

    public Builder layerSize(int layerSize) {
        this.layerSize = layerSize;
        return this;
    }

    public Builder learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public Builder xMax(double xMax) {
        this.xMax = xMax;
        return this;
    }

    public Builder windowSize(int windowSize) {
        this.windowSize = windowSize;
        return this;
    }

    public Builder coOccurrences(CoOccurrences coOccurrences) {
        this.coOccurrences = coOccurrences;
        return this;
    }

    public Builder stopWords(List<String> stopWords) {
        this.stopWords = stopWords;
        return this;
    }

    public Builder stem(boolean stem) {
        this.stem = stem;
        return this;
    }

    public Builder batchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public Glove build() {
        return new Glove(vocabCache, sentenceIterator, textVectorizer, tokenizerFactory, weightLookupTable, layerSize, learningRate, xMax, windowSize, coOccurrences, stopWords, stem, batchSize,minWordFrequency);
    }
}

}
