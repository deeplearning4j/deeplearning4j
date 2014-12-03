package org.deeplearning4j.models.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
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

    public Glove(VocabCache vocabCache, SentenceIterator sentenceIterator, TextVectorizer textVectorizer, TokenizerFactory tokenizerFactory, GloveWeightLookupTable weightLookupTable, int layerSize, double learningRate, double xMax, int windowSize, CoOccurrences coOccurrences, List<String> stopWords, boolean stem,int batchSize) {
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
    }

    public void fit() {
        textVectorizer = new TfidfVectorizer.Builder()
                .cache(vocabCache).iterate(sentenceIterator)
                .stopWords(stopWords).stem(stem).build();

        textVectorizer.fit();


        coOccurrences = new CoOccurrences.Builder()
                .cache(vocabCache).iterate(sentenceIterator)
                .tokenizer(tokenizerFactory).windowSize(windowSize)
                .build();

        coOccurrences.fit();

        weightLookupTable = new GloveWeightLookupTable.Builder().xMax(xMax)
                .cache(vocabCache).lr(learningRate).vectorLength(layerSize)
                .build();


        final List<List<VocabWord>> miniBatches = new ArrayList<>();
        new Thread(new Runnable() {
            @Override
            public void run() {
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
            }
        }).start();


        if(!miniBatches.isEmpty())
            jobQueue.add(miniBatches);


        final AtomicInteger processed = new AtomicInteger(coOccurrences.getCoOCurreneCounts().size());
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



}
