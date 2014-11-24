package org.deeplearning4j.models.word2vec;

import com.google.common.base.Function;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;

import javax.annotation.Nullable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Known in some circles as doc2vec. Officially called paragraph vectors.
 * See:
 * http://arxiv.org/pdf/1405.4053v2.pdf
 *
 * @author Adam Gibson
 */
public class ParagraphVectors extends Word2Vec {


    private boolean trainLabels = true;
    private boolean trainWords = true;

    public void dm(List<VocabWord> sentence,boolean trainLabels,boolean trainWords) {

    }


    @Override
    public void fit() throws IOException {
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            cache.saveVocab();
        if (stopWords == null)
            readStopWords();


        log.info("Training paragraph vectors multithreaded");

        if (sentenceIter != null)
            sentenceIter.reset();
        if (docIter != null)
            docIter.reset();


        final int[] docs = vectorizer.index().allDocs();

        final AtomicInteger numSentencesProcessed = new AtomicInteger(0);
        totalWords = vectorizer.numWordsEncountered();
        totalWords *= numIterations;



        log.info("Processing sentences...");

        List<Thread> work = new ArrayList<>();
        final AtomicInteger processed = new AtomicInteger(0);
        for(int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {

            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    final AtomicLong nextRandom = new AtomicLong(5);
                    while(true) {
                        if(processed.get() >= docs.length)
                            break;
                        List<VocabWord> job = jobQueue.poll();
                        if(job == null)
                            continue;
                        trainSentence(job,numSentencesProcessed,nextRandom);
                        processed.incrementAndGet();


                    }
                }
            });

            t.setName("worker" + i);
            t.setDaemon(true);
            t.start();
            work.add(t);
        }


        final List<VocabWord> batch = new ArrayList<>(batchSize);
        final AtomicLong nextRandom = new AtomicLong(5);
        final AtomicInteger doc = new AtomicInteger(0);
        final int numDocs = vectorizer.index().numDocuments();
        ExecutorService exec = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for(int i = 0; i < numIterations; i++)
            vectorizer.index().eachDoc(new Function<List<VocabWord>, Void>() {
                @Nullable
                @Override
                public Void apply(@Nullable List<VocabWord> input) {
                    addWords(input, nextRandom, batch);
                    try {
                        while(!jobQueue.offer(batch,1, TimeUnit.MILLISECONDS)) {
                            Thread.sleep(1);
                        }


                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }

                    doc.incrementAndGet();
                    if(doc.get() > 0 && doc.get() % 10000 == 0)
                        log.info("Doc " + doc.get() + " done so far out of " + numDocs);
                    batch.clear();

                    return null;
                }
            },exec);







        if(!jobQueue.isEmpty()) {
            jobQueue.add(new ArrayList<>(batch));
            batch.clear();
        }


        for(int i = 0; i < work.size(); i++)
            jobQueue.add(new ArrayList<VocabWord>());

        for(Thread t : work)
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

        for (int i = 0; i < numIterations; i++) {
            log.info("Training on " + docs.length);



        }

    }

    public static class Builder extends Word2Vec.Builder {

       private boolean trainWords = true;
       private boolean trainLabels = true;


        public Builder trainWords(boolean trainWords) {
            this.trainWords = trainWords;
            return this;
        }

        public Builder trainLabels(boolean trainLabels) {
            this.trainLabels = trainLabels;
            return this;
        }

        @Override
        public ParagraphVectors build() {

            if(iter == null) {
                ParagraphVectors ret = new ParagraphVectors();
                ret.layerSize = layerSize;
                ret.window = window;
                ret.alpha.set(lr);
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.setCache(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.sample = sampling;
                ret.trainLabels = trainLabels;
                ret.trainWords = trainWords;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr)
                            .vectorLength(layerSize).build();

                    ret.cache = vocabCache;
                }
                ret.docIter = docIter;
                ret.tokenizerFactory = tokenizerFactory;

                return ret;
            }

            else {
                ParagraphVectors ret = new ParagraphVectors();
                ret.alpha.set(lr);
                ret.layerSize = layerSize;
                ret.sentenceIter = iter;
                ret.window = window;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setCache(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;
                ret.trainLabels = trainLabels;
                ret.trainWords = trainWords;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr)
                            .vectorLength(layerSize).build();
                    ret.cache = vocabCache;
                }
                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }



        }
    }

}
