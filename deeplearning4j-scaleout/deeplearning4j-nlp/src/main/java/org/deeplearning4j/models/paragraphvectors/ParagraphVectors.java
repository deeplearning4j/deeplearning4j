package org.deeplearning4j.models.paragraphvectors;

import com.google.common.base.Function;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.eclipse.jetty.util.ConcurrentHashSet;

import javax.annotation.Nullable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Paragraph Vectors:
 * [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
 .. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
 .. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
 In Proceedings of NIPS, 2013.

 @author Adam Gibson

 */
public class ParagraphVectors extends Word2Vec {
    //labels are also vocab words
    protected Queue<LinkedList<Pair<List<VocabWord>, Collection<VocabWord>>>> jobQueue = new LinkedBlockingDeque<>(10000);

    /**
     * Train the model
     */
    @Override
    public void fit() throws IOException {
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            vocab().saveVocab();
        if (stopWords == null)
            readStopWords();


        log.info("Training word2vec multithreaded");

        if (sentenceIter != null)
            sentenceIter.reset();
        if (docIter != null)
            docIter.reset();


        final int[] docs = vectorizer.index().allDocs();

        totalWords = vectorizer.numWordsEncountered();
        totalWords *= numIterations;



        log.info("Processing sentences...");


        List<Thread> work = new ArrayList<>();
        final AtomicInteger processed = new AtomicInteger(0);
        final int allDocs = docs.length * numIterations;
        final AtomicLong numWordsSoFar = new AtomicLong(0);
        final AtomicLong lastReport = new AtomicLong(0);
        for(int i = 0; i < workers; i++) {
            final Set<List<VocabWord>> set = new ConcurrentHashSet<>();

            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    final AtomicLong nextRandom = new AtomicLong(5);
                    long checked = 0;
                    while(true) {
                        if(checked > 0 && checked % 1000 == 0 && processed.get() >= allDocs)
                            return;
                        checked++;
                        List<Pair<List<VocabWord>, Collection<VocabWord>>> job = jobQueue.poll();
                        if(job == null || job.isEmpty() || set.contains(job))
                            continue;

                        log.info("Job of " + job.size());
                        double alpha = Math.max(minLearningRate, ParagraphVectors.this.alpha.get() * (1 - (1.0 * (double) numWordsSoFar.get() / (double) totalWords)));
                        long diff = Math.abs(lastReport.get() - numWordsSoFar.get());
                        if(numWordsSoFar.get() > 0 && diff >=  10000) {
                            log.info("Words so far " + numWordsSoFar.get() + " with alpha at " + alpha);
                            lastReport.set(numWordsSoFar.get());
                        }
                        long increment = 0;
                        double diff2 = 0.0;
                        for(Pair<List<VocabWord>, Collection<VocabWord>> sentenceWithLabel : job) {
                            trainSentence(sentenceWithLabel, nextRandom, alpha);
                            increment += sentenceWithLabel.getFirst().size();
                        }

                        log.info("Train sentence avg took " + diff2 / (double) job.size());
                        numWordsSoFar.set(numWordsSoFar.get() + increment);
                        processed.set(processed.get() + job.size());



                    }
                }
            });

            t.setName("worker" + i);
            t.start();
            work.add(t);
        }


        final AtomicLong nextRandom = new AtomicLong(5);
        final AtomicInteger doc = new AtomicInteger(0);
        final int numDocs = vectorizer.index().numDocuments() * numIterations;
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });


        final Queue<Pair<List<VocabWord>,Collection<VocabWord>>> batch2 = new ConcurrentLinkedDeque<>();

        vectorizer.index().eachDocWithLabels(new Function<Pair<List<VocabWord>, Collection<String>>, Void>() {
            @Override
            public Void apply(@Nullable Pair<List<VocabWord>, Collection<String>> input) {
                List<VocabWord> batch = new ArrayList<>();
                addWords(input.getFirst(), nextRandom, batch);

                if (batch.isEmpty())
                    return null;

                Collection<VocabWord> docLabels = new ArrayList<>();
                for(String s : input.getSecond())
                    docLabels.add(vocab().wordFor(s));

                for (int i = 0; i < numIterations; i++) {
                    batch2.add(new Pair<>(batch, docLabels));
                }

                if (batch2.size() >= 100 || batch2.size() >= numDocs) {
                    boolean added = false;
                    while (!added) {
                        try {
                            jobQueue.add(new LinkedList<>(batch2));
                            batch2.clear();
                            added = true;
                        } catch (Exception e) {
                            continue;
                        }
                    }

                }


                doc.incrementAndGet();
                if (doc.get() > 0 && doc.get() % 10000 == 0)
                    log.info("Doc " + doc.get() + " done so far");

                return null;
            }



        }, exec);

        if(!batch2.isEmpty()) {
            jobQueue.add(new LinkedList<>(batch2));

        }


        exec.shutdown();
        try {
            exec.awaitTermination(1,TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        for(Thread t : work)
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }


    }




    /**
     * Train on a list of vocab words
     * @param sentenceWithLabel the list of vocab words to train on
     */
    public void trainSentence(final Pair<List<VocabWord>, Collection<VocabWord>> sentenceWithLabel,AtomicLong nextRandom,double alpha) {
        if(sentenceWithLabel == null || sentenceWithLabel.getFirst().isEmpty())
            return;
        for(int i = 0; i < sentenceWithLabel.getFirst().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            dbow(i, sentenceWithLabel, (int) nextRandom.get() % window, nextRandom, alpha);
        }





    }

    /**
     * Train the distributed bag of words
     * model
     * @param i the word to train
     * @param sentenceWithLabel the sentence with labels to train
     * @param b
     * @param nextRandom
     * @param alpha
     */
    public void dbow(int i, Pair<List<VocabWord>, Collection<VocabWord>> sentenceWithLabel, int b, AtomicLong nextRandom, double alpha) {

        final VocabWord word = sentenceWithLabel.getFirst().get(i);
        List<VocabWord> sentence = sentenceWithLabel.getFirst();
        List<VocabWord> labels = (List<VocabWord>) sentenceWithLabel.getSecond();

        if(word == null || sentence.isEmpty())
            return;


        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < labels.size()) {
                    VocabWord lastWord = labels.get(c);
                    iterate(word,lastWord,nextRandom,alpha);
                }
            }
        }




    }


    protected void addWords(List<VocabWord> sentence,AtomicLong nextRandom,List<VocabWord> currMiniBatch) {
        for (VocabWord word : sentence) {
            if(word == null)
                continue;
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0) {
                double numDocs =  vectorizer.index().numDocuments();
                double ran = (Math.sqrt(word.getWordFrequency() / (sample * numDocs)) + 1)
                        * (sample * numDocs) / word.getWordFrequency();

                if (ran < (nextRandom.get() & 0xFFFF) / (double) 65536) {
                    continue;
                }

                currMiniBatch.add(word);
            }
            else
                currMiniBatch.add(word);



        }
    }


    public static class Builder extends Word2Vec.Builder {

        @Override
        public Builder index(InvertedIndex index) {
             super.index(index);
            return this;
        }

        @Override
        public Builder workers(int workers) {
             super.workers(workers);
            return this;
        }

        @Override
        public Builder sampling(double sample) {
             super.sampling(sample);
            return this;
        }

        @Override
        public Builder negativeSample(double negative) {
             super.negativeSample(negative);
            return this;
        }

        @Override
        public Builder minLearningRate(double minLearningRate) {
             super.minLearningRate(minLearningRate);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean useAdaGrad) {
             super.useAdaGrad(useAdaGrad);
            return this;
        }

        @Override
        public Builder vectorizer(TextVectorizer textVectorizer) {
            super.vectorizer(textVectorizer);
            return this;
        }

        @Override
        public Builder learningRateDecayWords(int learningRateDecayWords) {
            super.learningRateDecayWords(learningRateDecayWords);
            return this;
        }

        @Override
        public Builder batchSize(int batchSize) {
             super.batchSize(batchSize);
            return this;
        }

        @Override
        public Builder saveVocab(boolean saveVocab) {
             super.saveVocab(saveVocab);
            return this;
        }

        @Override
        public Builder seed(long seed) {
             super.seed(seed);
            return this;
        }

        @Override
        public Builder iterations(int iterations) {
             super.iterations(iterations);
            return this;
        }

        @Override
        public Builder learningRate(double lr) {
             super.learningRate(lr);
            return this;
        }

        @Override
        public Builder iterate(DocumentIterator iter) {
             super.iterate(iter);
            return this;
        }

        @Override
        public  Builder vocabCache(VocabCache cache) {
             super.vocabCache(cache);
            return this;
        }

        @Override
        public Builder minWordFrequency(int minWordFrequency) {
             super.minWordFrequency(minWordFrequency);
            return this;
        }

        @Override
        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
             super.tokenizerFactory(tokenizerFactory);
            return this;
        }

        @Override
        public Builder layerSize(int layerSize) {
             super.layerSize(layerSize);
            return this;
        }

        @Override
        public Builder stopWords(List<String> stopWords) {
             super.stopWords(stopWords);
            return this;
        }

        @Override
        public Builder windowSize(int window) {
             super.windowSize(window);
            return this;
        }

        @Override
        public Builder iterate(SentenceIterator iter) {
             super.iterate(iter);
            return this;
        }

        @Override
        public Builder lookupTable(WeightLookupTable lookupTable) {
             super.lookupTable(lookupTable);
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
                ret.setVocab(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;
                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }


                ret.docIter = docIter;
                ret.lookupTable = lookupTable;
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
                ret.setVocab(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }
                ret.lookupTable = lookupTable;
                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }
        }
    }


}
