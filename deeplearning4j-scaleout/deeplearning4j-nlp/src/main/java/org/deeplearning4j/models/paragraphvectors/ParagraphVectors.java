package org.deeplearning4j.models.paragraphvectors;

import com.google.common.base.Function;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.eclipse.jetty.util.ConcurrentHashSet;

import javax.annotation.Nullable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by agibsonccc on 11/30/14.
 */
public class ParagraphVectors extends Word2Vec {
    protected boolean trainLabels = true;
    protected boolean trainWords = true;
    protected List<VocabWord> labels = new ArrayList<>();
    //labels are also vocab words
    protected Queue<List<Pair<List<VocabWord>,List<VocabWord>>>> jobQueue = new LinkedBlockingDeque<>(10000);

    /**
     * Train the model
     */
    public void fit() throws IOException {
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            cache.saveVocab();
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
                        List<Pair<List<VocabWord>,List<VocabWord>>> job = jobQueue.poll();
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
                        for(Pair<List<VocabWord>,List<VocabWord>> sentenceWithLabel : job) {
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


        //TODO: need to handle binding labels to documents from the inverted index...
        final Queue<Pair<List<VocabWord>,String>> batch2 = new ConcurrentLinkedDeque<>();
        vectorizer.index().eachDocWithLabel(new Function<Pair<List<VocabWord>, String>, Void>() {
            @Override
            public Void apply(@Nullable Pair<List<VocabWord>, String> input) {
                List<VocabWord> batch = new ArrayList<>();
                addWords(input.getFirst(), nextRandom, batch);

                if (batch.isEmpty())
                    return null;

                for (int i = 0; i < numIterations; i++) {
                    batch2.add(new Pair<>(batch,input.getSecond()));
                }

                if (batch2.size() >= 100 || batch2.size() >= numDocs) {
                    boolean added = false;
                    while (!added) {
                        try {
                            //jobQueue.add(new LinkedList<>(batch2));
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
    public void trainSentence(final Pair<List<VocabWord>,List<VocabWord>> sentenceWithLabel,AtomicLong nextRandom,double alpha) {
        if(sentenceWithLabel == null || sentenceWithLabel.getFirst().isEmpty())
            return;
        for(int i = 0; i < sentenceWithLabel.getFirst().size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            dm(i, sentenceWithLabel, (int) nextRandom.get() % window, nextRandom, alpha);
        }





    }

    /**
     * Train the distributed memory model
     * @param i the word to train
     * @param sentenceWithLabel the sentence with labels to train
     * @param b
     * @param nextRandom
     * @param alpha
     */
    public void dm(int i, Pair<List<VocabWord>,List<VocabWord>> sentenceWithLabel, int b, AtomicLong nextRandom, double alpha) {

        final VocabWord word = sentenceWithLabel.getFirst().get(i);
        List<VocabWord> sentence = sentenceWithLabel.getFirst();
        List<VocabWord> labels = sentenceWithLabel.getSecond();

        if(word == null || sentence.isEmpty())
            return;

        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    VocabWord lastWord = sentence.get(c);
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

}
