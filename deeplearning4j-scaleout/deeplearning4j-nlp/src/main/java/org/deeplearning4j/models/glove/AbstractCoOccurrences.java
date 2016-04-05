package org.deeplearning4j.models.glove;

import lombok.NonNull;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.glove.count.*;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.FilteredSequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.SynchronizedSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.PrefetchingSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SynchronizedSentenceIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class implements building cooccurrence map for abstract training corpus.
 * However it's performance rather low, due to exsessive IO that happens in ShadowCopyThread
 *
 * PLEASE NOTE: Current implementation involves massive IO, and it should be rewritter as soon as ND4j gets sparse arrays support
 *
 * @author raver119@gmail.com
 */
public class AbstractCoOccurrences<T extends SequenceElement> implements Serializable {

    protected boolean symmetric;
    protected int windowSize;
    protected VocabCache<T> vocabCache;
    protected SequenceIterator<T> sequenceIterator;

    // please note, we need enough room for ShadowCopy thread, that's why -1 there
    protected int workers = Math.max(Runtime.getRuntime().availableProcessors() - 1, 1);

    // target file, where text with cooccurrencies should be saved
    protected File targetFile;

    protected ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    protected long memory_threshold = 0;

    private ShadowCopyThread shadowThread;

//    private Counter<Integer> sentenceOccurrences = Util.parallelCounter();
    //private CounterMap<T, T> coOccurrenceCounts = Util.parallelCounterMap();
    private volatile CountMap<T> coOccurrenceCounts = new CountMap<>();
    //private Counter<Integer> occurrenceAllocations = Util.parallelCounter();
    //private List<Pair<T, T>> coOccurrences;
    private AtomicLong processedSequences = new AtomicLong(0);


    protected static final Logger logger = LoggerFactory.getLogger(AbstractCoOccurrences.class);

    // this method should be private, to avoid non-configured instantiation
    private AbstractCoOccurrences() {
        ;
    }

    /**
     * This method returns cooccurrence distance weights for two SequenceElements
     *
     * @param element1
     * @param element2
     * @return distance weight
     */
    public double getCoOccurrenceCount(@NonNull T element1, @NonNull T element2) {
        return coOccurrenceCounts.getCount(element1, element2);
    }

    /**
     * This method returns estimated memory footrpint, based on current CountMap content
     * @return
     */
    protected long getMemoryFootprint() {
        // TODO: implement this method. It should return approx. memory used by appropriate CountMap
        try {
            lock.readLock().lock();
            return ((long) coOccurrenceCounts.size()) * 24L * 5L;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * This memory returns memory threshold, defined as 1/2 of memory allowed for allocation
     * @return
     */
    protected long getMemoryThreshold() {
        return memory_threshold / 2L;
    }

    public void fit() {
        shadowThread = new ShadowCopyThread();
        shadowThread.start();

        // we should reset iterator before counting cooccurrences
        sequenceIterator.reset();

        List<CoOccurrencesCalculatorThread> threads = new ArrayList<>();
        for (int x = 0; x < workers; x++) {
            threads.add(x, new CoOccurrencesCalculatorThread(x, new FilteredSequenceIterator<T>(new SynchronizedSequenceIterator<T>(sequenceIterator), vocabCache), processedSequences));
            threads.get(x).start();
        }

        for (int x = 0; x < workers; x++) {
            try {
                threads.get(x).join();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        shadowThread.finish();
        logger.info("CoOccurrences map was built.");
    }

    /**
     *
     *  This method returns iterator with elements pairs and their weights. Resulting iterator is safe to use in multi-threaded environment.
     *
     * Developer's note: thread safety on received iterator is delegated to PrefetchedSentenceIterator
     * @return
     */
    public Iterator<Pair<Pair<T, T>, Double>> iterator() {
        final SentenceIterator iterator;

        try {
            iterator = new SynchronizedSentenceIterator(new PrefetchingSentenceIterator.Builder(new BasicLineIterator(targetFile))
                    .setFetchSize(500000)
                    .build());

        } catch (Exception e) {
            logger.error("Target file was not found on last stage!");
            throw new RuntimeException(e);
        }
        return new Iterator<Pair<Pair<T, T>, Double>>() {
            /*
                    iterator should be built on top of current text file with all pairs
             */

            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public Pair<Pair<T, T>, Double> next() {
                String line = iterator.nextSentence();
                String[] strings = line.split(" ");

                T element1 = vocabCache.elementAtIndex(Integer.valueOf(strings[0]));
                T element2 = vocabCache.elementAtIndex(Integer.valueOf(strings[1]));
                Double weight = Double.valueOf(strings[2]);

                return new Pair<>(new Pair<T, T>(element1, element2), weight);
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException("remove() method can't be supported on read-only interface");
            }
        };
    }

    public static class Builder<T extends SequenceElement> {

        protected boolean symmetric;
        protected int windowSize = 5;
        protected VocabCache<T> vocabCache;
        protected SequenceIterator<T> sequenceIterator;
        protected int workers = Runtime.getRuntime().availableProcessors();
        protected File target;
        protected long maxmemory = Runtime.getRuntime().maxMemory();

        public Builder() {

        }

        public Builder<T> symmetric(boolean reallySymmetric) {
            this.symmetric = reallySymmetric;
            return this;
        }

        public Builder<T> windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder<T> vocabCache(@NonNull VocabCache<T> cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder<T> iterate(@NonNull SequenceIterator<T> iterator) {
            this.sequenceIterator = new SynchronizedSequenceIterator<T>(iterator);
            return this;
        }

        public Builder<T> workers(int numWorkers) {
            this.workers = numWorkers;
            return this;
        }

        /**
         * This method allows you to specify maximum memory available for CoOccurrence map builder.
         *
         * Please note: this option can be considered a debugging method. In most cases setting proper -Xmx argument set to JVM is enough to limit this algorithm.
         * Please note: this option won't override -Xmx JVM value.
         *
         * @param gbytes memory available, in GigaBytes
         * @return
         */
        public Builder<T> maxMemory(int gbytes) {
            if (gbytes > 0) this.maxmemory = Math.max(gbytes - 1, 1) * 1024 * 1024 * 1024L;

            return this;
        }

        /**
         * Path to save cooccurrence map after construction.
         * If targetFile is not specified, temporary file will be used.
         *
         * @param path
         * @return
         */
        public Builder<T> targetFile(@NonNull String path) {
            this.targetFile(new File(path));
            return this;
        }

        /**
         * Path to save cooccurrence map after construction.
         * If targetFile is not specified, temporary file will be used.
         *
         * @param file
         * @return
         */
        public Builder<T> targetFile(@NonNull File file) {
            this.target = file;
            return this;
        }

        public AbstractCoOccurrences<T> build() {
            AbstractCoOccurrences<T> ret = new AbstractCoOccurrences<>();
            ret.sequenceIterator = this.sequenceIterator;
            ret.windowSize = this.windowSize;
            ret.vocabCache = this.vocabCache;
            ret.symmetric = this.symmetric;
            ret.workers = this.workers;

            if (this.maxmemory < 1) this.maxmemory = Runtime.getRuntime().maxMemory();
            ret.memory_threshold = this.maxmemory;


            logger.info("Actual memory limit: ["+ this.maxmemory +"]");

            // use temp file, if no target file was specified
            try {
                if (this.target == null) this.target = File.createTempFile("cooccurrence", "map");
                this.target.deleteOnExit();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            ret.targetFile = this.target;

            return ret;
        }
    }

    private class CoOccurrencesCalculatorThread extends Thread implements Runnable {

        private final SequenceIterator<T> iterator;
        private final AtomicLong sequenceCounter;
        private int threadId;

        public CoOccurrencesCalculatorThread(int threadId, @NonNull SequenceIterator<T> iterator, @NonNull AtomicLong sequenceCounter) {
            this.iterator = iterator;
            this.sequenceCounter = sequenceCounter;
            this.threadId = threadId;

            this.setName("CoOccurrencesCalculatorThread " + threadId);
        }

        @Override
        public void run() {
            while (iterator.hasMoreSequences()) {
                Sequence<T> sequence = iterator.nextSequence();

                List<String> tokens = new ArrayList<>(sequence.asLabels());
    //            logger.info("Tokens size: " + tokens.size());
                for (int x = 0; x < sequence.getElements().size(); x++) {
                    int wordIdx = vocabCache.indexOf(tokens.get(x));
                    if (wordIdx < 0) continue;
                    String w1 = vocabCache.wordFor(tokens.get(x)).getLabel();

                    // THIS iS SAFE TO REMOVE, NO CHANCE WE'll HAVE UNK WORD INSIDE SEQUENCE
                    /*if(w1.equals(Glove.UNK))
                        continue;
                    */

                    int windowStop = Math.min(x + windowSize + 1,tokens.size());
                    for(int j = x; j < windowStop; j++) {
                        int otherWord = vocabCache.indexOf(tokens.get(j));
                        if (otherWord < 0) continue;
                        String w2 = vocabCache.wordFor(tokens.get(j)).getLabel();

                        if(w2.equals(Glove.DEFAULT_UNK) || otherWord == wordIdx) {
                            continue;
                        }


                        T tokenX  = vocabCache.wordFor(tokens.get(x));
                        T tokenJ = vocabCache.wordFor(tokens.get(j));
                        double nWeight = 1.0 / (j - x + Nd4j.EPS_THRESHOLD);

                        while (getMemoryFootprint() >= getMemoryThreshold()) {
                            try {
                                shadowThread.invoke();
                                /*lock.readLock().lock();
                                int size = coOccurrenceCounts.size();
                                lock.readLock().unlock();
                                */
                                if (threadId == 0) logger.debug("Memory consuimption > threshold: {footrpint: ["+ getMemoryFootprint()+"], threshold: [" + getMemoryThreshold() +"] }");
                                Thread.sleep(10000);
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            } finally {

                            }
                        }
                        /*
                        if (getMemoryFootprint() == 0) {
                            logger.info("Zero size!");
                        }
                        */

                        try {
                            lock.readLock().lock();
                            if (wordIdx < otherWord) {
                                coOccurrenceCounts.incrementCount(tokenX, tokenJ, nWeight);
                                if (symmetric) {
                                    coOccurrenceCounts.incrementCount(tokenJ, tokenX, nWeight);
                                }
                            } else {
                                coOccurrenceCounts.incrementCount(tokenJ, tokenX, nWeight);

                                if (symmetric) {
                                    coOccurrenceCounts.incrementCount(tokenX, tokenJ, nWeight);
                                }
                            }
                        } finally {
                            lock.readLock().unlock();
                        }
                    }
                }

                sequenceCounter.incrementAndGet();
            }
        }
    }

    /**
     * This class is designed to provide shadow copy functionality for CoOccurence maps, since with proper corpus size you can't fit such a map into memory
     *
     */
    private class ShadowCopyThread extends Thread implements Runnable {

        private AtomicBoolean isFinished = new AtomicBoolean(false);
        private AtomicBoolean isTerminate = new AtomicBoolean(false);
        private AtomicBoolean isInvoked = new AtomicBoolean(false);
        private AtomicBoolean shouldInvoke = new AtomicBoolean(false);

        // file that contains resuts from previous runs
        private File[] tempFiles;
        private RoundCount counter;

        public ShadowCopyThread() {
            try {

                counter = new RoundCount(1);
                tempFiles = new File[2];

                tempFiles[0] = File.createTempFile("aco", "tmp");
                tempFiles[1] = File.createTempFile("aco", "tmp");

                tempFiles[0].deleteOnExit();
                tempFiles[1].deleteOnExit();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            this.setName("ACO ShadowCopy thread");
        }

        @Override
        public void run() {
            /*
                  Basic idea is pretty simple: run quetly, untill memory gets filled up to some high volume.
                  As soon as this happens - execute shadow copy.
            */
            while (!isFinished.get() && !isTerminate.get()) {
                // check used memory. if memory use below threshold - sleep for a while. if above threshold - invoke copier

                if (getMemoryFootprint() > getMemoryThreshold()  || (shouldInvoke.get() && !isInvoked.get())) {
                    // we'll just invoke copier, nothing else
                    shouldInvoke.compareAndSet(true, false);
                    invokeBlocking();
                } else {
                    try {
                        /*
                               commented and left here for future debugging purposes, if needed

                                //lock.readLock().lock();
                                //int size = coOccurrenceCounts.size();
                                //lock.readLock().unlock();
                                //logger.info("Current memory situation: {size: [" +size+ "], footprint: [" + getMemoryFootprint()+"], threshold: ["+ getMemoryThreshold() +"]}");
                         */
                        Thread.sleep(1000);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }

        /**
         * This methods advises shadow copy process to start
         */
        public void invoke() {
            shouldInvoke.compareAndSet(false, true);
        }

        /**
         * This methods dumps cooccurrence map into save file.
         * Please note: this method is synchronized and will block, until complete
         */
        public synchronized void invokeBlocking() {
            if (getMemoryFootprint() < getMemoryThreshold() && !isFinished.get()) return;

            int numberOfLinesSaved = 0;

            isInvoked.set(true);

            logger.debug("Memory purge started.");

            /*
                Basic plan:
                    1. Open temp file
                    2. Read that file line by line
                    3. For each read line do synchronization in memory > new file direction
             */

            counter.tick();

            CountMap<T> localMap;
            try {
                // in any given moment there's going to be only 1 WriteLock, due to invokeBlocking() being synchronized call
                lock.writeLock().lock();



                // obtain local copy of CountMap
                 localMap = coOccurrenceCounts;

                // set new CountMap, and release write lock
                coOccurrenceCounts = new CountMap<T>();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.writeLock().unlock();
            }

            try {

                File file = null;
                if (!isFinished.get()) {
                    file = tempFiles[counter.previous()];
                } else file = targetFile;


            //    PrintWriter pw = new PrintWriter(file);

                int linesRead = 0;

                logger.debug("Saving to: ["+ counter.get()+"], Reading from: [" + counter.previous()+"]");
                CoOccurenceReader<T> reader = new BinaryCoOccurrenceReader<>(tempFiles[counter.previous()], vocabCache, localMap);
                CoOccurrenceWriter<T> writer = (isFinished.get()) ? new ASCIICoOccurrenceWriter<T>(targetFile): new BinaryCoOccurrenceWriter<T>(tempFiles[counter.get()]);
                while (reader.hasMoreObjects()) {
                    CoOccurrenceWeight<T> line = reader.nextObject();

                    if (line != null) {
                        writer.writeObject(line);
                        numberOfLinesSaved++;
                        linesRead++;
                    }
                }
                reader.finish();

                logger.debug("Lines read: [" + linesRead + "]");

                //now, we can dump the rest of elements, which were not presented in existing dump
                Iterator<Pair<T, T>> iterator = localMap.getPairIterator();
                while (iterator.hasNext()) {
                    Pair<T, T> pair = iterator.next();
                    double mWeight = localMap.getCount(pair);
                    CoOccurrenceWeight<T> object = new CoOccurrenceWeight<>();
                    object.setElement1(pair.getFirst());
                    object.setElement2(pair.getSecond());
                    object.setWeight(mWeight);

                    writer.writeObject(object);

                    numberOfLinesSaved++;
                    //      if (numberOfLinesSaved % 100000 == 0) logger.info("Lines saved: [" + numberOfLinesSaved +"]");
                }

                writer.finish();

            /*
                SentenceIterator sIterator =  new PrefetchingSentenceIterator.Builder(new BasicLineIterator(tempFiles[counter.get()]))
                        .setFetchSize(500000)
                        .build();


                int linesRead = 0;
                while (sIterator.hasNext()) {
                    //List<Writable> list = new ArrayList<>(reader.next());
                    String sentence = sIterator.nextSentence();
                    if (sentence == null || sentence.isEmpty()) continue;
                    String[] strings = sentence.split(" ");


                    // first two elements are integers - vocab indexes
                    //T element1 = vocabCache.wordFor(vocabCache.wordAtIndex(list.get(0).toInt()));
                    //T element2 = vocabCache.wordFor(vocabCache.wordAtIndex(list.get(1).toInt()));
                    T element1 = vocabCache.elementAtIndex(Integer.valueOf(strings[0]));
                    T element2 = vocabCache.elementAtIndex(Integer.valueOf(strings[1]));

                    // getting third element, previously stored weight
                    double sWeight = Double.valueOf(strings[2]);  // list.get(2).toDouble();

                    // now, since we have both elements ready, we can check this pair against inmemory map
                        double mWeight = localMap.getCount(element1, element2);
                        if (mWeight <= 0) {
                            // this means we have no such pair in memory, so we'll do nothing to sWeight
                        } else {
                            // since we have new weight value in memory, we should update sWeight value before moving it off memory
                            sWeight += mWeight;

                            // original pair can be safely removed from CountMap
                            localMap.removePair(element1,element2);
                        }

                        StringBuilder builder = new StringBuilder().append(element1.getIndex()).append(" ").append(element2.getIndex()).append(" ").append(sWeight);
                        pw.println(builder.toString());
                        numberOfLinesSaved++;
                        linesRead++;

                   // if (numberOfLinesSaved % 100000 == 0) logger.info("Lines saved: [" + numberOfLinesSaved +"]");
                  //  if (linesRead % 100000 == 0) logger.info("Lines read: [" + linesRead +"]");
                }
                */
/*
                logger.info("Lines read: [" + linesRead + "]");

                //now, we can dump the rest of elements, which were not presented in existing dump
                Iterator<Pair<T, T>> iterator = localMap.getPairIterator();
                while (iterator.hasNext()) {
                    Pair<T, T> pair = iterator.next();
                    double mWeight = localMap.getCount(pair);

                    StringBuilder builder = new StringBuilder().append(pair.getFirst().getIndex()).append(" ").append(pair.getFirst().getIndex()).append(" ").append(mWeight);
                    pw.println(builder.toString());
                    numberOfLinesSaved++;

              //      if (numberOfLinesSaved % 100000 == 0) logger.info("Lines saved: [" + numberOfLinesSaved +"]");
                }

                pw.flush();
                pw.close();

*/

                // just a hint for gc
                localMap = null;
                //sIterator.finish();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            logger.info("Number of word pairs saved so far: [" + numberOfLinesSaved + "]");
            isInvoked.set(false);
        }

        /**
         * This method provides soft finish ability for shadow copy process.
         * Please note: it's blocking call, since it requires for final merge.
         */
        public void finish() {
            if (this.isFinished.get()) return;

            this.isFinished.set(true);
            invokeBlocking();
        }

        /**
         * This method provides hard fiinish ability for shadow copy process
         */
        public void terminate() {
            this.isTerminate.set(true);
        }
    }
}