package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;

import org.deeplearning4j.util.ThreadUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Wrapper over SentenceIterator, that allows background prefetch from original SentenceIterator
 * It could be useful, if your SentencePreProcessor implementation is CPU intensive as well as whole pipeline behind iterator is cpu intensive too.
 * This iterator will allow you to split workload in two different threads
 *
 * WORK IS IN PROGRESS, DO NOT USE PLEASE
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class PrefetchingSentenceIterator implements SentenceIterator {

    private SentenceIterator sourceIterator;
    private int fetchSize;
    private AsyncIteratorReader reader;
    private SentencePreProcessor preProcessor;

    protected static final Logger log = LoggerFactory.getLogger(PrefetchingSentenceIterator.class);

    private PrefetchingSentenceIterator() {

    }

    /**
     * Here we start async readers
     */
    private void init() {
        reader = new AsyncIteratorReader(sourceIterator, fetchSize, this.preProcessor);
        reader.start();
    }

    @Override
    public String nextSentence() {
        return reader.nextLine();
    }

    @Override
    public boolean hasNext() {
        return (reader != null) ? reader.hasMoreLines() : false;
    }

    @Override
    public void reset() {
        if (reader != null)
            reader.reset();
    }

    @Override
    public void finish() {
        if (reader != null)
            reader.terminate();
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    protected void finalize() throws Throwable {
        if (reader != null)
            reader.terminate();
        super.finalize();
    }

    public static class Builder {
        private SentenceIterator iterator;
        private int fetchSize = 10000;
        private SentencePreProcessor preProcessor;

        public Builder(@NonNull SentenceIterator iterator) {
            this.iterator = iterator;
        }

        public Builder setFetchSize(int fetchSize) {
            this.fetchSize = fetchSize;
            return this;
        }

        public Builder setSentencePreProcessor(@NonNull SentencePreProcessor preProcessor) {
            this.preProcessor = preProcessor;
            return this;
        }

        public PrefetchingSentenceIterator build() {
            PrefetchingSentenceIterator pre = new PrefetchingSentenceIterator();
            pre.sourceIterator = this.iterator;
            pre.fetchSize = this.fetchSize;
            pre.preProcessor = this.preProcessor;

            pre.init();
            return pre;
        }
    }

    private class AsyncIteratorReader extends Thread implements Runnable {
        private SentenceIterator iterator;
        private int fetchSize;
        private AtomicBoolean shouldTerminate = new AtomicBoolean(false);
        private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
        private SentencePreProcessor preProcessor;
        private AtomicBoolean isRunning = new AtomicBoolean(true);
        private ArrayBlockingQueue<String> buffer;

        public AsyncIteratorReader(@NonNull SentenceIterator iterator, int fetchSize,
                        SentencePreProcessor preProcessor) {
            this.iterator = iterator;
            this.fetchSize = fetchSize;
            this.preProcessor = preProcessor;

            buffer = new ArrayBlockingQueue<>(fetchSize * 3);
            this.setName("AsyncIteratorReader thread");
            this.setDaemon(true);
        }

        @Override
        public void run() {
            while (!shouldTerminate.get()) {
                if (iterator.hasNext())
                    isRunning.set(true);
                else
                    ThreadUtils.uncheckedSleep(50);
                while (!shouldTerminate.get() && iterator.hasNext()) {

                    int cnt = 0;
                    if (buffer.size() < fetchSize) {
                        while (!shouldTerminate.get() && cnt < fetchSize && iterator.hasNext()) {
                            try {
                                lock.writeLock().lock();
                                String line = iterator.nextSentence();
                                if (line != null)
                                    buffer.add((this.preProcessor == null) ? line : this.preProcessor.preProcess(line));
                            } finally {
                                lock.writeLock().unlock();
                            }
                            cnt++;
                        }
                        //                            log.info("Lines added: [" + cnt + "], buffer size: [" + buffer.size() + "]");
                    } else
                        ThreadUtils.uncheckedSleep(10);
                }
                isRunning.set(false);
            }
        }

        public String nextLine() {
            if (!buffer.isEmpty())
                return buffer.poll();

            try {
                return buffer.poll(2L, TimeUnit.SECONDS);
            } catch (Exception e) {
                return null;
            }
        }

        public boolean hasMoreLines() {
            if (!buffer.isEmpty())
                return true;

            try {
                this.lock.readLock().lock();
                return iterator.hasNext() || !buffer.isEmpty();
            } finally {
                this.lock.readLock().unlock();
            }
        }

        public void reset() {
            try {
                lock.writeLock().lock();
                buffer.clear();
                iterator.reset();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                lock.writeLock().unlock();
            }
        }

        public void terminate() {
            shouldTerminate.set(true);
        }
    }
}
