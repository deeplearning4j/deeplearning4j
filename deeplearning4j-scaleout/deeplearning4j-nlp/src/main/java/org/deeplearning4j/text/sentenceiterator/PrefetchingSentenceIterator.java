package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Wrapper over SentenceIterator, that allows background prefetch.
 *
 * NOT READY FOR USE, PLEASE DON" USE
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class PrefetchingSentenceIterator implements SentenceIterator {

    private SentenceIterator sourceIterator;
    private int fetchSize;
    private LinkedBlockingQueue<String> buffer = new LinkedBlockingQueue<>();
    private AsyncIteratorReader reader;
    private SentencePreProcessor preProcessor;

    private PrefetchingSentenceIterator() {

    }

    /**
     * Here we start async readers
     */
    private void init() {
        reader = new AsyncIteratorReader(sourceIterator, fetchSize, buffer, this.preProcessor);
        reader.start();
    }

    @Override
    public String nextSentence() {
        if (buffer.size() > 0) {
            return buffer.poll();
        } else if (reader.hasMoreLines()) {
            try {
                return buffer.take();
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        } else return null;
    }

    @Override
    public boolean hasNext() {
        return (reader != null) ? reader.hasMoreLines() : false;
    }

    @Override
    public void reset() {
        if (reader != null) reader.reset();
    }

    @Override
    public void finish() {
        if (reader != null) reader.terminate();
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
        if (reader != null) reader.terminate();
        super.finalize();
    }

    public static class Builder {
        private SentenceIterator iterator;
        private int fetchSize = 100;
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
        private LinkedBlockingQueue<String> buffer;
        private AtomicBoolean shouldWork = new AtomicBoolean(true);
        private AtomicBoolean shouldTerminate = new AtomicBoolean(false);
        private ReentrantReadWriteLock lock =  new ReentrantReadWriteLock();
        private SentencePreProcessor preProcessor;

        public AsyncIteratorReader(@NonNull SentenceIterator iterator, @NonNull int fetchSize, @NonNull LinkedBlockingQueue<String> buffer, SentencePreProcessor preProcessor) {
            this.iterator = iterator;
            this.fetchSize = fetchSize;
            this.buffer = buffer;
            this.preProcessor = preProcessor;

            this.setName("AsyncIteratorReader thread");
        }

        @Override
        public void run() {
            while (!shouldTerminate.get()) {
                while (!shouldTerminate.get() && iterator.hasNext()) {

                        int cnt = 0;
                        if (buffer.size() < fetchSize)
                            try {
                         //       lock.writeLock().lock();
                                while (!shouldTerminate.get() && cnt < fetchSize && iterator.hasNext()) {
                                    lock.writeLock().lock();
                                    buffer.add((this.preProcessor == null) ? iterator.nextSentence() : this.preProcessor.preProcess(iterator.nextSentence()));
                                    lock.writeLock().unlock();
                                    cnt++;
                                }
                            } finally {
                      //         lock.writeLock().unlock();
                            }
                        else try {
                                Thread.sleep(10);
                            } catch (Exception e) {}
                }
            }
        }

        public boolean hasMoreLines() {
            try {
                lock.readLock().lock();
                if (buffer.size() > 0) return true;
                else return iterator.hasNext();
            } finally {
                lock.readLock().unlock();
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
