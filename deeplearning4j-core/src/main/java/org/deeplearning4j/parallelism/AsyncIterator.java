package org.deeplearning4j.parallelism;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Iterator;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Asynchronous Iterator for better performance of iterators in dl4j-nn & dl4j-nlp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AsyncIterator<T extends Object> implements Iterator<T> {
    @Getter
    protected BlockingQueue<T> buffer;
    protected ReaderThread<T> thread;
    protected Iterator<T> iterator;
    @Getter
    protected T terminator = (T) new Object();
    protected T nextElement;
    protected AtomicBoolean shouldWork = new AtomicBoolean(true);

    public AsyncIterator(@NonNull Iterator<T> iterator, int bufferSize) {
        this.buffer = new LinkedBlockingQueue<>(bufferSize);
        this.iterator = iterator;

        thread = new ReaderThread<>(iterator, this.buffer, terminator);
        thread.start();
    }

    public AsyncIterator(@NonNull Iterator<T> iterator) {
        this(iterator, 1024);
    }

    @Override
    public boolean hasNext() {
        try {
            if (nextElement != null && nextElement != terminator) {
                return true;
            }
            nextElement = buffer.take();
            if (nextElement == terminator)
                return false;
            return true;
        } catch (Exception e) {
            log.error("Premature end of loop!");
            return false;
        }
    }

    @Override
    public T next() {
        T temp = nextElement;
        nextElement = null;
        return temp;
    }

    @Override
    public void remove() {
        // no-op
    }

    public void shutdown() {
        if (shouldWork.get()) {
            shouldWork.set(false);
            thread.interrupt();
            try {
                // Shutdown() should be a synchronous operation since the iterator is reset after shutdown() is
                // called in AsyncLabelAwareIterator.reset().
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            nextElement = terminator;
        }
    }


    private class ReaderThread<T> extends Thread implements Runnable {
        private BlockingQueue<T> buffer;
        private Iterator<T> iterator;
        private T terminator;

        public ReaderThread(Iterator<T> iterator, BlockingQueue<T> buffer, T terminator) {
            this.buffer = buffer;
            this.iterator = iterator;
            this.terminator = terminator;

            setDaemon(true);
            setName("AsyncIterator Reader thread");
        }

        @Override
        public void run() {
            try {
                while (iterator.hasNext() && shouldWork.get()) {
                    T smth = iterator.next();

                    if (smth != null)
                        buffer.put(smth);
                }
                buffer.put(terminator);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                // do nothing
                shouldWork.set(false);
            } catch (Exception e) {
                // TODO: pass that forward
                throw new RuntimeException(e);
            }
        }
    }
}
