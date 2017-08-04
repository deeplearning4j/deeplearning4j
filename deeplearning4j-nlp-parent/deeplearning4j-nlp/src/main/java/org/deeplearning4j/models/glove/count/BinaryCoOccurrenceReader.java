package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Binary implementation of CoOccurenceReader interface, used to provide off-memory storage for cooccurrence maps generated for GloVe
 *
 * @author raver119@gmail.com
 */
public class BinaryCoOccurrenceReader<T extends SequenceElement> implements CoOccurenceReader<T> {
    private VocabCache<T> vocabCache;
    private InputStream inputStream;
    private File file;
    private ArrayBlockingQueue<CoOccurrenceWeight<T>> buffer;
    int workers = Math.max(Runtime.getRuntime().availableProcessors() - 1, 1);
    private StreamReaderThread readerThread;
    private CountMap<T> countMap;


    protected static final Logger logger = LoggerFactory.getLogger(BinaryCoOccurrenceReader.class);

    public BinaryCoOccurrenceReader(@NonNull File file, @NonNull VocabCache<T> vocabCache, CountMap<T> map) {
        this.vocabCache = vocabCache;
        this.file = file;
        this.countMap = map;
        buffer = new ArrayBlockingQueue<>(200000);

        try {
            inputStream = new BufferedInputStream(new FileInputStream(this.file), 100 * 1024 * 1024);
            //inputStream = new BufferedInputStream(new FileInputStream(file), 1024 * 1024);
            readerThread = new StreamReaderThread(inputStream);
            readerThread.start();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean hasMoreObjects() {

        if (!buffer.isEmpty())
            return true;

        try {
            return readerThread.hasMoreObjects() || !buffer.isEmpty();
        } catch (Exception e) {
            throw new RuntimeException(e);
            //return false;
        }
    }

    @Override
    public CoOccurrenceWeight<T> nextObject() {
        if (!buffer.isEmpty()) {
            return buffer.poll();
        } else {
            // buffer can be starved, or we're already at the end of file.
            if (readerThread.hasMoreObjects()) {
                try {
                    return buffer.poll(3, TimeUnit.SECONDS);
                } catch (Exception e) {
                    return null;
                }
            }
        }


        return null;
        /*
        try {
            CoOccurrenceWeight<T> ret = new CoOccurrenceWeight<>();
            ret.setElement1(vocabCache.elementAtIndex(inputStream.readInt()));
            ret.setElement2(vocabCache.elementAtIndex(inputStream.readInt()));
            ret.setWeight(inputStream.readDouble());
        
            return ret;
        } catch (Exception e) {
            return null;
        }
        */
    }

    @Override
    public void finish() {
        try {
            if (inputStream != null)
                inputStream.close();
        } catch (Exception e) {
            //
        }
    }

    private class StreamReaderThread extends Thread implements Runnable {
        private InputStream stream;
        private AtomicBoolean isReading = new AtomicBoolean(false);

        public StreamReaderThread(@NonNull InputStream stream) {
            this.stream = stream;
            isReading.set(false);
        }

        @Override
        public void run() {
            try {
                // we read pre-defined number of objects as byte array
                byte[] array = new byte[16 * 500000];
                while (true) {
                    int count = stream.read(array);

                    isReading.set(true);
                    if (count == 0)
                        break;

                    // now we deserialize them in separate threads to gain some speedup, if possible
                    List<AsyncDeserializationThread> threads = new ArrayList<>();
                    AtomicInteger internalPosition = new AtomicInteger(0);

                    for (int t = 0; t < workers; t++) {
                        threads.add(t, new AsyncDeserializationThread(t, array, buffer, internalPosition, count));
                        threads.get(t).start();
                    }

                    // we'll block this cycle untill all objects are fit into queue
                    for (int t = 0; t < workers; t++) {
                        try {
                            threads.get(t).join();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }

                    isReading.set(false);
                    if (count < array.length)
                        break;
                }

            } catch (Exception e) {
                isReading.set(false);
                throw new RuntimeException(e);
            }
        }

        public boolean hasMoreObjects() {
            try {
                return stream.available() > 0 || isReading.get();
            } catch (Exception e) {
                return false;
            } finally {
            }
        }
    }

    /**
     * Utility class that accepts byte array as input, and deserialize it into set of CoOccurrenceWeight objects
     */
    private class AsyncDeserializationThread extends Thread implements Runnable {
        private int threadId;
        private byte[] arrayReference;
        private ArrayBlockingQueue<CoOccurrenceWeight<T>> targetBuffer;
        private AtomicInteger pointer;
        private int limit;

        public AsyncDeserializationThread(int threadId, @NonNull byte[] array,
                        @NonNull ArrayBlockingQueue<CoOccurrenceWeight<T>> targetBuffer,
                        @NonNull AtomicInteger sharedPointer, int limit) {
            this.threadId = threadId;
            this.arrayReference = array;
            this.targetBuffer = targetBuffer;
            this.pointer = sharedPointer;
            this.limit = limit;


            setName("AsynDeserialization thread " + this.threadId);
        }

        @Override
        public void run() {
            ByteBuffer bB = ByteBuffer.wrap(arrayReference);
            int position = 0;
            while ((position = pointer.getAndAdd(16)) < this.limit) {
                if (position >= limit) {
                    continue;
                }


                int e1idx = bB.getInt(position);
                int e2idx = bB.getInt(position + 4);
                double eW = bB.getDouble(position + 8);


                CoOccurrenceWeight<T> object = new CoOccurrenceWeight<>();
                object.setElement1(vocabCache.elementAtIndex(e1idx));
                object.setElement2(vocabCache.elementAtIndex(e2idx));

                if (countMap != null) {
                    double mW = countMap.getCount(object.getElement1(), object.getElement2());

                    if (mW > 0) {
                        eW += mW;
                        countMap.removePair(object.getElement1(), object.getElement2());
                    }
                }
                object.setWeight(eW);

                try {
                    targetBuffer.put(object);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
