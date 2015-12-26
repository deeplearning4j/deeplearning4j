package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
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
        buffer = new ArrayBlockingQueue<CoOccurrenceWeight<T>>(20000);

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
        try {
            return buffer.size() > 0 || readerThread.hasMoreObjects();
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public CoOccurrenceWeight<T> nextObject() {
        if (buffer.size() > 0) {
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


        try {
            return buffer.poll(5, TimeUnit.SECONDS);
        } catch (Exception e) {
            return null;
        }
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
        logger.debug("Finishing on BinaryReader");
        try {
            if (inputStream != null) inputStream.close();
        } catch (Exception e) {
            //
        }
    }

    private class StreamReaderThread extends Thread implements Runnable {
        private InputStream stream;
        private AtomicBoolean isReading = new AtomicBoolean(false);

        public StreamReaderThread(@NonNull InputStream stream) {
            this.stream = stream;
        }

        @Override
        public void run() {
            try {
                isReading.set(true);
                // we read pre-defined number of objects as byte array
                byte[] array = new byte[16 * 100000];
                for (int count = stream.read(array); count >= 0; count = stream.read(array)) {

                    // now we deserialize them in separate threads to gain some speedup, if possible
                    List<AsyncDeserializationThread> threads = new ArrayList<>();
                    AtomicInteger internalPosition = new AtomicInteger(0);
                    for (int t = 0; t < workers; t++ ) {
                        threads.add(t, new AsyncDeserializationThread(t, array, buffer, internalPosition));
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
                }
                isReading.set(false);
            } catch (Exception e) {
                isReading.set(false);
                throw new RuntimeException(e);
            }
        }

        public boolean hasMoreObjects() {
            try {
                return isReading.get();
            } catch (Exception e) {
                return false;
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

        public AsyncDeserializationThread(int threadId, @NonNull byte[] array, @NonNull ArrayBlockingQueue<CoOccurrenceWeight<T>> targetBuffer, @NonNull AtomicInteger sharedPointer) {
            this.threadId = threadId;
            this.arrayReference = array;
            this.targetBuffer = targetBuffer;
            this.pointer = sharedPointer;

            setName("AsynDeserialization thread " + this.threadId);
        }

        @Override
        public void run() {
            while (pointer.get() < arrayReference.length) {
                int position = pointer.getAndAdd(16);

                if (position >= arrayReference.length) continue;

                //logger.debug("Position: [" + position + "], Array len: ["+ arrayReference.length+"] ");
                int e1idx = ByteBuffer.wrap(arrayReference).getInt(position);
                int e2idx = ByteBuffer.wrap(arrayReference).getInt(position + 4);
                double eW = ByteBuffer.wrap(arrayReference).getDouble( position + 8);

                CoOccurrenceWeight<T> object = new CoOccurrenceWeight<T>();
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
