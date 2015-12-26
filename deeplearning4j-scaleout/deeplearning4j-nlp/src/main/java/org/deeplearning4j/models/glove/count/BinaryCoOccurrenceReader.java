package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
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
    private AtomicInteger arrayPosition = new AtomicInteger(0);
    private ArrayBlockingQueue<CoOccurrenceWeight<T>> buffer;

    public BinaryCoOccurrenceReader(@NonNull File file, @NonNull VocabCache<T> vocabCache) {
        this.vocabCache = vocabCache;
        this.file = file;
        try {
            inputStream = new BufferedInputStream(new FileInputStream(file), 100 * 1024 * 1024);
            //inputStream = new BufferedInputStream(new FileInputStream(file), 1024 * 1024);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean hasMoreObjects() {
        try {
            return buffer.size() > 0 || inputStream.available() > 0;
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
            if (inputStream != null) inputStream.close();
        } catch (Exception e) {
            //
        }
    }

    private class AsyncDeserializationThread extends Thread implements Runnable {
        private int threadId;
        private byte[] arrayReference;
        private ArrayBlockingQueue<CoOccurrenceWeight<T>> targetBuffer;
        private AtomicInteger pointer;

        public AsyncDeserializationThread(int threadId, byte[] array, ArrayBlockingQueue<CoOccurrenceWeight<T>> targetBuffer, AtomicInteger sharedPointer) {
            this.threadId = threadId;
            this.arrayReference = array;
            this.targetBuffer = targetBuffer;
            this.pointer = sharedPointer;

            setName("AsynDeserialization thread " + threadId);
        }

        @Override
        public void run() {
            while (pointer.get() < arrayReference.length) {
                int position = pointer.getAndAdd(16);

                int e1idx = ByteBuffer.wrap(arrayReference).getInt(position);
                int e2idx = ByteBuffer.wrap(arrayReference).getInt(position + 4);
                double eW = ByteBuffer.wrap(arrayReference).getDouble( position + 8);

                CoOccurrenceWeight<T> object = new CoOccurrenceWeight<T>();
                object.setElement1(vocabCache.elementAtIndex(e1idx));
                object.setElement2(vocabCache.elementAtIndex(e2idx));
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
