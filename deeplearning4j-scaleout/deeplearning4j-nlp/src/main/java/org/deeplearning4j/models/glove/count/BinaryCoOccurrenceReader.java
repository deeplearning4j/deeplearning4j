package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.io.*;

/**
 * Binary implementation of CoOccurenceReader interface, used to provide off-memory storage for cooccurrence maps generated for GloVe
 *
 * @author raver119@gmail.com
 */
public class BinaryCoOccurrenceReader<T extends SequenceElement> implements CoOccurenceReader<T> {
    private VocabCache<T> vocabCache;
    private DataInputStream inputStream;
    private DataOutputStream outputStream;
    private File file;


    public BinaryCoOccurrenceReader(@NonNull File file, @NonNull VocabCache<T> vocabCache) {
        this.vocabCache = vocabCache;
        this.file = file;
        try {
            inputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(file), 10 * 1024 * 1024));
            //inputStream = new BufferedInputStream(new FileInputStream(file), 1024 * 1024);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean hasMoreObjects() {
        try {
            return inputStream.available() > 0;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public CoOccurrenceWeight<T> nextObject() {
        try {
            CoOccurrenceWeight<T> ret = new CoOccurrenceWeight<>();
            ret.setElement1(vocabCache.elementAtIndex(inputStream.readInt()));
            ret.setElement2(vocabCache.elementAtIndex(inputStream.readInt()));
            ret.setWeight(inputStream.readDouble());

            return ret;
        } catch (Exception e) {
            return null;
        }
    }

    @Override
    public void finish() {
        try {
            if (inputStream != null) inputStream.close();
        } catch (Exception e) {
            //
        }

        try {
            if (outputStream != null) outputStream.close();
        } catch (Exception e) {
            //
        }
    }
}
