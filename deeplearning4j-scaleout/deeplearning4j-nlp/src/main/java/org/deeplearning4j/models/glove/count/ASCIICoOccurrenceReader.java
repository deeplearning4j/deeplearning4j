package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;

/**
 * @author raver119@gmail.com
 */
public class ASCIICoOccurrenceReader<T extends SequenceElement> implements CoOccurenceReader<T> {
    private File file;
    private PrintWriter writer;
    private SentenceIterator iterator;

    public ASCIICoOccurrenceReader(@NonNull File file, @NonNull VocabCache<T> vocabCache) {

    }

    public ASCIICoOccurrenceReader(@NonNull File file) {
        this.file = file;
        try {
            writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(file), 1024 * 1024));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean hasMoreObjects() {
        throw new UnsupportedOperationException("Not supported right now");
    }

    @Override
    public CoOccurrenceWeight<T> nextObject() {
        throw new UnsupportedOperationException("Not supported right now");
    }



    @Override
    public void finish() {
        try {
            if (writer != null) {
                writer.flush();
                writer.close();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
}
