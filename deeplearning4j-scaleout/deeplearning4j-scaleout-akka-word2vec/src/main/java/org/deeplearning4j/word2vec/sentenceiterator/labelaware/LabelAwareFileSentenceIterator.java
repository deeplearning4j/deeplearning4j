package org.deeplearning4j.word2vec.sentenceiterator.labelaware;

import org.deeplearning4j.word2vec.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;

import java.io.File;

/**
 *
 * Label aware sentence iterator
 *
 * @author Adam Gibson
 */
public class LabelAwareFileSentenceIterator extends FileSentenceIterator implements LabelAwareSentenceIterator {
    /**
     * Takes a single file or directory
     *
     * @param preProcessor the sentence pre processor
     * @param file         the file or folder to iterate over
     */
    public LabelAwareFileSentenceIterator(SentencePreProcessor preProcessor, File file) {
        super(preProcessor, file);
    }

    public LabelAwareFileSentenceIterator(File dir) {
        super(dir);
    }

    @Override
    public String currentLabel() {
        return file.getParentFile().getName();
    }
}
