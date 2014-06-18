package org.deeplearning4j.word2vec.sentenceiterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

import java.io.File;
import java.io.IOException;

/**
 * Each line is a sentence
 *
 * @author Adam Gibson
 */
public class LineSentenceIterator extends BaseSentenceIterator {

    private File file;
    private LineIterator iter;

    public LineSentenceIterator(File f) {
        if (!f.exists() || !f.isFile())
            throw new IllegalArgumentException("Please specify an existing file");
        this.file = f;
        try {
            iter = FileUtils.lineIterator(file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String nextSentence() {
        String line = iter.nextLine();
        if (preProcessor != null) {
            line = preProcessor.preProcess(line);
        }
        return line;
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public void reset() {
        try {
            if (iter != null)
                iter.close();
            iter = FileUtils.lineIterator(file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }


}
