package org.deeplearning4j.word2vec.sentenceiterator.labelaware;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;

import java.io.File;
import java.lang.reflect.Field;

/**
 *
 * Uima sentence iterator that is aware of the current file
 * @author Adam Gibson
 */
public class LabelAwareUimaSentenceIterator extends UimaSentenceIterator implements LabelAwareSentenceIterator {

    public LabelAwareUimaSentenceIterator(SentencePreProcessor preProcessor, String path, AnalysisEngine engine) {
        super(preProcessor, path, engine);
    }

    public LabelAwareUimaSentenceIterator(String path, AnalysisEngine engine) {
        super(path, engine);
    }


    /**
     * Returns the current label for nextSentence()
     *
     * @return the label for nextSentence()
     */
    @Override
    public String currentLabel() {

        try {
            /**
             * Little bit hacky, but most concise way to do it.
             * Get the parent collection reader's current file.
             * The collection reader is basically a wrapper for a file iterator.
             * We can use this to ge the current file for the iterator.
             */
            Field f = reader.getClass().getDeclaredField("currentFile");
            f.setAccessible(true);
            File file = (File) f.get(reader);
            return file.getParentFile().getName();
        }catch(Exception e) {
            throw new RuntimeException(e);
        }

    }
}
