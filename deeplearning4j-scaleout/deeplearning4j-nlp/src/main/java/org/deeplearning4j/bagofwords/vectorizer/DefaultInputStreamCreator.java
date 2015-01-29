package org.deeplearning4j.bagofwords.vectorizer;

import org.deeplearning4j.models.word2vec.InputStreamCreator;
import org.deeplearning4j.text.documentiterator.DocumentIterator;

import java.io.InputStream;

/**
 * Created by agibsonccc on 10/20/14.
 */
public class DefaultInputStreamCreator implements InputStreamCreator {
    private DocumentIterator iter;

    public DefaultInputStreamCreator(DocumentIterator iter) {
        this.iter = iter;
    }

    @Override
    public InputStream create() {
        return iter.nextDocument();
    }
}
