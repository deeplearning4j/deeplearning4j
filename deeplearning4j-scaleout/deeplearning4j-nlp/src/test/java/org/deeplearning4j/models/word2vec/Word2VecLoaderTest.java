package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.loader.Word2VecLoader;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;

/**
 * Created by agibsonccc on 9/21/14.
 */
public class Word2VecLoaderTest {

    @Test
    @Ignore
    public void testLoader() throws IOException {
        Word2Vec vec = Word2VecLoader.loadGoogleBinary("vectors.bin");
        assertTrue(vec.getCache().numWords() > 0);
    }

}
