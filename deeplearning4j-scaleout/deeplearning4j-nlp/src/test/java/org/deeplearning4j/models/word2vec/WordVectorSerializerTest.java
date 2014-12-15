package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.loader.WordVectorSerializer;
import org.junit.Before;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

/**
 * Created by agibsonccc on 9/21/14.
 */
public class WordVectorSerializerTest {

    private File textFile,binaryFile;

    @Before
    public void before() throws Exception {
        if(textFile == null)
            textFile = new ClassPathResource("vec.txt").getFile();
        if(binaryFile == null)
            binaryFile = new ClassPathResource("vec.bin").getFile();
    }

    @Test
    public void testLoaderText() throws IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(textFile.getAbsolutePath(), false);
        assertEquals(5,vec.vocab().numWords());
        assertTrue(vec.vocab().numWords() > 0);
    }

    @Test
    public void testLoaderBinary() throws  IOException {
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(binaryFile.getAbsolutePath(), true);
        assertEquals(5,vec.vocab().numWords());

    }

}
