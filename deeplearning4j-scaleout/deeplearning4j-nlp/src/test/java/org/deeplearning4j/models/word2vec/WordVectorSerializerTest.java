package org.deeplearning4j.models.word2vec;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
        FileUtils.deleteDirectory(new File("word2vec-index"));
n
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
        assertEquals(2,vec.vocab().numWords());

    }

    @Test
    public void testCurrentFile() throws Exception {
        Nd4j.dtype = DataBuffer.FLOAT;
        String url = "https://docs.google.com/uc?export=download&confirm=LDs-&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM";
        String path = "GoogleNews-vectors-negative300.bin.gz";
        File toDl = new File(path);
        if(!toDl.exists())
            FileUtils.copyURLToFile(new URL(url),toDl);
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(toDl.getAbsolutePath(), true);
        assertEquals(3000000,vec.vocab().numWords());

    }


}
