package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;


/**
 * Created by agibsonccc on 8/31/14.
 */
public class Word2VecTests {

    private static Logger log = LoggerFactory.getLogger(Word2VecTests.class);


    @Test
    public void testWord2VecRunThroughVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());
        new File("cache.ser").delete();


        TokenizerFactory t = new UimaTokenizerFactory();

        InMemoryLookupCache cache = new InMemoryLookupCache(100,false,0.025f);

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1).iterations(5)
                .layerSize(100)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());


        vec.fit();


        double sim = vec.similarity("Adam","deeplearning4j");
        new File("cache.ser").delete();

    }

    @Test
    public void testWord2VecRunThrough() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic/word2vec.txt");
        File file = resource.getFile().getParentFile();
        DocumentIterator iter = new FileDocumentIterator(file);
        new File("cache.ser").delete();


        TokenizerFactory t = new DefaultTokenizerFactory();

        InMemoryLookupCache cache = new InMemoryLookupCache(100,true,0.025f);
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1).layerSize(100).stopWords(new ArrayList<String>())
                .vocabCache(cache)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(),vec.getStopWords());
        vec.fit();

        assertTrue(Arrays.equals(cache.wordFor("This").getCodes(),new int[]{0}));
        assertTrue(Arrays.equals(cache.wordFor("This").getPoints(),new int[]{0}));

        assertTrue(Arrays.equals(cache.wordFor("test").getCodes(),new int[]{1}));
        assertTrue(Arrays.equals(cache.wordFor("test").getPoints(),new int[]{0}));


        assertTrue(vec.getCache().numWords() > 0);

        assertEquals(0,cache.indexOf("This"));
        assertEquals(1,cache.indexOf("test"));


        new File("cache.ser").delete();

    }

}
