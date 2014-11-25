package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;


/**
 * Created by agibsonccc on 8/31/14.
 */
public class Word2VecTests {

    private static Logger log = LoggerFactory.getLogger(Word2VecTests.class);


    @Before
    public void before() {
        new File("word2vec-index").delete();
    }

    @Test
    public void testWord2VecRunThroughVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());
        new File("cache.ser").delete();


        TokenizerFactory t = new UimaTokenizerFactory();

        InMemoryLookupCache cache = new InMemoryLookupCache.Builder().vectorLength(100).useAdaGrad(false).lr(0.025f).build();

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

 

}
