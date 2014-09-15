package org.deeplearning4j.models.word2vec;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;

/**
 * Created by agibsonccc on 8/31/14.
 */
public class Word2VecTests {

    private static Logger log = LoggerFactory.getLogger(Word2VecTests.class);

    @Test
    public void testWord2VecRunThrough() throws Exception {
        ClassPathResource resource = new ClassPathResource("/reuters/5250");
        File file = resource.getFile().getParentFile();
        SentenceIterator iter = new FileSentenceIterator(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return new InputHomogenization(sentence).transform();
            }
        },file);


        TokenizerFactory t = new UimaTokenizerFactory();


        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).vocabCache(new InMemoryLookupCache(50))
               .windowSize(5).iterate(iter).tokenizerFactory(t).build();
        vec.fit();
        assertTrue(vec.getCache().numWords() > 0);

        assertEquals(4,vec.getCache().wordFrequency("pearson"));



    }

}
