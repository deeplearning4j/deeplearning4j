package org.deeplearning4j.word2vec;

import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
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


        Word2Vec vec = new Word2Vec.Builder()
               .windowSize(5).iterate(iter).tokenizerFactory(t).build();
        vec.fit();
    }

}
