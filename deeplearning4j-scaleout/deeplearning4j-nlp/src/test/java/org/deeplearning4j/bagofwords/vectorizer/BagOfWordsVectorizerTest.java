package org.deeplearning4j.bagofwords.vectorizer;


import static org.junit.Assume.*;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 *@author Adam Gibson
 */
public class BagOfWordsVectorizerTest {

    private static Logger log = LoggerFactory.getLogger(BagOfWordsVectorizerTest.class);

    @Test
    public void testBagOfWordsVectorizer() throws Exception {
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> labels = Arrays.asList("label1", "label2");
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizer = new BagOfWordsVectorizer.Builder()
                .minWords(1).stopWords(new ArrayList<String>())
                .tokenize(tokenizerFactory).iterate(iter).labels(labels).build();
        vectorizer.fit();
        VocabWord word = vectorizer.vocab().wordFor("This");
        assumeNotNull(word);
        assertEquals(word,vectorizer.vocab().tokenFor("This"));
        assertEquals(2,vectorizer.index().documents(word).length);


    }


}
