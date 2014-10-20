package org.deeplearning4j.bagofwords.vectorizer;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.dataset.DataSet;
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

/**
 * @author Adam Gibson
 */
public class TfIdfVectorizerTest {

    private static Logger log = LoggerFactory.getLogger(TfIdfVectorizerTest.class);

    @Test
    public void testTfIdfVectorizer() throws Exception {
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> labels = Arrays.asList("label1","label2");
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizer = new TfidfVectorizer.Builder().minWords(1).stopWords(new ArrayList<String>())
                .tokenize(tokenizerFactory).labels(labels).iterate(iter).build();
        vectorizer.fit();
        VocabWord word = vectorizer.vocab().wordFor("This");
        assumeNotNull(word);
    }


}
