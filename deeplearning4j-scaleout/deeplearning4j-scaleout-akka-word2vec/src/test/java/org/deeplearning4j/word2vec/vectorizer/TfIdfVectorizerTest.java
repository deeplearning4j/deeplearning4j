package org.deeplearning4j.word2vec.vectorizer;

import static org.junit.Assert.*;

import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
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
        TextVectorizer vectorizer = new TfidfVectorizer(iter,tokenizerFactory,labels);
        DataSet vectorized = vectorizer.vectorize();
        assertEquals(3,vectorized.numInputs());
        assertEquals(2,vectorized.numOutcomes());
        log.info("Vectorized " + vectorized.toString());
    }


}
