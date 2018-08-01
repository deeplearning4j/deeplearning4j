package org.deeplearning4j.text.tokenization.tokenizerfactory;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.junit.Test;

import static org.junit.Assert.*;

@Slf4j
public class NGramTokenizerFactoryTest {

    @Test
    public void testEmptyLines_1() throws Exception {
        val string = "";
        val tokens = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 2).create(string).getTokens();

        assertEquals(0, tokens.size());
    }

    @Test
    public void testEmptyLines_2() throws Exception {
        val string = "";
        val tf = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 2);
        tf.setTokenPreProcessor(new CommonPreprocessor());
        val tokens = tf.create(string).getTokens();

        assertEquals(0, tokens.size());
    }

    @Test
    public void testEmptyLines_3() throws Exception {
        val string = "\n";
        val tokens = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 2).create(string).getTokens();

        assertEquals(0, tokens.size());
    }

    @Test
    public void testEmptyLines_4() throws Exception {
        val string = "   ";
        val tokens = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 2).create(string).getTokens();

        assertEquals(0, tokens.size());
    }
}