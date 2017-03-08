package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;


/**
 * @author raver119@gmail.com
 */
public class PosUimaTokenizerFactoryTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testCreate1() throws Exception {
        String[] posTags = new String[] {"NN"};
        PosUimaTokenizerFactory factory = new PosUimaTokenizerFactory(Arrays.asList(posTags));
        Tokenizer tokenizer = factory.create("some test string");
        List<String> tokens = tokenizer.getTokens();
        System.out.println("Tokens: " + tokens);

        Assert.assertEquals(3, tokens.size());
        Assert.assertEquals("NONE", tokens.get(0));
        Assert.assertEquals("test", tokens.get(1));
        Assert.assertEquals("string", tokens.get(2));
    }

    @Test
    public void testCreate2() throws Exception {
        String[] posTags = new String[] {"NN"};
        PosUimaTokenizerFactory factory = new PosUimaTokenizerFactory(Arrays.asList(posTags), true);
        Tokenizer tokenizer = factory.create("some test string");
        List<String> tokens = tokenizer.getTokens();
        System.out.println("Tokens: " + tokens);

        Assert.assertEquals(2, tokens.size());
        Assert.assertEquals("test", tokens.get(0));
        Assert.assertEquals("string", tokens.get(1));
    }
}
