package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class StemmingPreprocessorTest {

    @Test
    public void testPreProcess() throws Exception {
        StemmingPreprocessor preprocessor = new StemmingPreprocessor();

        String test = "TESTING.";

        String output = preprocessor.preProcess(test);

        System.out.println("Output: " + output);
        assertEquals("test", output);
    }
}