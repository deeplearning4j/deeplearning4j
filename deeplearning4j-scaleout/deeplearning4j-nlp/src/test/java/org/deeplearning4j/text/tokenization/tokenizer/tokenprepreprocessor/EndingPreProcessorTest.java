package org.deeplearning4j.text.tokenization.tokenizer.tokenprepreprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.junit.Test;

/**
 * Created by agibsonccc on 10/18/14.
 */
public class EndingPreProcessorTest {
    @Test
    public void testPreProcessor() {
        TokenPreProcess preProcess = new EndingPreProcessor();
        String endingTest = "ending";
        assertEquals("end",preProcess.preProcess(endingTest));

    }

}
