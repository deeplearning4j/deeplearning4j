package org.deeplearning4j.util;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.text.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.util.ContextLabelRetriever;
import org.junit.Test;
import org.junit.Before;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Basic test case for the context label test
 */
public class ContextLabelTest {
    private static Logger log = LoggerFactory.getLogger(ContextLabelTest.class);
    private TokenizerFactory tokenizerFactory;

    @Before
    public void init() throws  Exception {
          if(tokenizerFactory == null) {
              tokenizerFactory = new UimaTokenizerFactory(false);
          }
    }

    @Test
    public void testBasicLabel() {
        String labeledSentence = "<NEGATIVE> This sucks really bad </NEGATIVE> .";
        Pair<String,MultiDimensionalMap<Integer,Integer,String>> ret = ContextLabelRetriever.stringWithLabels(labeledSentence,tokenizerFactory);
        //positive and none
        assertEquals(2, ret.getSecond().size());
        List<String> vals = new ArrayList<>(ret.getSecond().values());
        assertEquals(true,vals.contains("negative"));
        assertEquals(true,vals.contains("none"));
        assertEquals("this suck realli bad .",ret.getFirst());
    }


}
