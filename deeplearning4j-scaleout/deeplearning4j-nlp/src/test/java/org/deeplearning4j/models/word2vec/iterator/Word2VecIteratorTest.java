package org.deeplearning4j.models.word2vec.iterator;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

/**
 * Created by agibsonccc on 3/5/15.
 */
public class Word2VecIteratorTest {
    private Word2Vec vec;

    @Before
    public void before() {
        if(vec == null) {

        }
    }

    @Test
    public void testLabeledExample() throws Exception {
        Word2VecDataSetIterator iter = new Word2VecDataSetIterator(vec,new LabelAwareFileSentenceIterator(null,new File("")), Arrays.asList(""));

    }

}

