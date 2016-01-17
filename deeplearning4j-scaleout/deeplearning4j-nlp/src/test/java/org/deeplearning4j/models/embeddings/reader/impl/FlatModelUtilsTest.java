package org.deeplearning4j.models.embeddings.reader.impl;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

import static org.junit.Assert.*;

/**
 * These are temporary tests and will be removed after issue is solved.
 *
 *
 *
 * @author raver119@gmail.com
 */
public class FlatModelUtilsTest {
    private Word2Vec vec;
    private static final Logger log = LoggerFactory.getLogger(FlatModelUtilsTest.class);

    @Before
    public void setUp() throws Exception {
        if (vec == null) {
            vec = WordVectorSerializer.loadFullModel("/Users/raver119/develop/model.dat");
        }
    }

    @Test
    public void testWordsNearestFlat1() throws Exception {
        vec.setModelUtils(new FlatModelUtils<VocabWord>());

        Collection<String> list = vec.wordsNearest("energy",10);
        System.out.println("energy: " + list);
    }

    @Test
    public void testWordsNearestBasic1() throws Exception {
        vec.setModelUtils(new BasicModelUtils<VocabWord>());

        Collection<String> list = vec.wordsNearest("energy",10);
        System.out.println("energy: " + list);
    }

    @Test
    public void testWordsNearestTree1() throws Exception {
        vec.setModelUtils(new TreeModelUtils<VocabWord>());

        Collection<String> list = vec.wordsNearest("energy",10);
        System.out.println("energy: " + list);
    }
}