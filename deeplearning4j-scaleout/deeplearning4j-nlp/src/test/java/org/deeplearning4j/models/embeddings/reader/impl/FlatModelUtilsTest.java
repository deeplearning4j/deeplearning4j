package org.deeplearning4j.models.embeddings.reader.impl;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

/**
 * These are temporary tests and will be removed after issue is solved.
 *
 *
 *
 * @author raver119@gmail.com
 */
@Ignore
public class FlatModelUtilsTest {
    private Word2Vec vec;
    private static final Logger log = LoggerFactory.getLogger(FlatModelUtilsTest.class);

    @Before
    public void setUp() throws Exception {
        if (vec == null) {
            //vec = WordVectorSerializer.loadFullModel("/Users/raver119/develop/model.dat");
            vec = WordVectorSerializer.loadFullModel("/ext/Temp/Models/model.dat");
            //vec = WordVectorSerializer.loadFullModel("/ext/Temp/Models/raw_sentences.dat");
        }
    }

    @Test
    public void testWordsNearestFlat1() throws Exception {
        vec.setModelUtils(new FlatModelUtils<VocabWord>());

        Collection<String> list = vec.wordsNearest("energy",10);
        log.info("Flat model results:");
        printWords("energy", list, vec);
    }

    @Test
    public void testWordsNearestBasic1() throws Exception {

        WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("/ext/Temp/Models/model.dat_trans"));
        vec.setModelUtils(new BasicModelUtils<VocabWord>());

        String target = "energy";

        INDArray arr1 = vec.getWordVectorMatrix(target).dup();

        System.out.println("[-]: " + arr1);
        System.out.println("[+]: " + Transforms.unitVec(arr1));

        Collection<String> list = vec.wordsNearest(target,10);
        log.info("Transpose model results:");
        printWords(target, list, vec);

        list = vec.wordsNearest(target,10);
        log.info("Transpose model results 2:");
        printWords(target, list, vec);

        list = vec.wordsNearest(target,10);
        log.info("Transpose model results 3:");
        printWords(target, list, vec);


        INDArray arr2 = vec.getWordVectorMatrix(target).dup();

        assertEquals(arr1, arr2);
    }

    @Test
    @Ignore
    public void testWordsNearestTree1() throws Exception {
        vec.setModelUtils(new TreeModelUtils<VocabWord>());

        Collection<String> list = vec.wordsNearest("energy",10);
        log.info("Tree model results:");
        printWords("energy", list, vec);
    }

    private static void printWords(String target, Collection<String> list, WordVectors vec) {
        System.out.println("Words close to ["+target+"]:");
        for (String word: list) {
            double sim = vec.similarity(target, word);
            System.out.print("'"+ word+"': ["+ sim+"]");
        }
        System.out.print("\n");
    }
}