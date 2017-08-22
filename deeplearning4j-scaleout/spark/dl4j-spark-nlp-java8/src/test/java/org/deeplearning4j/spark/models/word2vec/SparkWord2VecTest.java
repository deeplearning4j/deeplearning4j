package org.deeplearning4j.spark.models.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Tests for new Spark Word2Vec implementation
 *
 * @author raver119@gmail.com
 */
public class SparkWord2VecTest {
    private static List<String> sentences;
    private JavaSparkContext sc;

    @Before
    public void setUp() throws Exception {
        if (sentences == null) {
            sentences = new ArrayList<>();

            sentences.add("one two thee four");
            sentences.add("some once again");
            sentences.add("one another sentence");
        }

        SparkConf sparkConf = new SparkConf().setMaster("local[8]").setAppName("SeqVecTests");
        sc = new JavaSparkContext(sparkConf);
    }

    @After
    public void tearDown() throws Exception {
        sc.stop();
    }

    @Test
    public void testStringsTokenization1() throws Exception {
        JavaRDD<String> rddSentences = sc.parallelize(sentences);

        SparkWord2Vec word2Vec = new SparkWord2Vec();

        word2Vec.fitSentences(rddSentences);

        VocabCache<ShallowSequenceElement> vocabCache = word2Vec.getShallowVocabCache();

        assertNotEquals(null, vocabCache);

        assertEquals(9, vocabCache.numWords());
        assertEquals(2.0, vocabCache.wordFor(SequenceElement.getLongHash("one")).getElementFrequency(), 1e-5);
        assertEquals(1.0, vocabCache.wordFor(SequenceElement.getLongHash("two")).getElementFrequency(), 1e-5);
    }
}
