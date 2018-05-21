package org.deeplearning4j.spark.models.sequencevectors;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.primitives.Counter;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
public class SparkSequenceVectorsTest {
    protected static List<Sequence<VocabWord>> sequencesCyclic;
    private JavaSparkContext sc;

    @Before
    public void setUp() throws Exception {
        if (sequencesCyclic == null) {
            sequencesCyclic = new ArrayList<>();

            // 10 sequences in total
            for (int с = 0; с < 10; с++) {

                Sequence<VocabWord> sequence = new Sequence<>();

                for (int e = 0; e < 10; e++) {
                    // we will have 9 equal elements, with total frequency of 10
                    sequence.addElement(new VocabWord(1.0, "" + e, (long) e));
                }

                // and 1 element with frequency of 20
                sequence.addElement(new VocabWord(1.0, "0", 0L));
                sequencesCyclic.add(sequence);
            }
        }

        SparkConf sparkConf = new SparkConf().setMaster("local[8]").setAppName("SeqVecTests");
        sc = new JavaSparkContext(sparkConf);
    }

    @After
    public void tearDown() throws Exception {
        sc.stop();
    }

    @Test
    public void testFrequenciesCount() throws Exception {
        JavaRDD<Sequence<VocabWord>> sequences = sc.parallelize(sequencesCyclic);

        SparkSequenceVectors<VocabWord> seqVec = new SparkSequenceVectors<>();

        seqVec.fitSequences(sequences);

        Counter<Long> counter = seqVec.getCounter();

        // element "0" should have frequency of 20
        assertEquals(20, counter.getCount(0L), 1e-5);

        // elements 1 - 9 should have frequencies of 10
        for (int e = 1; e < sequencesCyclic.get(0).getElements().size() - 1; e++) {
            assertEquals(10, counter.getCount(sequencesCyclic.get(0).getElementByIndex(e).getStorageId()), 1e-5);
        }


        VocabCache<ShallowSequenceElement> shallowVocab = seqVec.getShallowVocabCache();

        assertEquals(10, shallowVocab.numWords());

        ShallowSequenceElement zero = shallowVocab.tokenFor(0L);
        ShallowSequenceElement first = shallowVocab.tokenFor(1L);

        assertNotEquals(null, zero);
        assertEquals(20.0, zero.getElementFrequency(), 1e-5);
        assertEquals(0, zero.getIndex());

        assertEquals(10.0, first.getElementFrequency(), 1e-5);
    }

}
