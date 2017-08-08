package org.deeplearning4j.spark.models.sequencevectors.export;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class ExportContainerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testToString() throws Exception {
        ExportContainer<VocabWord> container =
                        new ExportContainer<>(new VocabWord(1.0, "word"), Nd4j.create(new double[] {1.01, 2.01, 3.01}));
        String exp = "word 1.01 2.01 3.01";
        String string = container.toString();

        assertEquals(exp, string);
    }

}
